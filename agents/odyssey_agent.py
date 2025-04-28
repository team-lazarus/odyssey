import csv
import logging
import os
import torch
import numpy as np
import yaml
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Deque
from collections import deque
from datetime import datetime
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.console import Console
from odyssey.src.environment import OdysseyEnvironment
from daedalus.agentsv2.agent_ppo import PPOTrainer


class OdysseyAgent:
    def __init__(
        self,
        theseus_agent: Any,
        daedalus_agent: Any,  # Now required, no default
        env: Optional[OdysseyEnvironment] = None,
        log_window_size: int = 100,
        save_interval: int = 500,
        save_dir: str = "odyssey_model_saves",
        agent_type: str = "PPO",
    ) -> None:
        """
        theseus_agent: The Theseus agent for hero/gun control.
        daedalus_agent: The Daedalus agent for map generation. (REQUIRED)
        env: The OdysseyEnvironment instance.
            If None, a new OdysseyEnvironment will be created.
        log_window_size: Size of the rolling window for metric averaging.
        save_interval: Number of episodes between model checkpoints.
        save_dir: Directory to save model checkpoints.
        agent_type: Type of Theseus agent being used ("DQN" or "PPO").
        """
        self.logger = logging.getLogger("odyssey-agent")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize environment
        self.env = env if env is not None else OdysseyEnvironment()

        # Initialize agents
        self.theseus_agent = theseus_agent
        self.daedalus_agent = daedalus_agent  # Now always required

        # Store agent type for method selection
        self.agent_type = agent_type

        # Training parameters
        self.log_window_size = log_window_size
        self.save_interval = save_interval
        self.save_dir = save_dir

        # Initialize metrics tracking
        self.episodes_completed = 0
        self._initialize_metrics()

        self.console = Console()

        self.logger.info(f"OdysseyAgent initialized on {self.device}")
        self.logger.info(
            f"Theseus Agent: {type(self.theseus_agent).__name__} (Type: {self.agent_type})"
        )
        self.logger.info(f"Daedalus Agent: {type(self.daedalus_agent).__name__}")

    def _initialize_metrics(self) -> None:
        """Initialize metrics tracking deques."""
        self.metrics_deque: Dict[str, Deque] = {
            "hero_reward": deque(maxlen=self.log_window_size),
            "gun_reward": deque(maxlen=self.log_window_size),
            "daedalus_reward": deque(maxlen=self.log_window_size),
            "episode_length": deque(maxlen=self.log_window_size),
            "wave_clear_rate": deque(maxlen=self.log_window_size),
            "hero_loss": deque(maxlen=self.log_window_size * 10),
            "gun_loss": deque(maxlen=self.log_window_size * 10),
            "daedalus_policy_loss": deque(maxlen=self.log_window_size * 10),
            "daedalus_value_loss": deque(maxlen=self.log_window_size * 10),
            "daedalus_entropy": deque(maxlen=self.log_window_size * 10),
        }

        # Cumulative metrics
        self.total_metrics = {
            "hero_reward": 0.0,
            "gun_reward": 0.0,
            "daedalus_reward": 0.0,
            "waves_cleared": 0,
            "episodes": 0,
        }

        # Store full training data for summary
        self.training_summary_data: List[Dict[str, Any]] = []

    def _update_metrics(
        self, episode_summary: Dict[str, Any], daedalus_reward: float
    ) -> None:
        """
        Update metrics tracking with episode results for both Theseus and Daedalus agents.
        Args:
            episode_summary: Dictionary containing episode summary from the environment.
            daedalus_reward: The reward calculated for the Daedalus agent.
        """
        # Update rolling metrics for basic data
        self.metrics_deque["hero_reward"].append(episode_summary["total_hero_reward"])
        self.metrics_deque["gun_reward"].append(episode_summary["total_gun_reward"])
        self.metrics_deque["daedalus_reward"].append(daedalus_reward)
        self.metrics_deque["episode_length"].append(episode_summary["episode_length"])
        self.metrics_deque["wave_clear_rate"].append(
            1.0 if episode_summary["wave_clear"] else 0.0
        )

        # Update PPO-specific metrics if available from Daedalus agent
        if hasattr(self.daedalus_agent, "metrics"):
            # If using PPOTrainer, extract metrics from its state
            daedalus_metrics = self.daedalus_agent.metrics

            # Get the latest metrics if available
            if (
                daedalus_metrics.get("all_policy_losses")
                and daedalus_metrics["all_policy_losses"]
            ):
                latest_policy_losses = daedalus_metrics["all_policy_losses"][-1]
                if latest_policy_losses:
                    avg_policy_loss = np.mean(latest_policy_losses)
                    self.metrics_deque["daedalus_policy_loss"].append(avg_policy_loss)

            if (
                daedalus_metrics.get("all_value_losses")
                and daedalus_metrics["all_value_losses"]
            ):
                latest_value_losses = daedalus_metrics["all_value_losses"][-1]
                if latest_value_losses:
                    avg_value_loss = np.mean(latest_value_losses)
                    self.metrics_deque["daedalus_value_loss"].append(avg_value_loss)

            if (
                daedalus_metrics.get("all_entropies")
                and daedalus_metrics["all_entropies"]
            ):
                latest_entropies = daedalus_metrics["all_entropies"][-1]
                if latest_entropies:
                    avg_entropy = np.mean(latest_entropies)
                    self.metrics_deque["daedalus_entropy"].append(avg_entropy)

        # Update cumulative metrics
        self.total_metrics["hero_reward"] += episode_summary["total_hero_reward"]
        self.total_metrics["gun_reward"] += episode_summary["total_gun_reward"]
        self.total_metrics["daedalus_reward"] += daedalus_reward
        self.total_metrics["waves_cleared"] += 1 if episode_summary["wave_clear"] else 0
        self.total_metrics["episodes"] += 1

        # Store for summary table with data
        episode_data = {
            "episode": self.total_metrics["episodes"],
            "hero_reward": episode_summary["total_hero_reward"],
            "gun_reward": episode_summary["total_gun_reward"],
            "daedalus_reward": daedalus_reward,
            "episode_length": episode_summary["episode_length"],
            "wave_clear": episode_summary["wave_clear"],
            "terminated": episode_summary["terminated"],
        }

        # Add PPO-specific metrics
        if (
            hasattr(self.metrics_deque.get("daedalus_policy_loss", []), "__len__")
            and len(self.metrics_deque["daedalus_policy_loss"]) > 0
        ):
            episode_data["daedalus_policy_loss"] = self.metrics_deque[
                "daedalus_policy_loss"
            ][-1]

        if (
            hasattr(self.metrics_deque.get("daedalus_value_loss", []), "__len__")
            and len(self.metrics_deque["daedalus_value_loss"]) > 0
        ):
            episode_data["daedalus_value_loss"] = self.metrics_deque[
                "daedalus_value_loss"
            ][-1]

        if (
            hasattr(self.metrics_deque.get("daedalus_entropy", []), "__len__")
            and len(self.metrics_deque["daedalus_entropy"]) > 0
        ):
            episode_data["daedalus_entropy"] = self.metrics_deque["daedalus_entropy"][
                -1
            ]

        self.training_summary_data.append(episode_data)

    def _calculate_daedalus_reward(self, episode_summary: Dict[str, Any]) -> float:
        """
        Calculate reward for the Daedalus agent (map generator) based on hero survival and wave completion.
        Thresholds are based on real elapsed time for the episode.
        """
        # Extract episode details
        episode_length = episode_summary["episode_length"]
        wave_clear = episode_summary.get("wave_clear", False)
        terminated = episode_summary.get("terminated", False)

        start_time = episode_summary.get("start_time")
        end_time = episode_summary.get("end_time")
        if start_time is not None and end_time is not None:
            elapsed = end_time - start_time
        else:
            elapsed = episode_length  # fallback

        min_acceptable_time = elapsed * 0.3
        min_optimal_time = elapsed * 0.5
        max_optimal_time = elapsed * 0.8
        max_acceptable_time = elapsed * 1.0

        # Base reward calculation
        if wave_clear:
            if episode_length < min_acceptable_time:
                # Wave cleared way too quickly - penalize (map is trivial)
                base_reward = -5.0
            elif episode_length < min_optimal_time:
                # Wave cleared somewhat too quickly - mild penalty transitioning to reward
                normalized_time = (episode_length - min_acceptable_time) / (
                    min_optimal_time - min_acceptable_time
                )
                base_reward = -3.0 + (6.0 * normalized_time)  # Ranges from -3 to +3
            elif episode_length <= max_optimal_time:
                # Wave cleared in optimal time - excellent!
                normalized_time = (episode_length - min_optimal_time) / (
                    max_optimal_time - min_optimal_time
                )
                # Parabolic reward with maximum at normalized_time = 0.5
                base_reward = 7.0 * (1.0 - 4.0 * (normalized_time - 0.5) ** 2)
            else:
                # Wave cleared but took too long - still good but diminishing returns
                excess_time = (episode_length - max_optimal_time) / (
                    max_acceptable_time - max_optimal_time
                )
                base_reward = 5.0 * (1.0 - excess_time**2)
        elif terminated:
            # Hero died (episode terminated without wave clear)
            if episode_length < min_acceptable_time:
                # Died very quickly - map is too difficult
                base_reward = -8.0
            elif episode_length < min_optimal_time:
                # Died fairly early - map is somewhat difficult
                normalized_time = (episode_length - min_acceptable_time) / (
                    min_optimal_time - min_acceptable_time
                )
                base_reward = -6.0 + (3.0 * normalized_time)  # Ranges from -6 to -3
            elif episode_length <= max_optimal_time:
                # Died within optimal time range - neutral to slightly negative
                normalized_time = (episode_length - min_optimal_time) / (
                    max_optimal_time - min_optimal_time
                )
                base_reward = -3.0 + (2.0 * normalized_time)  # Ranges from -3 to -1
            else:
                # Died after surviving a long time - slightly negative
                base_reward = -1.0
        else:
            # Episode ended without wave clear or termination (likely truncated/timed out)
            if episode_length > max_optimal_time:
                # Survived a long time but didn't clear - map might be too easy or boring
                excess_time = min(
                    1.0,
                    (episode_length - max_optimal_time)
                    / (max_acceptable_time - max_optimal_time),
                )
                base_reward = -2.0 * excess_time**2  # Increasingly negative
            else:
                # Hard to evaluate - slightly negative
                base_reward = -1.0

        # Balance component - reward maps that require both combat and resource gathering
        balance_component = 0.0
        if (
            "enemies_killed" in episode_summary
            and "resources_collected" in episode_summary
        ):
            enemies_killed = episode_summary["enemies_killed"]
            resources_collected = episode_summary["resources_collected"]
            if enemies_killed > 0 and resources_collected > 0:
                # Ratio between the smaller and larger value (closer to 1 is better balanced)
                balance_ratio = min(enemies_killed, resources_collected) / max(
                    enemies_killed, resources_collected
                )
                balance_component = 3.0 * balance_ratio

        daedalus_reward = base_reward + balance_component
        return np.clip(daedalus_reward, -10.0, 10.0)

    def _log_episode_metrics(self, episode: int) -> None:
        """
        Log metrics for the completed episode for both Theseus and Daedalus agents.
        """
        # Calculate rolling averages for standard metrics
        avg_hero_reward = (
            np.mean(self.metrics_deque["hero_reward"])
            if self.metrics_deque["hero_reward"]
            else 0.0
        )
        avg_gun_reward = (
            np.mean(self.metrics_deque["gun_reward"])
            if self.metrics_deque["gun_reward"]
            else 0.0
        )
        avg_daedalus_reward = (
            np.mean(self.metrics_deque["daedalus_reward"])
            if self.metrics_deque["daedalus_reward"]
            else 0.0
        )
        avg_episode_length = (
            np.mean(self.metrics_deque["episode_length"])
            if self.metrics_deque["episode_length"]
            else 0.0
        )
        avg_wave_clear = (
            np.mean(self.metrics_deque["wave_clear_rate"])
            if self.metrics_deque["wave_clear_rate"]
            else 0.0
        )

        # Calculate PPO-specific metrics for Daedalus if available
        avg_policy_loss = (
            np.mean(self.metrics_deque["daedalus_policy_loss"])
            if self.metrics_deque.get("daedalus_policy_loss")
            and len(self.metrics_deque["daedalus_policy_loss"]) > 0
            else None
        )
        avg_value_loss = (
            np.mean(self.metrics_deque["daedalus_value_loss"])
            if self.metrics_deque.get("daedalus_value_loss")
            and len(self.metrics_deque["daedalus_value_loss"]) > 0
            else None
        )
        avg_entropy = (
            np.mean(self.metrics_deque["daedalus_entropy"])
            if self.metrics_deque.get("daedalus_entropy")
            and len(self.metrics_deque["daedalus_entropy"]) > 0
            else None
        )

        # Format basic metrics for logging
        metrics_list = [
            f"AvgLength={avg_episode_length:.2f}",
            f"AvgR_Hero={avg_hero_reward:.3f}",
            f"AvgR_Gun={avg_gun_reward:.3f}",
            f"AvgR_Daedalus={avg_daedalus_reward:.3f}",
            f"WaveClearRate={avg_wave_clear:.2f}",
            f"TotalWavesCleared={self.total_metrics['waves_cleared']}",
        ]

        # Add Theseus-specific metrics if available
        if hasattr(self.theseus_agent, "epsilon"):
            metrics_list.append(f"Epsilon={self.theseus_agent.epsilon:.4f}")

        # Add Daedalus PPO-specific metrics if available
        if avg_policy_loss is not None:
            metrics_list.append(f"DPolicyLoss={avg_policy_loss:.4f}")
        if avg_value_loss is not None:
            metrics_list.append(f"DValueLoss={avg_value_loss:.4f}")
        if avg_entropy is not None:
            metrics_list.append(f"DEntropy={avg_entropy:.4f}")

        # Check for additional metrics from Daedalus PPO agent
        if hasattr(self.daedalus_agent, "actor_optimizer") and hasattr(
            self.daedalus_agent.actor_optimizer, "param_groups"
        ):
            metrics_list.append(
                f"DLR={self.daedalus_agent.actor_optimizer.param_groups[0]['lr']:.6f}"
            )

        log_str = f"Episode {episode} Summary | " + " | ".join(metrics_list)
        self.logger.info(log_str)

        # Log detailed metrics for Daedalus every 10 episodes
        if episode % 10 == 0 and (
            avg_policy_loss is not None or avg_value_loss is not None
        ):
            detailed_metrics = [
                f"Daedalus PPO Metrics (Episode {episode}):",
                f"  Policy Loss: {avg_policy_loss if avg_policy_loss is not None else 'N/A'}",
                f"  Value Loss: {avg_value_loss if avg_value_loss is not None else 'N/A'}",
                f"  Entropy: {avg_entropy if avg_entropy is not None else 'N/A'}",
            ]

            # Add environment details if available
            if hasattr(self.daedalus_agent, "env") and hasattr(
                self.daedalus_agent.env, "batch_size"
            ):
                detailed_metrics.append(
                    f"  Batch Size: {self.daedalus_agent.env.batch_size}"
                )

            self.logger.info("\n".join(detailed_metrics))

    def _save_checkpoint_if_needed(self, episode: int) -> None:
        """
        Save agent checkpoints if the save interval is reached.
        """
        if (
            self.save_interval > 0
            and episode > 0
            and (episode + 1) % self.save_interval == 0
        ):
            self.logger.info(
                f"Reached save interval at episode {episode}. Saving checkpoint..."
            )
            save_path = self.save_checkpoint()
            if save_path:
                self.logger.info(f"Checkpoint saved successfully to: {save_path}")
            else:
                self.logger.error(f"Failed to save checkpoint for episode {episode}.")

    def _display_training_summary(self, completed_episodes: int) -> None:
        """
        Display a summary table of training performance.
        """
        if not self.training_summary_data or completed_episodes == 0:
            self.logger.info(
                "No training data recorded or no episodes completed, skipping summary."
            )
            return

        self.logger.info("Generating Training Summary Table...")

        # Define blocks (e.g., 10% of total completed episodes)
        block_size = max(1, completed_episodes // 10)  # Ensure block_size is at least 1
        num_blocks = (
            completed_episodes + block_size - 1
        ) // block_size  # Ceiling division

        summary_rows = []
        for i in range(num_blocks):
            start_episode = i * block_size + 1
            end_episode = min((i + 1) * block_size, completed_episodes)

            # Extract block data
            block_data = [
                entry
                for entry in self.training_summary_data
                if start_episode <= entry["episode"] <= end_episode
            ]

            if not block_data:
                continue

            # Calculate averages
            avg_hero_reward = np.mean([entry["hero_reward"] for entry in block_data])
            avg_gun_reward = np.mean([entry["gun_reward"] for entry in block_data])
            avg_daedalus_reward = np.mean(
                [entry["daedalus_reward"] for entry in block_data]
            )
            avg_episode_length = np.mean(
                [entry["episode_length"] for entry in block_data]
            )
            wave_clear_rate = np.mean(
                [1.0 if entry["wave_clear"] else 0.0 for entry in block_data]
            )

            episode_range = f"{start_episode}-{end_episode}"
            summary_rows.append(
                (
                    episode_range,
                    f"{avg_hero_reward:.3f}",
                    f"{avg_gun_reward:.3f}",
                    f"{avg_daedalus_reward:.3f}",
                    f"{avg_episode_length:.2f}",
                    f"{wave_clear_rate:.2f}",
                )
            )

        # Create and print the table using rich
        table = Table(
            title=f"Odyssey Training Summary (Completed {completed_episodes} Episodes)"
        )
        table.add_column("Episode Block", justify="center", style="cyan", no_wrap=True)
        table.add_column("Avg Hero Reward", justify="right", style="magenta")
        table.add_column("Avg Gun Reward", justify="right", style="green")
        table.add_column("Avg Daedalus Reward", justify="right", style="blue")
        table.add_column("Avg Episode Length", justify="right", style="yellow")
        table.add_column("Wave Clear Rate", justify="right", style="red")

        for row in summary_rows:
            table.add_row(*row)

        self.console.print(table)

    def save_checkpoint(self, custom_dir: Optional[str] = None) -> Optional[str]:
        """
        Save the complete agent state to a timestamped directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"odyssey_agent_{timestamp}"
        save_dir = custom_dir if custom_dir is not None else self.save_dir
        dpath = os.path.join(save_dir, base_name)

        try:
            os.makedirs(dpath, exist_ok=True)
            self.logger.info(f"Saving agent state to directory: {dpath}")

            # Save Theseus agent if it has a dump method
            theseus_path = None
            if hasattr(self.theseus_agent, "dump"):
                theseus_subdir = os.path.join(dpath, "theseus")
                os.makedirs(theseus_subdir, exist_ok=True)
                theseus_path = self.theseus_agent.dump(save_dir=theseus_subdir)
                self.logger.info(f"Theseus agent saved to: {theseus_path}")

            # Save Daedalus agent - handle PPO agent specifically
            daedalus_path = None
            daedalus_subdir = os.path.join(dpath, "daedalus")
            os.makedirs(daedalus_subdir, exist_ok=True)

            if isinstance(self.daedalus_agent, PPOTrainer):
                # Use PPO-specific checkpoint saving
                checkpoint_filename = f"checkpoint_episode_{self.episodes_completed}.pt"
                daedalus_path = os.path.join(daedalus_subdir, checkpoint_filename)

                # Create PPO checkpoint dictionary
                checkpoint = {
                    "actor_state_dict": self.daedalus_agent.actor.state_dict()
                    if hasattr(self.daedalus_agent, "actor")
                    else None,
                    "critic_state_dict": self.daedalus_agent.critic.state_dict()
                    if hasattr(self.daedalus_agent, "critic")
                    else None,
                    "actor_optimizer_state_dict": self.daedalus_agent.actor_optimizer.state_dict()
                    if hasattr(self.daedalus_agent, "actor_optimizer")
                    else None,
                    "critic_optimizer_state_dict": self.daedalus_agent.critic_optimizer.state_dict()
                    if hasattr(self.daedalus_agent, "critic_optimizer")
                    else None,
                    "episode": self.episodes_completed,
                    "metrics": self.daedalus_agent.metrics
                    if hasattr(self.daedalus_agent, "metrics")
                    else {},
                }

                # Save the checkpoint
                torch.save(checkpoint, daedalus_path)
                self.logger.info(f"Daedalus PPO agent saved to: {daedalus_path}")

            elif hasattr(self.daedalus_agent, "_save_checkpoint"):
                # If agent has its own save method, use that
                try:
                    # Some implementations expect episode number
                    self.daedalus_agent._save_checkpoint(self.episodes_completed)
                    daedalus_path = os.path.join(
                        self.daedalus_agent.checkpoint_dir,
                        f"checkpoint_episode_{self.episodes_completed}.pt",
                    )
                    self.logger.info(
                        f"Daedalus agent saved using its internal _save_checkpoint method"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to save Daedalus agent using its _save_checkpoint method: {e}"
                    )
                    # Fallback to dump method if available
                    if hasattr(self.daedalus_agent, "dump"):
                        daedalus_path = self.daedalus_agent.dump(
                            save_dir=daedalus_subdir
                        )
                        self.logger.info(
                            f"Daedalus agent saved using fallback dump method: {daedalus_path}"
                        )

            elif hasattr(self.daedalus_agent, "dump"):
                # Legacy method
                daedalus_path = self.daedalus_agent.dump(save_dir=daedalus_subdir)
                self.logger.info(
                    f"Daedalus agent saved using dump method: {daedalus_path}"
                )

            else:
                self.logger.warning(
                    "No known save method found for Daedalus agent, saving metadata only"
                )

            # Save metrics data for Daedalus in CSV format
            metrics_path = os.path.join(daedalus_subdir, "training_metrics.csv")
            try:
                with open(metrics_path, "w", newline="") as csvfile:
                    fieldnames = [
                        "episode",
                        "reward",
                        "policy_loss",
                        "value_loss",
                        "entropy",
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    # Get metrics history length
                    num_entries = (
                        min(
                            len(self.metrics_deque["daedalus_reward"]),
                            len(self.metrics_deque.get("daedalus_policy_loss", [])),
                            len(self.metrics_deque.get("daedalus_value_loss", [])),
                            len(self.metrics_deque.get("daedalus_entropy", [])),
                        )
                        if self.metrics_deque.get("daedalus_policy_loss")
                        else len(self.metrics_deque["daedalus_reward"])
                    )

                    # Write recent metrics history
                    for i in range(num_entries):
                        row = {
                            "episode": self.episodes_completed - num_entries + i + 1,
                            "reward": list(self.metrics_deque["daedalus_reward"])[i],
                        }

                        # Add PPO metrics if available
                        if self.metrics_deque.get("daedalus_policy_loss"):
                            row["policy_loss"] = list(
                                self.metrics_deque["daedalus_policy_loss"]
                            )[i]
                        if self.metrics_deque.get("daedalus_value_loss"):
                            row["value_loss"] = list(
                                self.metrics_deque["daedalus_value_loss"]
                            )[i]
                        if self.metrics_deque.get("daedalus_entropy"):
                            row["entropy"] = list(
                                self.metrics_deque["daedalus_entropy"]
                            )[i]

                        writer.writerow(row)

                self.logger.info(f"Daedalus metrics saved to: {metrics_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save Daedalus metrics CSV: {e}")

            # Save Odyssey Agent's state and configuration
            config_path = os.path.join(dpath, f"{base_name}_config.yaml")
            config_data = {
                "timestamp": timestamp,
                "episodes_completed": self.episodes_completed,
                "total_metrics": self.total_metrics,
                "theseus_path": theseus_path,
                "daedalus_path": daedalus_path,
                "theseus_agent_class": f"{self.theseus_agent.__class__.__module__}.{self.theseus_agent.__class__.__name__}",
                "daedalus_agent_class": f"{self.daedalus_agent.__class__.__module__}.{self.daedalus_agent.__class__.__name__}",
                "log_window_size": self.log_window_size,
                "save_interval": self.save_interval,
                "device": str(self.device),
                "agent_type": self.agent_type,
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

            self.logger.info(f"Config saved to: {config_path}")
            self.logger.info("Agent state saved successfully.")
            return dpath

        except Exception as e:
            self.logger.error(f"Failed to save agent state: {e}", exc_info=True)
            return None

    def train(
        self,
        num_episodes: Optional[int] = None,
        hero_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        self.logger.info(
            f"Starting Odyssey training on {self.device} for {num_episodes or 'infinite'} episodes..."
        )
        self.training_summary_data = []  # Reset summary data

        # Configure progress display
        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
        ]

        # Adjust for infinite training
        if num_episodes is None:
            progress_columns = [
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("Episode {task.completed}"),
            ]
            self.logger.warning(
                "Training infinitely (num_episodes=None). Progress bar will not show total or ETA."
            )

        with Progress(*progress_columns, transient=False) as progress:
            episode_task = progress.add_task(
                "[cyan]Training Episodes...", total=num_episodes
            )

            # Use count() for infinite iteration like in agent classes
            from itertools import count

            episode_iterator = (
                range(num_episodes) if num_episodes is not None else count()
            )
            completed_episodes = 0

            try:
                for episode in episode_iterator:
                    try:
                        # Update progress description
                        progress.update(
                            episode_task,
                            description=f"[cyan]Episode {episode+1}[/cyan]",
                        )

                        # Generate map using Daedalus
                        map_data = self.daedalus_agent(hero_tensor)

                        # Initialize environment with the map
                        state = self.env.initialise_environment(map_data)

                        # Initialize episode variables
                        terminated = False
                        truncated = False
                        wave_clear = False
                        total_hero_reward = 0
                        total_gun_reward = 0
                        time_alive = 0

                        # Run episode until termination or wave clear
                        # Note: The actual episode logic is already handled in _run_episode or _run_episode_or_rollout
                        # These methods contain their own inner loop that runs until termination
                        if self.agent_type == "PPO":
                            # PPO-specific handling
                            (
                                hero_reward,
                                gun_reward,
                                time_alive,
                                terminated,
                                truncated,
                            ) = self.theseus_agent._run_episode_or_rollout(
                                episode, progress, episode_task
                            )

                            # Get full episode summary
                            episode_summary = self.env.get_episode_summary()
                            wave_clear = episode_summary.get("wave_clear", False)

                        else:  # DQN
                            # DQN-specific handling
                            (
                                hero_reward,
                                gun_reward,
                                time_alive,
                            ) = self.theseus_agent._run_episode(
                                episode, progress, episode_task
                            )

                            # Get full episode summary
                            episode_summary = self.env.get_episode_summary()
                            terminated = episode_summary.get("terminated", False)
                            truncated = episode_summary.get("truncated", False)
                            wave_clear = episode_summary.get("wave_clear", False)

                            # For DQN, call learn and decay epsilon after each episode
                            self.theseus_agent._learn()
                            self.theseus_agent._decay_epsilon()

                        # Store training data
                        self.training_summary_data.append(
                            {
                                "episode": episode + 1,
                                "hero_reward": hero_reward,
                                "gun_reward": gun_reward,
                                "time_alive": time_alive,
                                "wave_clear": wave_clear,
                                "terminated": terminated,
                            }
                        )

                        # Calculate reward for Daedalus
                        daedalus_reward = self._calculate_daedalus_reward(
                            episode_summary
                        )

                        # Update Daedalus agent
                        self.env.update_daedalus(episode_summary)
                        self.daedalus_agent.update(daedalus_reward)

                        # Update metrics
                        self._update_metrics(episode_summary, daedalus_reward)
                        self._log_episode_metrics(episode)

                        # Save checkpoint if needed
                        self._save_checkpoint_if_needed(episode + 1)

                        # Update progress
                        progress.update(episode_task, advance=1)
                        completed_episodes += 1
                        self.episodes_completed += 1

                    except RuntimeError as e:
                        self.logger.critical(
                            f"Stopping training due to runtime error in episode {episode+1}: {e}",
                            exc_info=True,
                        )
                        progress.stop()
                        break
                    except Exception as e:
                        self.logger.critical(
                            f"Unexpected error during episode {episode+1}: {e}",
                            exc_info=True,
                        )
                        progress.stop()
                        break

            finally:
                if num_episodes is not None:
                    final_desc = (
                        "[green]Training Finished"
                        if completed_episodes == num_episodes
                        else "[yellow]Training Stopped Early"
                    )
                    progress.update(
                        episode_task,
                        description=final_desc,
                        completed=completed_episodes,
                    )
                    # Display summary
                    self._display_training_summary(completed_episodes)

                    # Save final metrics CSV if the method exists
                    if hasattr(self, "_save_episode_metrics_csv"):
                        self.logger.info(
                            "Attempting to save final episode metrics CSV..."
                        )
                        last_save_dir = (
                            self._get_last_save_directory()
                            if hasattr(self, "_get_last_save_directory")
                            else None
                        )
                        if last_save_dir:
                            self._save_episode_metrics_csv(last_save_dir)
                            self.logger.info(
                                f"Final metrics CSV saved in: {last_save_dir}"
                            )
                        else:
                            final_save_dir = os.path.join(
                                "model_saves", "final_run_metrics"
                            )
                            os.makedirs(final_save_dir, exist_ok=True)
                            self._save_episode_metrics_csv(final_save_dir)
                            self.logger.warning(
                                f"No checkpoint directory found. Final metrics saved to: {final_save_dir}"
                            )
                else:
                    progress.update(
                        episode_task,
                        description="[yellow]Training Stopped (Infinite Mode)",
                    )

        self.logger.info("Odyssey training finished.")


if __name__ == "__main__":
    # Setup logging
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO",
        format=FORMAT,
        datefmt="[%X]",
        handlers=[
            logging.FileHandler(rich_tracebacks=True, show_path=False),
            logging.FileHandler("odyssey_training.log"),
        ],
    )

    logging.getLogger("torch_geometric").setLevel(logging.WARNING)

    logger = logging.getLogger("odyssey_main")
    logger.info(
        "[bold green]Starting Odyssey Training System[/]", extra={"markup": True}
    )

    try:
        # Parse command-line arguments
        import argparse

        parser = argparse.ArgumentParser(description="Odyssey Training System")
        parser.add_argument(
            "--agent",
            type=str,
            choices=["DQN", "PPO"],
            default="PPO",
            help="Type of Theseus agent to use (DQN or PPO)",
        )
        args = parser.parse_args()

        env = OdysseyEnvironment()

        HIDDEN_CHANNELS = 16
        HERO_ACTION_SPACE_SIZE = 9
        GUN_ACTION_SPACE_SIZE = 8

        LEARNING_RATE = 1e-4
        DISCOUNT_FACTOR = 0.99

        # --- PPO Hyperparameters ---
        PPO_LEARNING_RATE = 3e-4
        PPO_DISCOUNT_FACTOR = 0.99
        PPO_HORIZON = 2048
        PPO_EPOCHS_PER_UPDATE = 10
        PPO_MINI_BATCH_SIZE = 64
        PPO_CLIP_EPSILON = 0.2
        PPO_GAE_LAMBDA = 0.95
        PPO_ENTROPY_COEFF = 0.01
        PPO_VF_COEFF = 0.5
        PPO_LOG_WINDOW = 50
        PPO_SAVE_INTERVAL = 200
        NUM_TRAINING_EPISODES = 50000

        # Initialize appropriate Theseus agent based on the argument
        theseus_agent = None
        agent_type = args.agent

        if agent_type == "DQN":
            # Import required models and agent for DQN
            from theseus.agent_gnn import AgentTheseusGNN
            from theseus.models.GraphDQN.ActionGNN import HeroGNN, GunGNN

            # Initialize models for DQN Theseus agent
            hero_policy = HeroGNN(
                hidden_channels=HIDDEN_CHANNELS, out_channels=HERO_ACTION_SPACE_SIZE
            )
            hero_target = HeroGNN(
                hidden_channels=HIDDEN_CHANNELS, out_channels=HERO_ACTION_SPACE_SIZE
            )
            gun_policy = GunGNN(
                hidden_channels=HIDDEN_CHANNELS, out_channels=GUN_ACTION_SPACE_SIZE
            )
            gun_target = GunGNN(
                hidden_channels=HIDDEN_CHANNELS, out_channels=GUN_ACTION_SPACE_SIZE
            )

            # Initialize DQN Theseus agent
            theseus_agent = AgentTheseusGNN(
                hero_policy_net=hero_policy,
                hero_target_net=hero_target,
                gun_policy_net=gun_policy,
                gun_target_net=gun_target,
                env=env,
                learning_rate=LEARNING_RATE,
                discount_factor=DISCOUNT_FACTOR,
            )

            logger.info("Initialized AgentTheseusGNN (DQN)")

        elif agent_type == "PPO":
            # Import required models and agent for PPO
            from theseus.agent_ppo import AgentTheseusPPO
            from theseus.models.GraphDQN.ActionGNN import HeroGNN, GunGNN

            # Initialize models for PPO Theseus agent
            hero_actor = HeroGNN(
                hidden_channels=HIDDEN_CHANNELS, out_channels=HERO_ACTION_SPACE_SIZE
            )
            hero_critic = HeroGNN(hidden_channels=HIDDEN_CHANNELS, out_channels=1)
            gun_actor = GunGNN(
                hidden_channels=HIDDEN_CHANNELS, out_channels=GUN_ACTION_SPACE_SIZE
            )
            gun_critic = GunGNN(hidden_channels=HIDDEN_CHANNELS, out_channels=1)

            # Initialize PPO Theseus agent
            theseus_agent = AgentTheseusPPO(
                hero_actor_net=hero_actor,
                hero_critic_net=hero_critic,
                gun_actor_net=gun_actor,
                gun_critic_net=gun_critic,
                env=env,
                learning_rate=PPO_LEARNING_RATE,
                discount_factor=PPO_DISCOUNT_FACTOR,
                horizon=PPO_HORIZON,
                epochs_per_update=PPO_EPOCHS_PER_UPDATE,
                mini_batch_size=PPO_MINI_BATCH_SIZE,
                clip_epsilon=PPO_CLIP_EPSILON,
                gae_lambda=PPO_GAE_LAMBDA,
                entropy_coeff=PPO_ENTROPY_COEFF,
                vf_coeff=PPO_VF_COEFF,
                log_window_size=PPO_LOG_WINDOW,
                save_interval=PPO_SAVE_INTERVAL,
            )

            logger.info("Initialized AgentTheseusPPO (PPO)")

        from daedalus.agentsv2.agent_ppo import PPOTrainer

        daedalus_agent = PPOTrainer(
            env_mode="TURTLE",
            batch_size=8192 // 64,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.85,
            clip_epsilon=0.2,
            critic_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            ppo_epochs=10,
            device="cuda" if torch.cuda.is_available() else "cpu",
            checkpoint_dir="ppo_daedalus_checkpoints",
            log_dir="ppo_daedalus_logs",
            critic_path="latest_checkpoint.pth",
            map_size=(12, 12),
            steps_per_episode=256,
            update_interval=32,
            num_episodes=50000,
            mini_batch_factor=4,
            log_level=logging.INFO,
            log_window_size=50,
        )

        # Create Odyssey agent with the selected Theseus and Daedalus agents
        odyssey_agent = OdysseyAgent(
            theseus_agent=theseus_agent,
            daedalus_agent=daedalus_agent,
            env=env,
            log_window_size=100,
            save_interval=500,
            agent_type=agent_type,
        )

        # Start training with fixed episode count
        logger.info(f"Starting training for {NUM_TRAINING_EPISODES} episodes")
        odyssey_agent.train(num_episodes=NUM_TRAINING_EPISODES)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error in Odyssey training: {e}", exc_info=True)
    finally:
        logger.info(
            "[bold red]Odyssey training finished or interrupted[/]",
            extra={"markup": True},
        )
