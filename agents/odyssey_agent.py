"""
Odyssey Agent - Integrates Theseus and Daedalus agents for the Odyssey environment.

This module provides the OdysseyAgent class which coordinates the interaction between
a TheseusAgent (for hero/gun control) and a DaedalusAgent (for map generation).
"""

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
    TaskID,
)
from rich.table import Table
from rich.console import Console
from odyssey.src.environment import OdysseyEnvironment


class DummyDaedalusAgent:
    """
    Temporary implementation of Daedalus Agent until the real one is available.

    Generates simple map data for testing the Odyssey environment.
    """

    def __init__(self, map_size: Tuple[int, int] = (12, 12), seed: int = 42):
        """
        Initialize the dummy Daedalus agent.

        Args:
            map_size: Tuple of (height, width) for the map size.
            seed: Random seed for reproducibility.
        """
        self.map_size = map_size
        self.rng = np.random.RandomState(seed)
        self.logger = logging.getLogger("dummy-daedalus-agent")

    def __call__(self, hero_tensor: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Generate a random map with values between 0-7.

        Args:
            hero_tensor: Optional tensor containing hero state information.
                         Currently unused in this dummy implementation.

        Returns:
            A 2D list representing the map with random integers from 0 to 7.
        """
        self.logger.info(f"Generating dummy map of size {self.map_size}")
        map_data = []
        for _ in range(self.map_size[0]):
            row = self.rng.randint(0, 8, size=self.map_size[1]).tolist()
            map_data.append(row)

        # Ensure hero spawn point (typically value 1) exists
        hero_x, hero_y = self.rng.randint(0, self.map_size[0]), self.rng.randint(
            0, self.map_size[1]
        )
        map_data[hero_x][hero_y] = 1

        return map_data

    def update(self, reward: float) -> None:
        """
        Update the Daedalus agent with a reward.

        Args:
            reward: The reward value to update the agent with.
        """
        self.logger.info(f"Dummy Daedalus agent received reward: {reward}")

class OdysseyAgent:
    """
    Agent that coordinates Theseus and Daedalus agents in the Odyssey environment.

    This agent handles the coordination between the Theseus agent (hero/gun control)
    and the Daedalus agent (map generation), while interfacing with the OdysseyEnvironment.

    Attributes:
        env: The OdysseyEnvironment instance.
        theseus_agent: The agent responsible for hero and gun control.
        daedalus_agent: The agent responsible for map generation.
        device: The computation device ('cuda' or 'cpu').
        episodes_completed: Counter for total episodes completed.
        log_window_size: Size of the rolling window for metric averaging.
        metrics_deque: Dictionary of deques for tracking metrics.
    """

    def __init__(
        self,
        theseus_agent: Any,
        daedalus_agent: Optional[Any] = None,
        env: Optional[OdysseyEnvironment] = None,
        log_window_size: int = 100,
        save_interval: int = 500,
        save_dir: str = "odyssey_model_saves",
        agent_type: str = "PPO",
    ) -> None:
        """
        Initialize the Odyssey agent.

        Args:
            theseus_agent: The Theseus agent for hero/gun control.
            daedalus_agent: The Daedalus agent for map generation.
                        If None, a DummyDaedalusAgent will be created.
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
        self.daedalus_agent = (
            daedalus_agent if daedalus_agent is not None else DummyDaedalusAgent()
        )
        
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
        self.logger.info(f"Theseus Agent: {type(self.theseus_agent).__name__} (Type: {self.agent_type})")
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
        Update metrics tracking with episode results.

        Args:
            episode_summary: Dictionary containing episode summary from the environment.
            daedalus_reward: The reward calculated for the Daedalus agent.
        """
        # Update rolling metrics
        self.metrics_deque["hero_reward"].append(episode_summary["total_hero_reward"])
        self.metrics_deque["gun_reward"].append(episode_summary["total_gun_reward"])
        self.metrics_deque["daedalus_reward"].append(daedalus_reward)
        self.metrics_deque["episode_length"].append(episode_summary["episode_length"])
        self.metrics_deque["wave_clear_rate"].append(
            1.0 if episode_summary["wave_clear"] else 0.0
        )

        # Update cumulative metrics
        self.total_metrics["hero_reward"] += episode_summary["total_hero_reward"]
        self.total_metrics["gun_reward"] += episode_summary["total_gun_reward"]
        self.total_metrics["daedalus_reward"] += daedalus_reward
        self.total_metrics["waves_cleared"] += 1 if episode_summary["wave_clear"] else 0
        self.total_metrics["episodes"] += 1

        # Store for summary table
        self.training_summary_data.append(
            {
                "episode": self.total_metrics["episodes"],
                "hero_reward": episode_summary["total_hero_reward"],
                "gun_reward": episode_summary["total_gun_reward"],
                "daedalus_reward": daedalus_reward,
                "episode_length": episode_summary["episode_length"],
                "wave_clear": episode_summary["wave_clear"],
                "terminated": episode_summary["terminated"],
            }
        )

    def _calculate_daedalus_reward(self, episode_summary: Dict[str, Any]) -> float:
        """
        Calculate the reward for the Daedalus agent based on the episode outcome.

        A good map should:
        1. Allow the hero to survive for a reasonable duration
        2. Provide opportunities for high rewards
        3. Be challenging but not impossible (wave_clear is a good indicator)

        Args:
            episode_summary: Dictionary containing episode summary from the environment.

        Returns:
            The calculated reward for the Daedalus agent.
        """
        # Base reward is the sum of hero and gun rewards (normalized)
        base_reward = (
            episode_summary["total_hero_reward"] + episode_summary["total_gun_reward"]
        ) / 100.0

        # Bonus for wave clear
        wave_clear_bonus = 5.0 if episode_summary["wave_clear"] else 0.0

        # Penalty for early termination (hero death)
        early_termination_penalty = (
            -3.0
            if (episode_summary["terminated"] and not episode_summary["wave_clear"])
            else 0.0
        )

        # Episode length factor: reward longer episodes but with diminishing returns
        # Maximum reward for this component at ~500 steps
        episode_length = episode_summary["episode_length"]
        length_factor = min(1.0, episode_length / 500.0) * 2.0

        # Calculate final reward
        daedalus_reward = (
            base_reward + wave_clear_bonus + early_termination_penalty + length_factor
        )

        return daedalus_reward

    def _log_episode_metrics(self, episode: int) -> None:
        """
        Log metrics for the completed episode.

        Args:
            episode: The current episode number.
        """
        # Calculate rolling averages
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

        # Format metrics for logging
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

        log_str = f"Episode {episode} Summary | " + " | ".join(metrics_list)
        self.logger.info(log_str)

    def _save_checkpoint_if_needed(self, episode: int) -> None:
        """
        Save agent checkpoints if the save interval is reached.

        Args:
            episode: The current episode number.
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

        Args:
            completed_episodes: The total number of completed episodes.
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

        Args:
            custom_dir: Optional custom directory to save to, overrides self.save_dir.

        Returns:
            The path to the saved checkpoint directory, or None on failure.
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

            # Save Daedalus agent if it has a dump method
            daedalus_path = None
            if hasattr(self.daedalus_agent, "dump"):
                daedalus_subdir = os.path.join(dpath, "daedalus")
                os.makedirs(daedalus_subdir, exist_ok=True)
                daedalus_path = self.daedalus_agent.dump(save_dir=daedalus_subdir)

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
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

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
            episode_iterator = range(num_episodes) if num_episodes is not None else count()
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
                                truncated
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
                        self.training_summary_data.append({
                            "episode": episode + 1,
                            "hero_reward": hero_reward,
                            "gun_reward": gun_reward,
                            "time_alive": time_alive,
                            "wave_clear": wave_clear,
                            "terminated": terminated,
                        })

                        # Calculate reward for Daedalus
                        daedalus_reward = self._calculate_daedalus_reward(episode_summary)

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
                        self.logger.info("Attempting to save final episode metrics CSV...")
                        last_save_dir = self._get_last_save_directory() if hasattr(self, "_get_last_save_directory") else None
                        if last_save_dir:
                            self._save_episode_metrics_csv(last_save_dir)
                            self.logger.info(f"Final metrics CSV saved in: {last_save_dir}")
                        else:
                            final_save_dir = os.path.join("model_saves", "final_run_metrics")
                            os.makedirs(final_save_dir, exist_ok=True)
                            self._save_episode_metrics_csv(final_save_dir)
                            self.logger.warning(f"No checkpoint directory found. Final metrics saved to: {final_save_dir}")
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
            logging.FileHandler("odyssey_training.log")
        ],
    )
    
    logging.getLogger("torch_geometric").setLevel(logging.WARNING)
    
    logger = logging.getLogger("odyssey_main")
    logger.info("[bold green]Starting Odyssey Training System[/]", extra={"markup": True})
    
    try:
        # Parse command-line arguments
        import argparse
        parser = argparse.ArgumentParser(description="Odyssey Training System")
        parser.add_argument("--agent", type=str, choices=["DQN", "PPO"], default="PPO", 
                            help="Type of Theseus agent to use (DQN or PPO)")
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
                hidden_channels=HIDDEN_CHANNELS, 
                out_channels=HERO_ACTION_SPACE_SIZE
            )
            hero_target = HeroGNN(
                hidden_channels=HIDDEN_CHANNELS, 
                out_channels=HERO_ACTION_SPACE_SIZE
            )
            gun_policy = GunGNN(
                hidden_channels=HIDDEN_CHANNELS, 
                out_channels=GUN_ACTION_SPACE_SIZE
            )
            gun_target = GunGNN(
                hidden_channels=HIDDEN_CHANNELS, 
                out_channels=GUN_ACTION_SPACE_SIZE
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
                hidden_channels=HIDDEN_CHANNELS, 
                out_channels=HERO_ACTION_SPACE_SIZE
            )
            hero_critic = HeroGNN(
                hidden_channels=HIDDEN_CHANNELS, 
                out_channels=1
            )
            gun_actor = GunGNN(
                hidden_channels=HIDDEN_CHANNELS, 
                out_channels=GUN_ACTION_SPACE_SIZE
            )
            gun_critic = GunGNN(
                hidden_channels=HIDDEN_CHANNELS, 
                out_channels=1
            )
            
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
        
        # Create Daedalus agent (dummy implementation for now)
        daedalus_agent = DummyDaedalusAgent()
        
        # Create Odyssey agent with the selected Theseus agent
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
        logger.info("[bold red]Odyssey training finished or interrupted[/]", extra={"markup": True})