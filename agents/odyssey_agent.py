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
from tqdm import tqdm

from odyssey.utils import (
    OdysseyEnvironment, 
    TheseusAgentConfig, 
    DaedalusAgentConfig,
    construct_daedalus_from_config,
    construct_theseus_from_checkpoint,
)
from daedalus.agentsv2.agent_ppo import PPOTrainer


class OdysseyAgent:
    def __init__(
        self,
        theseus_checkpoint: str,
        daedalus_agent_config: DaedalusAgentConfig,
        env: Optional[OdysseyEnvironment] = None,
        *,
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
        self.training_summary_data = []

        self.daedalus_steps = []

        # Initialize agents
        self.daedalus_agent_config = daedalus_agent_config 
        self.initialise_agents()
    
    def initialise_agents(self):
        self.theseus = construct_theseus_from_checkpoint(self.theseus_checkpoint)
        self.horizon = self.theseus.horizon
        self.daedalus = construct_daedalus_from_config(self.daedalus_agent_config)
    
    def train(self, num_episodes:int):
        self.num_episodes = num_episodes
        for ep in range(num_episodes):
            self.train_episode(ep)
    
    def train_one_episode(self, ep: int):
        progress_columns = [
            TextColumn("[progress.description]{task.description}"), BarColumn(),
            MofNCompleteColumn(), TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
        ]

        self.progress = Progress(*progress_columns, transient=False)
        self.episode_task = self.progress.add_task(
                "[cyan]Training Episodes...", total=self.num_episodes
            )

        self.generate_map()
        self.env.prepare_for_theseus()
        self.train_theseus_for_one_episode(ep=ep)
    
    def train_theseus_for_one_episode(self, ep: int):
        try:
            ep_reward_hero, ep_reward_gun, time_alive, terminated, truncated = self.theseus.run_trajectory_collection(
                ep, self.progress, self.episode_task
            )
            if terminated or truncated:
                self._update_metrics(ep_reward_hero, ep_reward_gun, time_alive)
                self._log_episode_metrics(ep, time_alive)

                if time_alive > 0:
                    self.training_summary_data.append({
                        "Episode": ep + 1,
                        "Time_Alive": time_alive,
                        "Reward_Hero": ep_reward_hero,
                        "Reward_Gun": ep_reward_gun,
                    })
                else:
                    self.logger.warning(f"Episode {ep+1} ended with 0 time alive. Not adding to summary data.")

                self._save_checkpoint_if_needed(ep + 1)
                self.progress.update(ep, advance=1)

            if self.total_steps_collected >= self.horizon:
                self.logger.info(f"Horizon {self.horizon} reached. Starting PPO update.")
                self.progress.update(ep, description=f"[cyan]Ep. {ep+1} (Updating Policy...)")
                self._update_policy()
                self.progress.update(self.episode_task, description=f"[cyan]Ep. {ep+1} (Collecting Data...)")
        except RuntimeError as e:
            self.logger.critical(f"Stopping training due to runtime error in episode {ep+1}: {e}", exc_info=True)
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user.")
        except Exception as e:
            self.logger.critical(f"Unexpected error during episode {ep+1}: {e}", exc_info=True)
        finally:
            self.progress.stop()
            if self.num_episodes is not None:
                final_desc = "[green]Training Finished" 
                self.progress.update(self.episode_task, description=final_desc)
            else:
                final_desc = "[yellow]Training Stopped (Infinite Mode)"
                self.progress.update(self.episode_task, description=final_desc)
        
        ep_reward_hero, ep_reward_gun, time_alive, terminated, _ = self.theseus.run_trajectory_collection() 
    
    def generate_map(self, *, n_steps=256):
        state = self.env.reset()
        previous_rewards = self.env.get_initial_daedalus_rewards()
        
        for _ in tqdm(range(n_steps)):
            actions, log_probs, values = self.daedalus.select_action(
                state
            )

            next_state, reward, done, truncated, info = (
                self.env.step_daedalus(actions.item())
            )

            rewards, previous_rewards = rewards - previous_rewards, rewards
            state = next_state