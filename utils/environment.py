import json
import logging
import torch
import random
import numpy as np

from theseus.utils.network.environment import Environment
from theseus.utils import State
from typing import Any, Dict, List, Tuple, Optional

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskID,
)

logger = logging.getLogger("odyssey_environment")

TILE_WALL = 0
TILE_EMPTY = 1
ENEMY_TILE = 2

console = Console()

def clamp(n: float):
    return int(max(64, min(n, 128)))

def visualize_maps(map, x=0, y=0, title: str = "- Map 0 -") -> None:
    """Visualize maps using rich."""

    color_map = {
        0: "[grey27]0[/grey27]",  # Empty/Wall
        1: "[white]1[/white]",  # Path
        2: "[red]2[/red]",  # Enemy
        3: "[red]3[/red]",  # Damaged Enemy?
        4: "[red]4[/red]",  # Dead Enemy?
        5: "[red]5[/red]",  # Player (if present)
        6: "[green]6[/green]",  # Door
    }

    console.print(f"\n{title}")

    table = Table(
        title=title,
        show_header=False,
        show_lines=False,
        box=None,
        padding=0,
    )

    map_height, map_width = map.shape[0], map.shape[1]
    for _ in range(map_width):
        table.add_column()  # No header text needed

    for j in range(map_height):
        row = []
        for k in range(map_width):
            cell_value = int(map[j, k].item())
            if j == x and k == y:
                row.append(f"[yellow]{cell_value}[/yellow]")
            row.append(color_map.get(cell_value, f"[cyan]{cell_value}[/cyan]"))
        table.add_row(*row)

    console.print(table)
    rprint("")  # Use rich print for spacing



class OdysseyEnvironment(Environment):
    def __init__(self, *, seed: int=42, map_size: Tuple[int]=(12,12)):
        # Initialize the base Theseus environment
        super().__init__()

        self.seed = seed
        self.map_size = map_size
        self.current_map = None
        self.wave_clear = False
        self.terminated = False
        self.episode_rewards = []
        self.logger = logging.getLogger("odyssey-environment")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.initialise_environment()
    
    def reset(self):
        """Apply random walk algorithm to generate a map for a specific batch index."""
        # Clear the existing map for this index first
        pcgrl_map = torch.zeros(
            (self.map_size[0], self.map_size[1]),
            dtype=torch.int64,
            device=self.device,
        )
        
        steps = clamp(random.normalvariate(88, 16))
        row, col = random.randint(0, self.map_size[0] - 1), random.randint(
            1, self.map_size[1] - 2
        )

        for _ in range(steps):
            # Place a tile
            tile_choice = random.random()
            if tile_choice < 0.90:  # 80% empty # 90% removing doors
                tile_type = TILE_EMPTY
            # elif tile_choice < 0.84:  # 4% door
            #     tile_type = c.TILE_DOOR
            else:  # 16% enemy # 10% chance
                tile_type = ENEMY_TILE

            pcgrl_map[row, col] = tile_type

            # Move randomly
            direction = random.randint(0, 3)  # 0: up, 1: down, 2: left, 3: right
            if direction == 0 and row > 1:
                row -= 1
            elif direction == 1 and row < self.map_size[0] - 2:
                row += 1
            elif direction == 2 and col > 1:
                col -= 1
            elif direction == 3 and col < self.map_size[1] - 2:
                col += 1

        self.current_map = pcgrl_map

    def initialise_environment(self, map_data: List[List]) -> State:
        # Initialize Theseus environment using parent class method
        if not self.tcp_client.connected:
            self.tcp_client.connect()

        self.reset()

        self.wave_clear = False
        self.terminated = False
        self.episode_rewards = []

        """
        self.send_game_matrix_to_labyrinth(self.current_game_matrix)

        # Get initial state from Labyrinth
        game_response = self.tcp_client.read()
        state, _, _, _ = self.parse_game_response(game_response)

        return state
        """
    
    def step_daedalus(self, action):
        pass

    def send_game_matrix_to_labyrinth(self, game_matrix: Any):
        serialized_matrix = self.serialize_game_matrix(game_matrix)

        # Send the initial command to set up a new game
        try:
            self.tcp_client.write(f"NEW_GAME:{serialized_matrix}")

            # Wait for acknowledgment from Labyrinth
            response = self.tcp_client.read()

            # Check if the response indicates successful game initialization
            if response is None or (
                isinstance(response, dict) and "game_initialized" not in response
            ):
                self.logger.error(f"Failed to initialize game in Labyrinth: {response}")
                raise Exception(f"Failed to initialize game in Labyrinth: {response}")

            self.logger.info("Game matrix successfully sent to Labyrinth")
        except Exception as e:
            self.logger.error(f"Error sending game matrix to Labyrinth: {e}")
            raise Exception(f"Error sending game matrix to Labyrinth: {e}")

    def serialize_game_matrix(self, game_matrix: Any) -> str:
        # For consistency, converted to JSON
        try:
            return json.dumps(game_matrix)
        except Exception as e:
            self.logger.error(f"Error serializing game matrix: {e}")
            raise Exception(f"Error serializing game matrix: {e}")

    def step(
        self, action: int | str | List[int]
    ) -> Tuple[State, Tuple[float, float], bool]:
        # Use the parent class step method to execute the action
        next_state, rewards, terminated = super().step(action)

        # Add rewards to episode tracking
        self.episode_rewards.append(rewards)

        # Return the results
        # If the wave is cleared OR the hero died (terminated), signal episode end
        return next_state, rewards, terminated or self.wave_clear

    def parse_game_response(
        self, game_response: Dict[str, Any]
    ) -> Tuple[State, bool, bool, Tuple[float, float]]:
        # Use the parent class method for parsing
        state, terminated, wave_clear, rewards = super().parse_game_response(
            game_response
        )

        # Update our wave_clear flag from the parent method's return
        self.wave_clear = wave_clear
        self.terminated = terminated

        # Return all values as is
        return state, terminated, wave_clear, rewards

    def get_episode_summary(self) -> Dict[str, Any]:
        if not self.episode_rewards:
            return {
                "total_hero_reward": 0,
                "total_gun_reward": 0,
                "episode_length": 0,
                "wave_clear": self.wave_clear,
                "terminated": False,  # Just report the termination status
            }

        total_hero_reward = sum(reward[0] for reward in self.episode_rewards)
        total_gun_reward = sum(reward[1] for reward in self.episode_rewards)
        episode_length = len(self.episode_rewards)

        return {
            "total_hero_reward": total_hero_reward,
            "total_gun_reward": total_gun_reward,
            "episode_length": episode_length,
            "wave_clear": self.wave_clear,
            "terminated": self.terminated,
        }

    def update_daedalus(self, episode_summary: Dict[str, Any]):
        """
        Passes the episode summary to the Daedalus agent.
        This enables Daedalus to learn from the outcome of the episode.
        """
        try:
            # If the environment has a reference to the Daedalus agent, update it
            if hasattr(self, "daedalus_agent") and hasattr(
                self.daedalus_agent, "update_episode"
            ):
                self.daedalus_agent.update_episode(episode_summary)
                self.logger.info(
                    "Called daedalus_agent.update_episode with episode summary."
                )
            else:
                # Otherwise, just log the summary (as before)
                self.logger.info(f"Episode summary for Daedalus: {episode_summary}")
        except Exception as e:
            self.logger.warning(f"Warning: Failed to update Daedalus agent: {e}")
            print(f"Warning: Failed to update Daedalus agent: {e}")

    def reset_for_next_wave(self, map_data: List[List]) -> State:
        # Reset wave_clear flag
        self.wave_clear = False

        # Set current game matrix
        self.current_game_matrix = map_data

        # Send the new game matrix to Labyrinth
        self.send_game_matrix_to_labyrinth(self.current_game_matrix)

        # Get initial state for the new wave
        game_response = self.tcp_client.read()
        state, _, _, _ = self.parse_game_response(game_response)

        return state
