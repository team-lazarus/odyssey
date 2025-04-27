from theseus.utils.network.environment import Environment
from theseus.utils import State
from typing import Any, Dict, List, Tuple, Optional
import json
import logging

logger = logging.getLogger("odyssey_environment")

class OdysseyEnvironment(Environment):    
    def __init__(self):
        # Initialize the base Theseus environment
        super().__init__()
        
        # Initialize clients for both Theseus and Daedalus
        self.theseus_client = None
        self.daedalus_client = None
        
        # Track additional state for Odyssey
        self.current_game_matrix = None
        self.wave_clear = False
        self.terminated = False
        self.episode_rewards = []
        self.logger = logging.getLogger("odyssey-environment")
    
    def initialise_environment(self, map_data: List[List]) -> State:
        # Initialize Theseus environment using parent class method
        if not self.tcp_client.connected:
            self.tcp_client.connect()
        
        # Set current game matrix
        self.current_game_matrix = map_data
        
        # Reset the wave_clear flag
        self.wave_clear = False
        self.episode_rewards = []
        
        # Send the game matrix to Labyrinth
        self.send_game_matrix_to_labyrinth(self.current_game_matrix)
        
        # Get initial state from Labyrinth
        game_response = self.tcp_client.read()
        state, _, _, _ = self.parse_game_response(game_response)
        
        return state
    
    def send_game_matrix_to_labyrinth(self, game_matrix: Any):
        serialized_matrix = self.serialize_game_matrix(game_matrix)
        
        # Send the initial command to set up a new game
        try:
            self.tcp_client.write(f"NEW_GAME:{serialized_matrix}")
            
            # Wait for acknowledgment from Labyrinth
            response = self.tcp_client.read()
            
            # Check if the response indicates successful game initialization
            if response is None or (isinstance(response, dict) and "game_initialized" not in response):
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
    
    def step(self, action: int | str | List[int]) -> Tuple[State, Tuple[float, float], bool]:
        # Use the parent class step method to execute the action
        next_state, rewards, terminated = super().step(action)
        
        # Add rewards to episode tracking
        self.episode_rewards.append(rewards)
        
        # Return the results
        # If the wave is cleared OR the hero died (terminated), signal episode end
        return next_state, rewards, terminated or self.wave_clear
    
    def parse_game_response(self, game_response: Dict[str, Any]) -> Tuple[State, bool, bool, Tuple[float, float]]:
        # Use the parent class method for parsing
        state, terminated, wave_clear, rewards = super().parse_game_response(game_response)
        
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
                "terminated": False  # Just report the termination status
            }
        
        total_hero_reward = sum(reward[0] for reward in self.episode_rewards)
        total_gun_reward = sum(reward[1] for reward in self.episode_rewards)
        episode_length = len(self.episode_rewards)
        
        return {
            "total_hero_reward": total_hero_reward,
            "total_gun_reward": total_gun_reward,
            "episode_length": episode_length,
            "wave_clear": self.wave_clear,
            "terminated": self.terminated
        }
    
    def update_daedalus(self, episode_summary: Dict[str, Any]):
        try:
            #figure out, talk to others abt it
            self.logger.info(f"Updating Daedalus with episode summary: {episode_summary}")
            pass
        except Exception as e:
            self.logger.warning(f"Warning: Failed to update Daedalus model: {e}")
            print(f"Warning: Failed to update Daedalus model: {e}")
    
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