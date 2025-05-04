from dataclasses import dataclass
from torch import optim, cuda
from typing import Tuple
import logging

from daedalus.agentsv2.agent_PPO import PPOTrainer
from theseus.agent_ppo import AgentTheseusPPO

LOGGING_WINDOW: int = 50
THESEUS_SAVE_INTERVAL: int = 5
THESEUS_DEFAULT_HORIZON: int = 2048
THESEUS_DEFAULT_EPOCHS_PER_UPDATE: int = 10
THESEUS_DEFAULT_MINI_BATCH_SIZE_PPO: int = 64
THESEUS_DEFAULT_CLIP_EPSILON: float = 0.2
THESEUS_DEFAULT_GAE_LAMBDA: float = 0.95
THESEUS_DEFAULT_ENTROPY_COEFF: float = 0.01
THESEUS_DEFAULT_VF_COEFF: float = 0.5
THESEUS_DEFAULT_LEARNING_RATE_PPO: float = 3e-4
THESEUS_DEFAULT_DISCOUNT_FACTOR_PPO: float = 0.99
THESEUS_DEFAULT_HERO_HIDDEN_CHANNELS: int = 64
THESEUS_DEFAULT_GUN_HIDDEN_CHANNELS: int = 64
THESEUS_NUMERICAL_STABILITY_EPS: float = 1e-8
THESEUS_GRAD_CLIP_NORM: float = 0.5


@dataclass
class TheseusAgentConfig:
    optimizer_class: optim.Optimizer = (optim.AdamW,)
    learning_rate: float = (THESEUS_DEFAULT_LEARNING_RATE_PPO,)
    discount_factor: float = (THESEUS_DEFAULT_DISCOUNT_FACTOR_PPO,)
    horizon: int = (THESEUS_DEFAULT_HORIZON,)
    epochs_per_update: int = (THESEUS_DEFAULT_EPOCHS_PER_UPDATE,)
    mini_batch_size: int = (THESEUS_DEFAULT_MINI_BATCH_SIZE_PPO,)
    clip_epsilon: float = (THESEUS_DEFAULT_CLIP_EPSILON,)
    gae_lambda: float = (THESEUS_DEFAULT_GAE_LAMBDA,)
    entropy_coeff: float = (THESEUS_DEFAULT_ENTROPY_COEFF,)
    vf_coeff: float = (THESEUS_DEFAULT_VF_COEFF,)
    log_window_size: int = (LOGGING_WINDOW,)
    save_interval: int = (THESEUS_SAVE_INTERVAL,)
    hero_hidden_channels: int = (THESEUS_DEFAULT_HERO_HIDDEN_CHANNELS,)
    gun_hidden_channels: int = (THESEUS_DEFAULT_GUN_HIDDEN_CHANNELS,)


@dataclass
class DaedalusAgentConfig:
    env_mode: str = ("TURTLE",)
    batch_size: int = (8192 // 256,)
    learning_rate: int = (3e-4,)
    gamma: float = (0.99,)
    gae_lambda: float = (0.9,)
    clip_epsilon: float = (0.2,)
    critic_coef: float = (0.5,)
    entropy_coef: float = (0.01,)
    max_grad_norm: float = (0.5,)
    ppo_epochs: float = (10,)
    device: str = ("cuda" if cuda.is_available() else "cpu",)
    checkpoint_dir: str = ("ppo_daedalus_checkpoints",)
    log_dir: str = ("ppo_daedalus_logs",)
    critic_path: str = (
        "latest_checkpoint.pth",
    )  # "latest_checkpoint.pth" # Set path if needed
    map_size: Tuple[int] = ((12, 12),)
    steps_per_episode: int = (256,)  # Total steps collected across envs per episode
    update_interval: int = (128,)  # Perform PPO update every 128 steps
    num_episodes: int = (50000,)
    mini_batch_factor: int = (4,)
    log_window_size: int = (50,)


def construct_daedalus_from_config(config: DaedalusAgentConfig, checkpoint_path: str):
    model = PPOTrainer(
        env_mode=config.env_mode,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_epsilon=config.clip_epsilon,
        critic_coef=config.critic_coef,
        entropy_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm,
        ppo_epochs=config.ppo_epochs,
        device=config.device,
        checkpoint_dir=config.checkpoint_dir,
        log_dir=config.log_dir,
        critic_path=config.critic_path,  # "latest_checkpoint.pth" # Set path if needed
        map_size=config.map_size,
        steps_per_episode=config.steps_per_episode,  # Total steps collected across envs per episode
        update_interval=config.update_interval,  # Perform PPO update every 128 steps
        num_episodes=config.num_episodes,
        mini_batch_factor=config.mini_batch_factor,
        log_level=logging.INFO,  # Change to logging.DEBUG for more detail
        log_window_size=config.log_window_size,
    )

    model.load_checkpoint(checkpoint_path)
    

def construct_theseus_from_checkpoint(model_path: str):
    return AgentTheseusPPO.load(model_path)