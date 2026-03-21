"""Configuration dataclass for PipelineRL."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PipelineRLConfig:
    """All hyperparameters for PipelineRL training.

    Loaded from a YAML file or constructed directly.
    """

    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_seq_len: int = 2048
    max_new_tokens: int = 1024

    # Actor (inference)
    actor_gpu: int = 0
    generation_batch_size: int = 32  # H in the paper
    temperature: float = 1.0
    top_p: float = 1.0

    # Trainer
    trainer_gpu: int = 1
    train_batch_size: int = 16  # B in the paper
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    total_optimizer_steps: int = 500
    gradient_checkpointing: bool = True

    # PipelineRL-specific
    importance_weight_clamp: float = 5.0  # c in Eq. 5
    length_penalty_start: float = 0.9
    length_penalty_value: float = -0.5
    ring_buffer_size: int = 256  # max sequences in the Actor→Trainer queue

    # Weight sync
    weight_sync_dir: str = "/tmp/pipelinerl/weights"

    # Dataset
    dataset_name: str = "openai/gsm8k"
    dataset_split: str = "main"

    # Logging
    log_interval: int = 1
    save_interval: int = 50
    output_dir: str = "outputs"

    # Weights & Biases
    wandb_project: str = "pipelinerl"
    wandb_run_name: str = ""  # auto-generated if empty
    wandb_enabled: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineRLConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
