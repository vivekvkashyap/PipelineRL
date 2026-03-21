"""Utility functions: ESS computation, token lag tracking, logging, W&B integration."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("pipelinerl")


def setup_logging(output_dir: str) -> None:
    """Configure console + file logging."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(output_dir) / "train.log"),
        ],
    )


def compute_ess(importance_weights: torch.Tensor) -> float:
    """Compute the Effective Sample Size (ESS) from importance weights.

    ESS = (sum(w_i))^2 / (N * sum(w_i^2))    (Eq. 6 in the paper)

    Values close to 1.0 indicate on-policy data. Low ESS means high
    variance in importance weights, signaling stale off-policy data
    that could destabilize training.
    """
    w = importance_weights.float()
    n = w.numel()
    if n == 0:
        return 0.0
    sum_w = w.sum()
    sum_w2 = (w ** 2).sum()
    if sum_w2 == 0:
        return 0.0
    return (sum_w ** 2 / (n * sum_w2)).item()


def init_wandb(config) -> bool:
    """Initialize Weights & Biases run if enabled.

    Returns True if wandb was successfully initialized.
    """
    if not config.wandb_enabled:
        return False

    try:
        import wandb

        run_name = config.wandb_run_name or None
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
                # Model
                "model_name": config.model_name,
                "max_seq_len": config.max_seq_len,
                "max_new_tokens": config.max_new_tokens,
                # Actor
                "generation_batch_size_H": config.generation_batch_size,
                "temperature": config.temperature,
                # Trainer
                "train_batch_size_B": config.train_batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "max_grad_norm": config.max_grad_norm,
                "total_optimizer_steps": config.total_optimizer_steps,
                # PipelineRL
                "importance_weight_clamp_c": config.importance_weight_clamp,
                "ring_buffer_size": config.ring_buffer_size,
                "length_penalty_start": config.length_penalty_start,
                "length_penalty_value": config.length_penalty_value,
                # Dataset
                "dataset_name": config.dataset_name,
            },
        )
        logger.info(f"W&B initialized: project={config.wandb_project}, run={wandb.run.name}")
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}")
        return False


def finish_wandb(wandb_enabled: bool) -> None:
    """Finish the W&B run."""
    if not wandb_enabled:
        return
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


@dataclass
class MetricsTracker:
    """Track and log training metrics to console, JSON, and W&B.

    Logs are organized into structured groups for W&B:
      - train/reward, train/loss           → training progress
      - policyness/ess, policyness/max_token_lag, ...  → on-policyness
      - throughput/tokens_per_s, throughput/seq_length  → speed
      - pipeline/async_level, pipeline/mixed_policy_seqs → pipeline behavior
    """

    output_dir: str = "outputs"
    wandb_enabled: bool = False
    history: list[dict] = field(default_factory=list)
    _start_time: float = field(default_factory=time.time)
    _step_start_time: float = field(default_factory=time.time)
    _total_tokens: int = 0

    def step_start(self) -> None:
        """Call at the beginning of each training step to track step time."""
        self._step_start_time = time.time()

    def log_step(
        self,
        step: int,
        reward: float,
        loss: float,
        ess: float,
        max_token_lag: int,
        min_token_lag: int,
        avg_token_lag: float,
        mixed_policy_seqs: int,
        num_sequences: int,
        batch_total_tokens: int,
        avg_seq_length: float,
    ) -> None:
        elapsed = time.time() - self._start_time
        step_time = time.time() - self._step_start_time
        self._total_tokens += batch_total_tokens
        throughput = self._total_tokens / elapsed if elapsed > 0 else 0
        async_level = max_token_lag - min_token_lag + 1

        entry = {
            "step": step,
            "time": step_time,
            "reward": reward,
            "loss": loss,
            "ess": ess,
            "throughput_tokens_per_s": throughput,
            "avg_seq_length": avg_seq_length,
            "max_token_lag": max_token_lag,
            "min_token_lag": min_token_lag,
            "avg_token_lag": avg_token_lag,
            "async_level": async_level,
            "mixed_policy_seqs": mixed_policy_seqs,
            "num_sequences": num_sequences,
            "batch_total_tokens": batch_total_tokens,
            "elapsed_seconds": elapsed,
        }
        self.history.append(entry)

        # Console logging
        logger.info(
            f"Step {step:4d} | Time: {step_time:.2f}s | "
            f"Reward: {reward:.4f} | "
            f"Throughput: {throughput:.1f} tokens/s | "
            f"Seq. Length: {avg_seq_length:.1f} tokens/sample | "
            f"Async Level: {async_level} | "
            f"Max. Off-Policy Level: {max_token_lag}"
        )
        logger.info(
            f"         ESS: {ess:.4f} | "
            f"Token Lag: [{min_token_lag}..{max_token_lag}] avg={avg_token_lag:.1f} | "
            f"Mixed-Policy Seqs: {mixed_policy_seqs} | "
            f"Loss: {loss:.6f}"
        )

        # W&B logging — structured into groups
        if self.wandb_enabled:
            try:
                import wandb
                wandb.log(
                    {
                        # Training progress
                        "train/reward": reward,
                        "train/loss": loss,
                        "train/step_time_s": step_time,
                        "train/total_sequences": num_sequences,
                        # On-policyness metrics (Figure 6a, 6b)
                        "policyness/ess": ess,
                        "policyness/max_token_lag": max_token_lag,
                        "policyness/min_token_lag": min_token_lag,
                        "policyness/avg_token_lag": avg_token_lag,
                        # Throughput metrics (Figure 5c)
                        "throughput/tokens_per_s": throughput,
                        "throughput/avg_seq_length": avg_seq_length,
                        "throughput/batch_total_tokens": batch_total_tokens,
                        # Pipeline behavior (unique to PipelineRL)
                        "pipeline/async_level": async_level,
                        "pipeline/mixed_policy_seqs": mixed_policy_seqs,
                        # Time
                        "time/elapsed_s": elapsed,
                        "time/step_time_s": step_time,
                    },
                    step=step,
                )
            except Exception as e:
                logger.warning(f"W&B log failed: {e}")

    def save(self) -> None:
        """Save metrics history to JSON file."""
        path = Path(self.output_dir) / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
