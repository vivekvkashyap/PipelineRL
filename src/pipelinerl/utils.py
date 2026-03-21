"""Utility functions: ESS computation, token lag tracking, logging."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class MetricsTracker:
    """Track and log training metrics over time."""

    output_dir: str = "outputs"
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
            "mixed_policy_seqs": mixed_policy_seqs,
            "num_sequences": num_sequences,
            "batch_total_tokens": batch_total_tokens,
            "elapsed_seconds": elapsed,
        }
        self.history.append(entry)

        # Async level = number of weight versions spanned by sequences in batch
        # (how many in-flight updates happened during generation of this batch)
        async_level = max_token_lag - min_token_lag + 1

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

    def save(self) -> None:
        path = Path(self.output_dir) / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
