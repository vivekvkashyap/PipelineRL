"""Main entry point for PipelineRL training.

Launches the Actor (vLLM inference) and Trainer (REINFORCE optimization)
as concurrent processes on separate GPUs. This is the core PipelineRL
architecture — unlike conventional RL which alternates between generation
and training, PipelineRL runs them simultaneously with in-flight weight
updates to keep data fresh and GPUs busy.

Architecture:
  - Actor runs as a separate OS subprocess with CUDA_VISIBLE_DEVICES=<actor_gpu>
  - Trainer runs in the main process with CUDA_VISIBLE_DEVICES=<trainer_gpu>
  - Communication via filesystem (pickle files for results, .pt files for weights)
  - This avoids multiprocessing Queue issues with vLLM's internal subprocess

Usage:
    uv run python scripts/run.py [--config configs/default.yaml]
"""

import argparse
import logging
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelinerl.config import PipelineRLConfig
from src.pipelinerl.dataset import load_gsm8k
from src.pipelinerl.utils import setup_logging
from src.pipelinerl.weight_sync import WeightSynchronizer

logger = logging.getLogger("pipelinerl")


def main():
    parser = argparse.ArgumentParser(description="PipelineRL Training")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--actor-only", action="store_true",
        help="Run only the Actor (used internally for subprocess launch)",
    )
    args = parser.parse_args()

    config = PipelineRLConfig.from_yaml(args.config)

    # -----------------------------------------------------------------
    # Actor subprocess mode: launched by the main process below
    # -----------------------------------------------------------------
    if args.actor_only:
        _run_actor_subprocess(config)
        return

    # -----------------------------------------------------------------
    # Main process: orchestrates Actor + Trainer
    # -----------------------------------------------------------------
    setup_logging(config.output_dir)

    logger.info("=" * 60)
    logger.info("PipelineRL: Faster On-policy RL for Long Sequence Generation")
    logger.info("=" * 60)
    logger.info(f"Model:          {config.model_name}")
    logger.info(f"Actor GPU:      {config.actor_gpu}")
    logger.info(f"Trainer GPU:    {config.trainer_gpu}")
    logger.info(f"Gen batch (H):  {config.generation_batch_size}")
    logger.info(f"Train batch (B):{config.train_batch_size}")
    logger.info(f"Optimizer steps: {config.total_optimizer_steps}")
    logger.info(f"IS clamp (c):   {config.importance_weight_clamp}")
    logger.info("=" * 60)

    # Prepare shared directories
    sync_dir = Path(config.weight_sync_dir)
    results_dir = sync_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    done_path = sync_dir / "done"
    done_path.unlink(missing_ok=True)  # clean up from previous runs

    # Initialize weight synchronizer and publish initial weights
    weight_sync = WeightSynchronizer(config.weight_sync_dir)

    # Save initial weights from trainer's model
    logger.info("Initializing Trainer model and publishing initial weights...")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.trainer_gpu)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    weight_sync.publish_weights(model, step=0)
    # Save tokenizer + config alongside weights so vLLM can reload them
    tokenizer.save_pretrained(weight_sync.hf_weight_dir)
    logger.info("Initial weights published")

    # -----------------------------------------------------------------
    # Launch Actor as a separate OS subprocess on the actor GPU.
    # This gives vLLM full control of its GPU without interference.
    # -----------------------------------------------------------------
    logger.info("Launching Actor subprocess...")
    actor_env = os.environ.copy()
    actor_env["CUDA_VISIBLE_DEVICES"] = str(config.actor_gpu)

    actor_proc = subprocess.Popen(
        [sys.executable, __file__, "--config", args.config, "--actor-only"],
        env=actor_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Stream Actor logs in a background thread
    def _stream_actor_logs():
        for line in actor_proc.stdout:
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                logger.info(f"[Actor] {text}")

    log_thread = Thread(target=_stream_actor_logs, daemon=True)
    log_thread.start()

    # -----------------------------------------------------------------
    # Run Trainer in the main process (Algorithm 2, Trainer Process)
    # -----------------------------------------------------------------
    start_time = time.time()

    _run_trainer(config, model, tokenizer, weight_sync, results_dir, done_path)

    elapsed = time.time() - start_time

    # Signal Actor to stop
    done_path.touch()
    actor_proc.wait(timeout=60)

    logger.info(f"Training complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info(f"Final model saved to: {config.output_dir}/final_model")


def _run_actor_subprocess(config: PipelineRLConfig) -> None:
    """Runs inside the Actor subprocess with its own CUDA_VISIBLE_DEVICES."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Actor] %(levelname)s: %(message)s",
    )

    from src.pipelinerl.dataset import load_gsm8k
    from src.pipelinerl.weight_sync import WeightSynchronizer

    train_data, _ = load_gsm8k(config)
    weight_sync = WeightSynchronizer(config.weight_sync_dir)

    # Import and run the actor main loop
    from src.pipelinerl.actor import actor_main
    actor_main(config, None, weight_sync, train_data, None)


def _run_trainer(
    config: PipelineRLConfig,
    model,
    tokenizer,
    weight_sync: WeightSynchronizer,
    results_dir: Path,
    done_path: Path,
) -> None:
    """Trainer loop — runs in the main process on the trainer GPU.

    Implements Algorithm 2 (Trainer Process):
      1. Wait for B completed sequences from Actor (via filesystem)
      2. Compute importance-weighted REINFORCE loss
      3. Optimizer step
      4. Publish new weights for in-flight update
    """
    import torch
    from src.pipelinerl.actor import SequenceResult
    from src.pipelinerl.trainer import compute_reinforce_loss
    from src.pipelinerl.utils import MetricsTracker

    device = torch.device("cuda:0")
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    metrics = MetricsTracker(output_dir=config.output_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_sequences = 0
    next_result_idx = 0  # track which result file to read next

    logger.info("Trainer starting training loop...")

    for step in range(1, config.total_optimizer_steps + 1):
        metrics.step_start()

        # -----------------------------------------------------------------
        # Collect B sequences from the results directory
        # Actor writes result_XXXXXXXX.pkl files; we read them in order.
        # -----------------------------------------------------------------
        batch: list[SequenceResult] = []
        wait_start = time.time()

        while len(batch) < config.train_batch_size:
            result_path = results_dir / f"result_{next_result_idx:08d}.pkl"
            if result_path.exists():
                try:
                    with open(result_path, "rb") as f:
                        result = pickle.load(f)
                    batch.append(result)
                    next_result_idx += 1
                    result_path.unlink()  # clean up after reading
                except Exception as e:
                    time.sleep(0.1)
                    continue
            else:
                # Wait for Actor to produce more results
                time.sleep(0.5)
                if time.time() - wait_start > 300:
                    logger.error("Timeout waiting for Actor results")
                    return

        total_sequences += len(batch)

        # -----------------------------------------------------------------
        # Compute importance-weighted REINFORCE loss (Eq. 5)
        # -----------------------------------------------------------------
        loss, batch_metrics = compute_reinforce_loss(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            device=device,
            importance_clamp=config.importance_weight_clamp,
            current_step=step,
        )

        # -----------------------------------------------------------------
        # Optimizer step (Algorithm 2, line 20)
        # -----------------------------------------------------------------
        optimizer.zero_grad()
        loss.backward()
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        # -----------------------------------------------------------------
        # Publish new weights for in-flight update (Algorithm 2, line 18)
        # The Actor will pick these up on its next check and load them
        # into vLLM — this is the in-flight weight update.
        # -----------------------------------------------------------------
        weight_sync.publish_weights(model, step=step)

        # -----------------------------------------------------------------
        # Log metrics
        # -----------------------------------------------------------------
        if step % config.log_interval == 0:
            metrics.log_step(
                step=step,
                reward=batch_metrics["mean_reward"],
                loss=loss.item(),
                ess=batch_metrics["ess"],
                max_token_lag=batch_metrics["max_token_lag"],
                min_token_lag=batch_metrics["min_token_lag"],
                avg_token_lag=batch_metrics["avg_token_lag"],
                mixed_policy_seqs=batch_metrics["mixed_policy_seqs"],
                num_sequences=total_sequences,
                batch_total_tokens=batch_metrics["batch_total_tokens"],
                avg_seq_length=batch_metrics["avg_seq_length"],
            )

        if step % config.save_interval == 0:
            ckpt_path = output_dir / f"checkpoint_step_{step}"
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    # Save final checkpoint
    final_path = output_dir / "final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    metrics.save()

    logger.info(
        f"Trainer finished | {config.total_optimizer_steps} steps | "
        f"{total_sequences} sequences"
    )


if __name__ == "__main__":
    main()
