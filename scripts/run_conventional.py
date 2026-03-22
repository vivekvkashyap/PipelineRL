"""Conventional (synchronous) RL baseline for comparison with PipelineRL.

Implements Algorithm 1 from the paper:
  1. Generate B*G*accum sequences (BLOCKING — trainer GPU idle)
  2. Train G optimizer steps on the generated data (actor GPU idle)
  3. Sync weights to the inference engine
  4. Repeat

This is the standard RL training loop used by most implementations.
The key difference from PipelineRL: generation and training are
strictly sequential — GPUs take turns being idle.

Usage:
    uv run python scripts/run_conventional.py --config configs/conventional.yaml
"""

import argparse
import logging
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelinerl.config import PipelineRLConfig
from src.pipelinerl.dataset import load_gsm8k
from src.pipelinerl.utils import MetricsTracker, finish_wandb, init_wandb, setup_logging

logger = logging.getLogger("pipelinerl")


def main():
    parser = argparse.ArgumentParser(description="Conventional RL Baseline")
    parser.add_argument("--config", type=str, default="configs/conventional.yaml")
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Internal: run as vLLM generation subprocess",
    )
    args = parser.parse_args()

    config = PipelineRLConfig.from_yaml(args.config)

    # -----------------------------------------------------------------
    # vLLM generation subprocess mode
    # -----------------------------------------------------------------
    if args.generate_only:
        _run_generator_subprocess(config)
        return

    # -----------------------------------------------------------------
    # Main process: orchestrates synchronous generate → train loop
    # -----------------------------------------------------------------
    setup_logging(config.output_dir)

    G = config.conventional_G
    B = config.train_batch_size
    accum = config.gradient_accumulation_steps
    seqs_per_round = B * accum * G

    logger.info("=" * 60)
    logger.info("Conventional RL Baseline (Algorithm 1)")
    logger.info("=" * 60)
    logger.info(f"Model:          {config.model_name}")
    logger.info(f"G (steps/round):{G}")
    logger.info(f"Micro-batch (B):{B}")
    logger.info(f"Accum steps:    {accum}")
    logger.info(f"Effective batch:{B * accum}")
    logger.info(f"Seqs/round:     {seqs_per_round}")
    logger.info(f"Total steps:    {config.total_optimizer_steps}")
    logger.info("=" * 60)

    wandb_active = init_wandb(config)

    # Clean up shared directories
    import shutil
    sync_dir = Path(config.weight_sync_dir)
    if sync_dir.exists():
        shutil.rmtree(sync_dir)
    sync_dir.mkdir(parents=True, exist_ok=True)
    results_dir = sync_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Initialize training model on Trainer GPU
    # -----------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.trainer_gpu)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.pipelinerl.actor import SequenceResult
    from src.pipelinerl.trainer import compute_reinforce_loss

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda:0")
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )

    # Save initial model for vLLM to load
    weight_dir = sync_dir / "hf_model"
    weight_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(weight_dir, safe_serialization=True)
    tokenizer.save_pretrained(weight_dir)

    # -----------------------------------------------------------------
    # Launch vLLM generator subprocess on Actor GPU
    # -----------------------------------------------------------------
    gen_env = os.environ.copy()
    gen_env["CUDA_VISIBLE_DEVICES"] = str(config.actor_gpu)

    gen_log_path = Path(config.output_dir) / "generator.log"
    gen_log_path.parent.mkdir(parents=True, exist_ok=True)
    gen_log_file = open(gen_log_path, "w")

    gen_proc = subprocess.Popen(
        [sys.executable, __file__, "--config", args.config, "--generate-only"],
        env=gen_env,
        stdout=gen_log_file,
        stderr=subprocess.STDOUT,
    )
    logger.info(f"Generator subprocess started (logs → {gen_log_path})")

    # Wait for generator to be ready
    ready_path = sync_dir / "generator_ready"
    while not ready_path.exists():
        time.sleep(1)
        if gen_proc.poll() is not None:
            logger.error("Generator subprocess died during startup")
            return
    logger.info("Generator ready")

    metrics = MetricsTracker(output_dir=config.output_dir, wandb_enabled=wandb_active)
    device = torch.device("cuda:0")
    current_step = 0
    total_sequences = 0
    num_rounds = config.total_optimizer_steps // G

    for round_idx in range(num_rounds):
        # =============================================================
        # GENERATION PHASE (blocking — trainer GPU idle)
        # =============================================================
        gen_start = time.time()

        # Write generation command: how many sequences to generate
        cmd_path = sync_dir / "gen_command.pkl"
        with open(cmd_path, "wb") as f:
            pickle.dump({
                "action": "generate",
                "num_sequences": seqs_per_round,
                "weight_version": current_step,
            }, f)

        # Signal generator
        (sync_dir / "gen_signal").touch()

        # Wait for results
        done_path = sync_dir / "gen_done"
        while not done_path.exists():
            time.sleep(0.5)
            if gen_proc.poll() is not None:
                logger.error("Generator subprocess died")
                return
        done_path.unlink()

        # Read results
        results_path = results_dir / "batch_results.pkl"
        with open(results_path, "rb") as f:
            all_sequences: list[SequenceResult] = pickle.load(f)

        gen_time = time.time() - gen_start
        logger.info(
            f"Round {round_idx + 1}/{num_rounds}  |  "
            f"Generated {len(all_sequences)} seqs in {gen_time:.1f}s  |  "
            f"Trainer was IDLE during generation"
        )

        # =============================================================
        # TRAINING PHASE (G optimizer steps — actor GPU idle)
        # =============================================================
        random.shuffle(all_sequences)

        for g in range(G):
            metrics.step_start()
            optimizer.zero_grad()
            current_step += 1

            # This G-step's sequences, split into accum micro-batches
            g_start = g * B * accum
            g_end = g_start + B * accum
            g_sequences = all_sequences[g_start:g_end]

            step_loss_accum = 0.0
            step_rewards = []
            step_total_tokens = 0
            step_total_seq_len = 0.0
            last_batch_metrics = None

            for a in range(accum):
                micro_batch = g_sequences[a * B : (a + 1) * B]
                if not micro_batch:
                    continue

                loss, batch_metrics = compute_reinforce_loss(
                    model=model,
                    tokenizer=tokenizer,
                    batch=micro_batch,
                    device=device,
                    importance_clamp=config.importance_weight_clamp,
                    current_step=current_step,
                )
                (loss / accum).backward()

                step_loss_accum += loss.item()
                step_rewards.extend([s.reward for s in micro_batch])
                step_total_tokens += batch_metrics["batch_total_tokens"]
                step_total_seq_len += batch_metrics["avg_seq_length"] * len(micro_batch)
                last_batch_metrics = batch_metrics

            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            total_sequences += len(g_sequences)

            # Log metrics
            if last_batch_metrics and current_step % config.log_interval == 0:
                metrics.log_step(
                    step=current_step,
                    reward=sum(step_rewards) / len(step_rewards) if step_rewards else 0,
                    loss=step_loss_accum / accum,
                    ess=last_batch_metrics["ess"],
                    max_token_lag=last_batch_metrics["max_token_lag"],
                    min_token_lag=last_batch_metrics["min_token_lag"],
                    avg_token_lag=last_batch_metrics["avg_token_lag"],
                    mixed_policy_seqs=0,  # always 0 for conventional RL
                    num_sequences=total_sequences,
                    batch_total_tokens=step_total_tokens,
                    avg_seq_length=step_total_seq_len / max(len(g_sequences), 1),
                )

        # =============================================================
        # WEIGHT SYNC (blocking — save model, tell generator to reload)
        # =============================================================
        model.save_pretrained(weight_dir, safe_serialization=True)

        # Signal generator to reload weights
        cmd_path = sync_dir / "gen_command.pkl"
        with open(cmd_path, "wb") as f:
            pickle.dump({"action": "reload"}, f)
        (sync_dir / "gen_signal").touch()

        # Wait for reload to complete
        reload_done = sync_dir / "reload_done"
        while not reload_done.exists():
            time.sleep(0.1)
        reload_done.unlink()

        # Save checkpoints
        if current_step % config.save_interval == 0:
            ckpt = Path(config.output_dir) / f"checkpoint_step_{current_step}"
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            logger.info(f"Saved checkpoint to {ckpt}")

    # Final save
    final_path = Path(config.output_dir) / "final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    metrics.save()

    # Shutdown generator
    cmd_path = sync_dir / "gen_command.pkl"
    with open(cmd_path, "wb") as f:
        pickle.dump({"action": "shutdown"}, f)
    (sync_dir / "gen_signal").touch()
    gen_proc.wait(timeout=30)
    gen_log_file.close()

    elapsed = metrics.history[-1]["elapsed_seconds"] if metrics.history else 0
    logger.info(f"Training complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info(f"Final model saved to: {final_path}")
    finish_wandb(wandb_active)


def _run_generator_subprocess(config: PipelineRLConfig) -> None:
    """vLLM generation subprocess. Waits for commands from the main process.

    Commands (via gen_command.pkl + gen_signal):
      - "generate": generate N sequences, write results, touch gen_done
      - "reload": reload weights from hf_model dir, touch reload_done
      - "shutdown": exit
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Generator] %(levelname)s: %(message)s",
    )

    from vllm import LLM, SamplingParams

    from src.pipelinerl.actor import SequenceResult
    from src.pipelinerl.dataset import load_gsm8k
    from src.pipelinerl.reward import compute_reward

    sync_dir = Path(config.weight_sync_dir)
    results_dir = sync_dir / "results"
    weight_dir = sync_dir / "hf_model"

    train_data, _ = load_gsm8k(config)

    # Initialize vLLM
    logger.info("Initializing vLLM engine...")
    llm = LLM(
        model=str(weight_dir),
        trust_remote_code=True,
        max_model_len=config.max_seq_len,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
        logprobs=1,
    )

    # Signal ready
    (sync_dir / "generator_ready").touch()
    logger.info("Generator ready, waiting for commands...")

    while True:
        # Wait for signal
        signal_path = sync_dir / "gen_signal"
        while not signal_path.exists():
            time.sleep(0.1)
        signal_path.unlink()

        # Read command
        cmd_path = sync_dir / "gen_command.pkl"
        with open(cmd_path, "rb") as f:
            cmd = pickle.load(f)

        action = cmd["action"]

        if action == "shutdown":
            logger.info("Shutting down")
            break

        elif action == "reload":
            logger.info("Reloading weights...")
            try:
                llm.llm_engine.engine_core.collective_rpc(
                    "reload_weights", args=(None, str(weight_dir), True)
                )
                logger.info("Weights reloaded")
            except Exception as e:
                logger.warning(f"Weight reload failed: {e}")
            (sync_dir / "reload_done").touch()

        elif action == "generate":
            num_sequences = cmd["num_sequences"]
            weight_version = cmd["weight_version"]
            logger.info(f"Generating {num_sequences} sequences...")

            # Sample prompts and generate
            sampled = random.choices(train_data, k=num_sequences)
            prompts = [item["prompt"] for item in sampled]

            gen_start = time.time()
            outputs = llm.generate(prompts, sampling_params)
            gen_time = time.time() - gen_start

            # Build SequenceResults
            results = []
            for item, output in zip(sampled, outputs):
                completion = output.outputs[0]

                token_log_probs = []
                if completion.logprobs:
                    for lp in completion.logprobs:
                        if lp:
                            values = list(lp.values())
                            token_log_probs.append(
                                values[0].logprob if values else 0.0
                            )

                num_tokens = len(completion.token_ids)
                reward = compute_reward(
                    response=completion.text,
                    ground_truth=item["answer"],
                    seq_len=num_tokens,
                    max_seq_len=config.max_seq_len,
                    length_penalty_start=config.length_penalty_start,
                    length_penalty_value=config.length_penalty_value,
                )

                results.append(SequenceResult(
                    prompt=item["prompt"],
                    question=item["question"],
                    response=completion.text,
                    ground_truth=item["answer"],
                    reward=reward,
                    log_probs=token_log_probs,
                    start_weight_version=weight_version,
                    end_weight_version=weight_version,  # no in-flight updates
                    num_tokens=num_tokens,
                ))

            # Write results
            results_path = results_dir / "batch_results.pkl"
            with open(results_path, "wb") as f:
                pickle.dump(results, f)

            logger.info(
                f"Generated {len(results)} seqs in {gen_time:.1f}s "
                f"({sum(r.num_tokens for r in results) / gen_time:.0f} tok/s)"
            )
            (sync_dir / "gen_done").touch()


if __name__ == "__main__":
    main()
