"""Importance-weighted REINFORCE loss computation for PipelineRL.

Implements the loss function from Eq. 5 of the PipelineRL paper:

    ∇̂J_IS(π) = (1/m) Σ_j Σ_t min(c, π(y_t|x_j)/μ(y_t|x_j))
                 * (R(x_j,y_j) - baseline) * ∇ log π(y_t|x_j, y_{<t})

For verifiable tasks like GSM8K with binary rewards (1 if correct, 0
otherwise), the baseline is the mean reward in the batch (GRPO-style
group relative baseline). No learned value function is needed.

Uses **batched forward pass** (sequence packing) for efficiency:
instead of B separate forward passes (one per sequence), all sequences
are padded and processed in a single batched forward pass. This is
significantly faster, especially on GPU where batch parallelism is key.
"""

import logging

import torch
import torch.nn.functional as F

from .actor import SequenceResult
from .utils import compute_ess

logger = logging.getLogger("pipelinerl")


def compute_reinforce_loss(
    model: torch.nn.Module,
    tokenizer,
    batch: list[SequenceResult],
    device: torch.device,
    importance_clamp: float,
    current_step: int,
) -> tuple[torch.Tensor, dict]:
    """Compute importance-weighted REINFORCE loss with batched forward pass.

    Instead of processing each sequence individually, this function:
      1. Tokenizes all sequences and pads them into a single batch tensor
      2. Runs ONE forward pass through the model for the entire batch
      3. Extracts per-sequence, per-token log probs from the batched output
      4. Computes importance-weighted REINFORCE loss (Eq. 5)

    This is the "sequence packing" optimization from the paper (Section 4):
    "Key optimizations include online sequence packing for fast training."

    Args:
        model: Current policy model on the trainer GPU.
        tokenizer: Tokenizer for encoding prompts + responses.
        batch: List of SequenceResult from the Actor.
        device: Training device.
        importance_clamp: Maximum importance weight (c in the paper).
        current_step: Current optimizer step for lag computation.

    Returns:
        (loss, metrics_dict) with reward, ESS, token lag, and throughput stats.
    """
    rewards = torch.tensor([s.reward for s in batch], dtype=torch.float32)
    # GRPO-style baseline: mean reward in the batch.
    # No learned value function needed for verifiable rewards.
    baseline = rewards.mean()
    advantages = rewards - baseline

    # Token lag tracking (Figure 3a, Figure 6a)
    max_token_lags = []
    min_token_lags = []
    for s in batch:
        max_token_lags.append(current_step - s.start_weight_version)
        min_token_lags.append(current_step - s.end_weight_version)

    # -------------------------------------------------------------------------
    # Step 1: Tokenize all sequences and identify valid ones
    # -------------------------------------------------------------------------
    full_texts = []
    prompt_lengths = []
    seq_indices = []  # which batch items are valid

    for i, seq in enumerate(batch):
        if not seq.response.strip() or seq.num_tokens == 0:
            continue
        full_texts.append(seq.prompt + seq.response)

        # Compute prompt length in tokens (needed to split prompt/response)
        prompt_enc = tokenizer(
            seq.prompt, add_special_tokens=False,
        )
        prompt_lengths.append(len(prompt_enc.input_ids))
        seq_indices.append(i)

    if not full_texts:
        zero_loss = torch.zeros(1, device=device, requires_grad=True)
        return zero_loss, _empty_metrics(batch, max_token_lags, min_token_lags)

    # -------------------------------------------------------------------------
    # Step 2: Pad and batch all sequences for a SINGLE forward pass.
    #
    # We use RIGHT padding so that prompt tokens are aligned at the left
    # and the causal attention mask works correctly. Each sequence in the
    # batch is independent (no cross-sequence attention between rows).
    # -------------------------------------------------------------------------
    tokenizer.padding_side = "right"
    encoded = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model.config.max_position_embeddings,
    ).to(device)

    input_ids = encoded.input_ids        # (num_valid, max_len)
    attention_mask = encoded.attention_mask  # (num_valid, max_len)

    # -------------------------------------------------------------------------
    # Step 3: Single batched forward pass — this is the key speedup.
    # Instead of B separate forward passes, ONE pass processes everything.
    # -------------------------------------------------------------------------
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    all_logits = outputs.logits  # (num_valid, max_len, vocab_size)

    # -------------------------------------------------------------------------
    # Step 4: Extract per-sequence loss from the batched output
    # -------------------------------------------------------------------------
    total_loss = torch.zeros(1, device=device)
    all_importance_weights = []

    for batch_row, orig_idx in enumerate(seq_indices):
        seq = batch[orig_idx]
        prompt_len = prompt_lengths[batch_row]
        seq_len = attention_mask[batch_row].sum().item()
        response_len = int(seq_len - prompt_len)

        if response_len <= 0:
            continue

        # Extract logits for the response tokens only.
        # logits[t] predicts token[t+1], so response logits start at prompt_len-1
        logits = all_logits[batch_row, prompt_len - 1 : seq_len - 1, :]  # (response_len, vocab)
        response_ids = input_ids[batch_row, prompt_len : seq_len]  # (response_len,)

        current_log_probs = F.log_softmax(logits.float(), dim=-1)
        current_token_log_probs = current_log_probs.gather(
            1, response_ids.unsqueeze(1)
        ).squeeze(1)  # (response_len,)

        # -----------------------------------------------------------------
        # Importance weights: w_t = π_θ(y_t|x) / μ(y_t|x)
        # Clamped to c to reduce variance (paper: c = 5)
        # -----------------------------------------------------------------
        if seq.log_probs and len(seq.log_probs) > 0:
            behavior_log_probs = torch.tensor(
                seq.log_probs[:response_len], device=device, dtype=torch.float32
            )
            if behavior_log_probs.shape[0] < response_len:
                behavior_log_probs = F.pad(
                    behavior_log_probs, (0, response_len - behavior_log_probs.shape[0])
                )
            elif behavior_log_probs.shape[0] > response_len:
                behavior_log_probs = behavior_log_probs[:response_len]

            log_ratios = current_token_log_probs.detach() - behavior_log_probs
            importance_weights = torch.exp(log_ratios).clamp(max=importance_clamp)
        else:
            importance_weights = torch.ones(response_len, device=device)

        all_importance_weights.append(importance_weights)

        # -----------------------------------------------------------------
        # REINFORCE loss for this sequence (Eq. 5):
        #   loss_j = -(1/T_j) Σ_t w_t * (R - baseline) * log π_θ(y_t)
        #
        # We normalize by response_len (T_j) per sequence to keep the
        # loss magnitude stable regardless of sequence length. This is
        # standard practice in REINFORCE for LLMs (GRPO, DeepSeek-R1).
        # Without this, the loss scales linearly with sequence length
        # and explodes for long sequences.
        # -----------------------------------------------------------------
        advantage = advantages[orig_idx].to(device)
        seq_loss = -(importance_weights * advantage * current_token_log_probs).sum()
        total_loss = total_loss + seq_loss / response_len

    num_valid = len(seq_indices)
    if num_valid > 0:
        total_loss = total_loss / num_valid

    # Effective Sample Size (Eq. 6)
    if all_importance_weights:
        ess = compute_ess(torch.cat(all_importance_weights))
    else:
        ess = 1.0

    # Throughput stats
    token_counts = [s.num_tokens for s in batch]
    batch_total_tokens = sum(token_counts)
    avg_seq_length = batch_total_tokens / len(batch) if batch else 0.0

    batch_metrics = {
        "mean_reward": rewards.mean().item(),
        "ess": ess,
        "max_token_lag": max(max_token_lags) if max_token_lags else 0,
        "min_token_lag": min(min_token_lags) if min_token_lags else 0,
        "avg_token_lag": sum(max_token_lags) / len(max_token_lags) if max_token_lags else 0.0,
        "mixed_policy_seqs": sum(
            1 for s in batch if s.start_weight_version != s.end_weight_version
        ),
        "batch_total_tokens": batch_total_tokens,
        "avg_seq_length": avg_seq_length,
    }

    return total_loss, batch_metrics


def _empty_metrics(
    batch: list[SequenceResult],
    max_token_lags: list[int],
    min_token_lags: list[int],
) -> dict:
    """Return zero-valued metrics when no valid sequences in batch."""
    return {
        "mean_reward": 0.0,
        "ess": 1.0,
        "max_token_lag": max(max_token_lags) if max_token_lags else 0,
        "min_token_lag": min(min_token_lags) if min_token_lags else 0,
        "avg_token_lag": 0.0,
        "mixed_policy_seqs": 0,
        "batch_total_tokens": 0,
        "avg_seq_length": 0.0,
    }
