"""Reward computation for math reasoning tasks.

Reward is 1.0 for a correct answer, 0.0 otherwise, with an optional
soft length penalty when the generated sequence approaches max_seq_len
(as described in the PipelineRL paper).
"""

import re


def extract_answer_from_response(response: str) -> str | None:
    """Extract the numerical answer from model output.

    Looks for patterns like:
      - '#### <number>'
      - '\\boxed{<number>}'
      - The last number in the response as a fallback
    """
    # Try #### format first (GSM8K style)
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", response)
    if match:
        return match.group(1).replace(",", "")

    # Try \\boxed{} format (MATH style)
    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        return match.group(1).strip().replace(",", "")

    # Fallback: last number in the response
    numbers = re.findall(r"-?[\d,]+\.?\d*", response)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def normalize_answer(answer: str) -> str:
    """Normalize a numerical answer for comparison."""
    answer = answer.strip().replace(",", "").replace(" ", "")
    # Remove trailing .0 for integer comparisons
    try:
        val = float(answer)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return answer


def compute_reward(
    response: str,
    ground_truth: str,
    seq_len: int,
    max_seq_len: int,
    length_penalty_start: float = 0.9,
    length_penalty_value: float = -0.5,
) -> float:
    """Compute reward for a generated response.

    Returns:
        1.0 if the extracted answer matches ground truth, 0.0 otherwise.
        Adds a soft penalty if the sequence length is close to max_seq_len
        (paper: "a soft penalty to the model when it gets close to the
        max sequence length").
    """
    predicted = extract_answer_from_response(response)
    if predicted is None:
        reward = 0.0
    else:
        reward = 1.0 if normalize_answer(predicted) == normalize_answer(ground_truth) else 0.0

    # Soft length penalty near the max sequence length
    threshold = int(max_seq_len * length_penalty_start)
    if seq_len >= threshold:
        reward += length_penalty_value

    return reward
