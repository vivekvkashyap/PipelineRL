"""Evaluate a trained PipelineRL checkpoint on GSM8K test set.

Usage:
    uv run python scripts/evaluate.py --model outputs/final_model [--num_samples 200]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelinerl.config import PipelineRLConfig
from src.pipelinerl.dataset import load_gsm8k
from src.pipelinerl.reward import compute_reward

logger = logging.getLogger("pipelinerl")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PipelineRL model")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num_samples", type=int, default=0, help="0 = all")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = PipelineRLConfig.from_yaml(args.config)
    _, test_data = load_gsm8k(config)

    if args.num_samples > 0:
        test_data = test_data[: args.num_samples]

    logger.info(f"Evaluating {args.model} on {len(test_data)} test problems")

    # Use vLLM for fast batched inference
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=config.max_seq_len,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=config.max_new_tokens,
    )

    prompts = [item["prompt"] for item in test_data]
    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    total = len(test_data)
    for item, output in zip(test_data, outputs):
        response = output.outputs[0].text
        reward = compute_reward(
            response=response,
            ground_truth=item["answer"],
            seq_len=len(output.outputs[0].token_ids),
            max_seq_len=config.max_seq_len,
            length_penalty_start=1.0,  # no penalty during eval
            length_penalty_value=0.0,
        )
        if reward > 0:
            correct += 1

    accuracy = correct / total * 100
    logger.info(f"Results: {correct}/{total} correct ({accuracy:.1f}%)")
    print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
