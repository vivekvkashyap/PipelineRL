"""GSM8K dataset loading and prompt formatting for math reasoning.

Uses the model's built-in HuggingFace chat_template for proper prompt
formatting rather than hardcoding special tokens.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

from .config import PipelineRLConfig

# System prompt encouraging step-by-step reasoning with a clear final answer.
SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step. "
    "At the very end, write your final numerical answer after '#### '."
)


def format_prompt(question: str, tokenizer) -> str:
    """Format a math question using the model's chat_template.

    Uses tokenizer.apply_chat_template() which handles model-specific
    special tokens (e.g., <|im_start|> for Qwen, <|begin_of_text|> for
    Llama, etc.) automatically.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    # tokenize=False returns the formatted string, add_generation_prompt=True
    # appends the assistant turn prefix so the model starts generating
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_ground_truth(answer_text: str) -> str:
    """Extract the final numerical answer from GSM8K's '#### <number>' format."""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    return answer_text.strip()


def load_gsm8k(config: PipelineRLConfig) -> tuple[list[dict], list[dict]]:
    """Load GSM8K and return (train_data, test_data).

    Each item is {"prompt": str, "answer": str, "question": str} where
    prompt is formatted using the model's chat_template and answer is
    the ground-truth numerical value.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )

    dataset = load_dataset(config.dataset_name, config.dataset_split)

    def _convert(split):
        items = []
        for row in split:
            items.append({
                "prompt": format_prompt(row["question"], tokenizer),
                "answer": extract_ground_truth(row["answer"]),
                "question": row["question"],
            })
        return items

    train_data = _convert(dataset["train"])
    test_data = _convert(dataset["test"])
    return train_data, test_data
