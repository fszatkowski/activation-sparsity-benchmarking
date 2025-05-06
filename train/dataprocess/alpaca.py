import random
from typing import Callable

from transformers import AutoTokenizer

from train.args import FinetuningArguments

QA_PREFIX_PAIRS = [
    ("Instruction: ", "Response: "),
    ("Q: ", "A: "),
    ("Question: ", "Answer: "),
    ("Question:\n", "Answer:\n"),
    ("Q:\n", "A:\n"),
    ("Instruction:\n", "Response:\n"),
]
INPUT_PREFIXES = [""]
SEPARATORS = ["\n", "\n\n", " "]


def get_alpaca_preprocess_fn(
    training_args: FinetuningArguments, tokenizer: AutoTokenizer
) -> Callable:
    mask_prompt_labels = training_args.mask_prompt
    max_seq_length = training_args.max_seq_length

    def preprocess_fn(row):
        instruction = row["instruction"].strip()
        input_text = row["input"].strip()
        output = row["output"].strip()

        if not instruction:
            raise ValueError("Missing instruction")
        if not output:
            raise ValueError("Output text is empty.")

        q_prefix, a_prefix = random.choice(QA_PREFIX_PAIRS)
        input_prefix = random.choice(INPUT_PREFIXES) if input_text else ""
        sep = random.choice(SEPARATORS)

        if input_text:
            prompt = (
                f"{q_prefix}{instruction}{sep}{input_prefix}{input_text}{sep}{a_prefix}"
            )
        else:
            prompt = f"{q_prefix}{instruction}{sep}{a_prefix}"

        full_text = prompt + output

        # Tokenize the full prompt + output
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        labels = input_ids.clone()

        # If the flag is set ignore prompt part in loss
        if mask_prompt_labels:
            prompt_tokens = tokenizer(
                prompt,
                return_tensors="pt",
            )[
                "input_ids"
            ][0]
            prompt_len = len(prompt_tokens)
            labels[:prompt_len] = -100

        # Set labels to -100 where attention mask is 0, but don't mask the last EOS token
        # Tokenizer by default does not include EOS in the attention mask
        seq_len = attention_mask.sum()
        labels[seq_len + 1 :] = -100

        output_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "text": full_text,
        }
        return output_dict

    return preprocess_fn
