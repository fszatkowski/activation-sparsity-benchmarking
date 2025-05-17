from typing import Callable

from transformers import AutoTokenizer

from train.args import FinetuningArguments


def get_fineweb_preprocess_fn(
    training_args: FinetuningArguments, tokenizer: AutoTokenizer
) -> Callable:
    mask_prompt_labels = training_args.mask_prompt
    max_seq_length = training_args.max_seq_length
    if mask_prompt_labels:
        raise ValueError(
            "mask_prompt_labels is not supported for fineweb dataset. Please set it to False."
        )

    def preprocess_fn(row):
        text = row["text"]
        # Tokenize the full prompt + output
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        labels = input_ids.clone()

        # Set labels to -100 where attention mask is 0, but don't mask the last EOS token
        # Tokenizer by default does not include EOS in the attention mask
        seq_len = attention_mask.sum()
        labels[seq_len + 1 :] = -100

        output_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "text": text,
        }
        return output_dict

    return preprocess_fn
