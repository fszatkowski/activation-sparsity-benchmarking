from datasets import load_dataset
from transformers import (
    AutoTokenizer,
)

from train.args import FinetuningArguments


def prepare_dataset(training_args: FinetuningArguments, tokenizer: AutoTokenizer):
    dataset = load_dataset(training_args.dataset_name, split="train")
    dataset = dataset.shuffle()

    # Split dataset into train and test
    dataset = dataset.train_test_split(test_size=training_args.test_size)

    if training_args.train_size_limit is not None:
        dataset["train"] = dataset["train"].select(
            range(training_args.train_size_limit)
        )

    def format_chat_template(row):
        # TODO do we need system prompt here?

        instruction_text = row["instruction"]
        input_text = row["input"]

        if instruction_text == "" and input_text == "":
            raise ValueError()
        elif instruction_text == "":
            full_input = input_text
        elif input_text == "":
            full_input = instruction_text
        else:
            full_input = instruction_text + " " + input_text
        assert row["output"] != ""

        row_json = [
            {"role": "user", "content": full_input},
            {"role": "assistant", "content": row["output"]},
        ]

        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    dataset = dataset.map(
        format_chat_template,
        num_proc=4,
    )
    # Drop unused colums, keep only text
    column_names = dataset["train"].column_names
    columns_to_drop = [name for name in column_names if "text" not in name]
    dataset = dataset.remove_columns(columns_to_drop)

    return dataset
