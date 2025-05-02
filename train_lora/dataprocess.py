import random

from datasets import load_dataset
from transformers import AutoTokenizer

from train.args import FinetuningArguments


def prepare_dataset(training_args: FinetuningArguments, tokenizer: AutoTokenizer):
    dataset = load_dataset(
        training_args.dataset_name, split=training_args.dataset_split
    )
    dataset = dataset.shuffle()

    # Split dataset into train and test
    dataset = dataset.train_test_split(test_size=training_args.test_size)

    preprocess = training_args.preprocess
    if preprocess == "chat_template":

        def format_fn(row):
            if "instruction" in row and "output" in row:
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
            elif "messages" in row:
                row_json = row["messages"]

            row["text"] = tokenizer.apply_chat_template(
                row_json, tokenize=False, return_assistant_tokens_mask=True
            )
            return row

    elif preprocess == "none":

        def format_fn(row):
            if "instruction" in row and "output" in row:
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
                output = row["output"]
                text = full_input + " " + output
            elif "messages" in row:
                txt = ""
                for msg in row["messages"]:
                    if msg["role"] == "user":
                        # Randomly append either "Q: " or "Question: " or nothing
                        rand_num = random.random()
                        if 0.0 <= rand_num < 0.33:
                            txt += "Q: " + msg["content"]
                        elif 0.33 <= rand_num < 0.66:
                            txt += "Question: " + msg["content"]
                        else:
                            txt += msg["content"]
                    elif msg["role"] == "assistant":
                        # Randomly append either "A: " or "Answer: " or nothing
                        rand_num = random.random()
                        if 0.0 <= rand_num < 0.33:
                            txt += "A: " + msg["content"]
                        elif 0.33 <= rand_num < 0.66:
                            txt += "Answer: " + msg["content"]
                        else:
                            txt += msg["content"]
                text = txt

            row["text"] = text
            return row

    else:
        raise NotImplementedError("Preprocess {} is not implemented".format(preprocess))

    dataset = dataset.map(
        format_fn,
        num_proc=4,
    )

    # Drop unused colums, keep only text
    column_names = dataset["train"].column_names
    columns_to_drop = [name for name in column_names if "text" not in name]
    dataset = dataset.remove_columns(columns_to_drop)

    return dataset
