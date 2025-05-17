from datasets import load_dataset
from transformers import AutoTokenizer

from train.args import FinetuningArguments
from train.dataprocess.alpaca import get_alpaca_preprocess_fn
from train.dataprocess.fineweb import get_fineweb_preprocess_fn


def prepare_dataset(training_args: FinetuningArguments, tokenizer: AutoTokenizer):
    if "fineweb" in training_args.dataset_name.lower():
        # Load just sample 10BT for fineweb dataset to avoid excess downloads
        dataset = load_dataset(
            training_args.dataset_name,
            name="sample-10BT",
            split=training_args.dataset_split,
        )
    else:
        dataset = load_dataset(
            training_args.dataset_name, split=training_args.dataset_split
        )
    dataset = dataset.shuffle(seed=training_args.seed)

    if "alpaca" in training_args.dataset_name.lower():
        preprocess_fn = get_alpaca_preprocess_fn(training_args, tokenizer)
    elif "fineweb" in training_args.dataset_name.lower():
        preprocess_fn = get_fineweb_preprocess_fn(training_args, tokenizer)
    else:
        raise NotImplementedError(
            "Dataset {} is not implemented".format(training_args.dataset_name)
        )

    dataset = dataset.map(
        preprocess_fn,
        num_proc=4,
        remove_columns=dataset.column_names,
    )

    # Filter out the examples with all labels masked out to -100
    dataset = dataset.filter(
        lambda example: any(label != -100 for label in example["labels"]),
        num_proc=4,
    )
    dataset = dataset.train_test_split(
        test_size=training_args.test_size, seed=training_args.seed
    )

    train_set = dataset["train"]

    # Use map to calculate total and valid tokens
    token_sums = train_set.map(
        lambda example: {
            "total_tokens": sum(example["attention_mask"]) + 1,
            "valid_tokens": sum(label != -100 for label in example["labels"]),
        },
        num_proc=4,
        desc="Counting tokens for training...",
    )

    # Aggregate the results
    total_tokens = sum(token_sums["total_tokens"])
    valid_tokens = sum(token_sums["valid_tokens"])

    # Determine the unit (M or B) and print the results
    unit, divisor = ("B", 1e9) if total_tokens > 1e9 else ("M", 1e6)
    print(f"Total tokens across training sequences: {total_tokens/divisor:.3}{unit}")
    print(
        f"Total valid tokens for loss propagation: {valid_tokens/divisor:.3}{unit} "
        f"({valid_tokens/total_tokens:.2%})"
    )

    return dataset
