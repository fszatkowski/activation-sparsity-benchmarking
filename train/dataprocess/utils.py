import os

from datasets import load_dataset
from transformers import AutoTokenizer

from train.args import FinetuningArguments
from train.dataprocess.alpaca import get_alpaca_preprocess_fn
from train.dataprocess.fineweb import get_fineweb_preprocess_fn
from train.dataprocess.tulu import get_tulu_preprocess_fn


def prepare_dataset(training_args: FinetuningArguments, tokenizer: AutoTokenizer):
    dataset = load_dataset(
        training_args.dataset_name, split=training_args.dataset_split
    )
    dataset = dataset.shuffle(seed=training_args.seed)
    dataset = dataset.train_test_split(
        test_size=training_args.test_size, seed=training_args.seed
    )

    preprocess = training_args.preprocess
    if preprocess == "chat_template":
        raise NotImplementedError(
            "Preprocessing with chat template is not implemented yet."
        )

    if training_args.dataset_name == "yahma/alpaca-cleaned":
        preprocess_fn = get_alpaca_preprocess_fn(training_args, tokenizer)
    elif training_args.dataset_name == "allenai/tulu-3-sft-mixture":
        preprocess_fn = get_tulu_preprocess_fn(training_args, tokenizer)
    elif training_args.dataset_name == "HuggingFaceFW/fineweb":
        preprocess_fn = get_fineweb_preprocess_fn(training_args, tokenizer)
    else:
        raise NotImplementedError(
            "Dataset {} is not implemented".format(training_args.dataset_name)
        )

    for split in dataset:
        dataset[split] = dataset[split].map(
            preprocess_fn,
            num_proc=os.cpu_count() - 1,
            remove_columns=dataset[split].column_names,
        )

    train_set = dataset["train"]
    total_tokens = sum([sum(mask) for mask in train_set["attention_mask"]])
    valid_tokens = sum(
        [sum([l != -100 for l in labels]) for labels in train_set["labels"]]
    )
    print(f"Total tokens across training sequences: {total_tokens/1e9}B")
    print(
        f"Total valid tokens for loos propagation: {valid_tokens/1e9}B ({valid_tokens/total_tokens:.2%})"
    )
    return dataset
