from typing import Callable

from transformers import AutoTokenizer

from train.args import FinetuningArguments


def get_fineweb_preprocess_fn(
    training_args: FinetuningArguments, tokenizer: AutoTokenizer
) -> Callable:
    raise NotImplementedError(
        "Dataset {} is not implemented".format(training_args.dataset_name)
    )
