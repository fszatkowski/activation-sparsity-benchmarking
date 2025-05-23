import random

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from train.args import FinetuningArguments, SparsityEnforcementArguments
from train.dataprocess.utils import prepare_dataset
from train.trainer import SparsityEnforcementTrainer
from train.wandb_utils import wandb_initialize


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    parser = HfArgumentParser((FinetuningArguments, SparsityEnforcementArguments))
    training_args, sparsity_enforcement_args = parser.parse_args_into_dataclasses()

    if "wandb" in training_args.report_to:
        # Handle logging for the distributed runs
        wandb_initialize(args_to_log=[training_args, sparsity_enforcement_args])

    set_seed(training_args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        training_args.model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        attn_implementation=training_args.attn_implementation,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    dataset = prepare_dataset(training_args, tokenizer)

    trainer = SparsityEnforcementTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        tokenizer=tokenizer,
        loss_type=sparsity_enforcement_args.loss_type,
        loss_weight=sparsity_enforcement_args.loss_weight,
        modules_to_sparsify=sparsity_enforcement_args.modules_to_sparsify,
        sparsification_modes=sparsity_enforcement_args.sparsification_modes,
        modules_to_monitor=sparsity_enforcement_args.modules_to_monitor,
        monitor_modes=sparsity_enforcement_args.monitor_modes,
        sparsity_metrics=sparsity_enforcement_args.sparsity_metrics,
    )
    trainer.train()

    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
