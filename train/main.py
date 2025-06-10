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

    if sparsity_enforcement_args.kd_loss_weight > 0:
        if training_args.fp16:
            torch_dtype = torch.float16
        elif training_args.bf16:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        teacher = AutoModelForCausalLM.from_pretrained(
            sparsity_enforcement_args.teacher_model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            attn_implementation=training_args.attn_implementation,
            device_map="auto",
            torch_dtype=torch_dtype,
        )
    else:
        teacher = None

    dataset = prepare_dataset(training_args, tokenizer)

    trainer = SparsityEnforcementTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        tokenizer=tokenizer,
        sparsity_loss_type=sparsity_enforcement_args.loss_type,
        sparsity_loss_weight=sparsity_enforcement_args.loss_weight,
        sparsity_loss_shift=sparsity_enforcement_args.loss_shift,
        modules_to_sparsify=sparsity_enforcement_args.modules_to_sparsify,
        sparsification_modes=sparsity_enforcement_args.sparsification_modes,
        modules_to_monitor=sparsity_enforcement_args.modules_to_monitor,
        monitor_modes=sparsity_enforcement_args.monitor_modes,
        sparsity_metrics=sparsity_enforcement_args.sparsity_metrics,
        teacher=teacher,
        kd_loss_weight=sparsity_enforcement_args.kd_loss_weight,
        kd_temperature=sparsity_enforcement_args.kd_temperature,
        relufication=sparsity_enforcement_args.relufication,
        relufication_target_modules=sparsity_enforcement_args.relufication_target_modules,
        relufication_mode=sparsity_enforcement_args.relufication_mode,
    )
    trainer.train()

    if trainer.relufication:
        model.config.hidden_act = "relu"

    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
