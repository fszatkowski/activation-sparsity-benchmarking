from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from train.args import FinetuningArguments, LORAArguments, SparsityEnforcementArguments
from train.dataprocess import prepare_dataset
from train.trainer import SparsityEnforcementTrainer
from train.training_utils import count_tokens, set_seed, wandb_initialize

# Use flash attentnion if the package is installed
try:
    import flash_attn_interface

    attn_implementation = "flash_attention_2"
except ImportError:
    print(
        "Flash attention is not installed, using eager implementation. "
        "Please install the package to use it as the attention implementation."
    )
    attn_implementation = "eager"


if __name__ == "__main__":
    parser = HfArgumentParser(
        (FinetuningArguments, LORAArguments, SparsityEnforcementArguments)
    )
    training_args, lora_args, sparsity_enforcement_args = (
        parser.parse_args_into_dataclasses()
    )

    if "wandb" in training_args.report_to:
        # Handle logging for the distributed runs
        wandb_initialize(
            args_to_log=[training_args, lora_args, sparsity_enforcement_args]
        )

    set_seed(training_args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        training_args.model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        attn_implementation=attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    dataset = prepare_dataset(training_args, tokenizer)

    if training_args.use_lora:
        peft_config = LoraConfig(
            r=lora_args.lora_rank,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(
            model,
            peft_config=peft_config,
        )
        model.print_trainable_parameters()
    else:
        peft_config = None

    trainer = SparsityEnforcementTrainer(
        loss_type=sparsity_enforcement_args.loss_type,
        loss_weight=sparsity_enforcement_args.loss_weight,
        modules_to_sparsify=sparsity_enforcement_args.modules_to_sparsify,
        sparsification_modes=sparsity_enforcement_args.sparsification_modes,
        modules_to_monitor=sparsity_enforcement_args.modules_to_monitor,
        monitor_modes=sparsity_enforcement_args.monitor_modes,
        monitor_top_p=sparsity_enforcement_args.monitor_top_p,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        args=training_args,
        tokenizer=tokenizer,
    )

    count_tokens(
        dataloader=trainer.get_train_dataloader(),
        num_epochs=training_args.num_train_epochs,
    )

    trainer.train()

    if training_args.use_lora:
        # Merge PEFT model
        model = model.merge_and_unload()

    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
