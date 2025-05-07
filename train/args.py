from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments


@dataclass
class FinetuningArguments(TrainingArguments):
    model_name: str = field(
        default="meta-llama/Llama-3.2-1B-Instruct",
        metadata={"help": "The name of the model to use for fine-tuning."},
    )
    dataset_name: str = field(
        default="yahma/alpaca-cleaned",
        metadata={"help": "The name of the dataset to use for fine-tuning."},
    )
    dataset_split: str = field(
        default="train",
        metadata={
            "help": "The split of the dataset to use for fine-tuning. "
            "Use train[:10] or train[:20%] to limit the dataset size."
        },
    )
    test_size: float = field(
        default=0.01,
        metadata={"help": "The size of the test dataset."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum sequence length of the dataset. "
            "Sequences longer than this will be truncated."
        },
    )
    mask_prompt: bool = field(
        default=False,
        metadata={
            "help": "Whether to mask loss corresponding to the prompt during training. "
            "If True, the prompt will be masked and only the response will be used for training."
        },
    )
    output_dir: str = field(
        default="outputs/models/test",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    wandb_tags: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Tags to use for wandb logging."},
    )
    attn_implementation: str = field(
        default="eager",
        metadata={
            "help": "The attention implementation to use. "
            "Use 'flash_attention_2' for flash attention and 'eager' for eager implementation."
        },
    )


@dataclass
class SparsityEnforcementArguments:
    loss_type: str = field(
        default="l1",
        metadata={"help": "The type of sparsity loss to use."},
    )
    loss_weight: float = field(
        default=0.01,
        metadata={"help": "The weight of the sparsity loss."},
    )
    modules_to_sparsify: List[str] = field(
        default_factory=lambda: ["mlp.down_proj"],
        metadata={"help": "The modules to sparsify."},
    )
    sparsification_modes: List[str] = field(
        default_factory=lambda: ["input"],
        metadata={"help": "The sparsification mode."},
    )
    modules_to_monitor: List[str] = field(
        default_factory=lambda: ["mlp", "mlp.down_proj"],
        metadata={"help": "The modules to monitor."},
    )
    monitor_modes: List[str] = field(
        default_factory=lambda: ["input", "input"],
        metadata={"help": "The monitor mode."},
    )
    monitor_top_p: List[float] = field(
        default_factory=lambda: [0.99, 0.9, 0.75, 0.5],
        metadata={"help": "The top p values to monitor during evaluation."},
    )
