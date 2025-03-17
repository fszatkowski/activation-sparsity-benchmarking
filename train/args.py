from dataclasses import dataclass, field
from typing import List, Optional

from trl import SFTConfig


@dataclass
class FinetuningArguments(SFTConfig):
    model_name: str = field(
        default="meta-llama/Llama-3.2-1B-Instruct",
        metadata={"help": "The name of the model to use for fine-tuning."},
    )
    dataset_name: str = field(
        default="yahma/alpaca-cleaned",
        metadata={"help": "The name of the dataset to use for fine-tuning."},
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LORA."},
    )
    train_size_limit: Optional[int] = field(
        default=None,
        metadata={"help": "The size of the training dataset."},
    )
    test_size: float = field(
        default=0.01,
        metadata={"help": "The size of the test dataset."},
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


@dataclass
class LORAArguments:
    lora_rank: int = field(
        default=32,
        metadata={"help": "The rank of the LORA matrix."},
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "The alpha parameter of the LORA matrix."},
    )
    lora_dropout: float = field(
        default=0.005,
        metadata={"help": "The dropout rate of the LORA matrix."},
    )
    lora_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "up_proj",
            "down_proj",
            "o_proj",
            "gate_proj",
        ],
        metadata={"help": "The modules to apply LORA to."},
    )
