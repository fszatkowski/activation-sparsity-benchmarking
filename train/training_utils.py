import os
import random
from dataclasses import asdict

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def count_tokens(dataloader: DataLoader, num_epochs: int) -> int:
    # Count number of training tokens in the dataset
    num_tokens = 0
    for batch in dataloader:
        training_tokens = batch["attention_mask"].sum().item()
        num_tokens += training_tokens

    num_tokens *= num_epochs
    print(f"Training on total {num_tokens / 1e9}B tokens.")
    return num_tokens


def wandb_initialize(args_to_log):
    assert "WANDB_ENTITY" in os.environ, "Please set WANDB_ENTITY environment variable."
    assert (
        "WANDB_PROJECT" in os.environ
    ), "Please set WANDB_PROJECT environment variable."
    wandb_entity = os.environ["WANDB_ENTITY"]
    wandb_project = os.environ["WANDB_PROJECT"]

    if "LOCAL_RANK" in os.environ:
        if os.environ["LOCAL_RANK"] == "0":
            wandb.init(
                entity=wandb_entity,
                project=wandb_project,
            )
            for args in args_to_log:
                wandb.config.update(asdict(args))
    else:
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
        )
        for args in args_to_log:
            wandb.config.update(asdict(args))


def wandb_log_if_enabled(metrics):
    if wandb.run is not None:
        if "LOCAL_RANK" in os.environ:
            if os.environ["LOCAL_RANK"] == "0":
                wandb.log(metrics)
        else:
            wandb.log(metrics)
