import os
from dataclasses import asdict

import wandb


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
