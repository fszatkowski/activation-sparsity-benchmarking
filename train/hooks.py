from typing import Dict, List, Optional

import torch

from train.sparsity_losses import ActivationSparsityLoss
from train.sparsity_metrics import SparsityMetric

LOSSES = ["l1", "l2", "hoyer", "none"]
SPARSIFICATION_MODES = ["input", "output"]


class ActivationSparstityHook:
    """
    Base class for activation sparsity hooks. This class is used to compute the sparsity loss and metrics.
    It is used as a base class for both input and output activation sparsity hooks.
    Sparsity loss and metrics should be passed to enable training and evaluation functionalities in general.
    Whether or not the the sparsity loss is backpropagated during training and whether the metrics are computed for evaluation
      depends on the 'sparsfiy' and 'monitor' flags.
    """

    def __init__(
        self,
        module_name,
        sparsity_loss: Optional[ActivationSparsityLoss],
        metrics: List[SparsityMetric],
        sparsify: bool = True,
        monitor: bool = True,
    ):
        self.sparsify = sparsify
        self.monitor = monitor

        self.module_name = module_name
        self.disabled = True  # Disable by default
        self.train = False
        self.evaluate = False

        self.loss_fn = sparsity_loss
        self.loss_val: Optional[torch.Tensor] = None
        self.attn_mask: Optional[torch.Tensor] = None

        self.metrics = metrics
        self.train_metrics: Dict[str, Optional[float]] = {"sparsity_loss": None}
        self.eval_metrics: Dict[str, Optional[List[float]]] = {
            "sparsity_loss": None,
            **{metric.name: None for metric in metrics},
            "token_counts": None,
        }

    def set_attn_mask(self, attn_mask: torch.Tensor):
        self.attn_mask = attn_mask

    def disable(self):
        self.disabled = True
        self.train = False
        self.evaluate = False

        self.attn_mask = None
        self.loss_val = None

    def set_train_mode(self):
        self.disabled = False
        self.train = True
        self.evaluate = False

        self.attn_mask = None
        self.loss_val = None

    def set_eval_mode(self):
        self.disabled = False
        self.train = False
        self.evaluate = True

        self.attn_mask = None
        self.loss_val = None

    def call(self, activation_vector: torch.Tensor):
        # General hook call signature that can be applied to both forward and forward_pre hooks
        # activation_vector: (batch_size, seq_len, hidden_size)

        # If disabled, just return
        if self.disabled:
            return

        assert (
            self.attn_mask is not None
        ), "Attention mask is not set. Please set it before calling the hook."

        # Compute sparsity loss, if set
        if self.loss_fn is not None:
            # Compute the loss
            loss = self.loss_fn(tensor=activation_vector, attn_mask=self.attn_mask)
            # Distribute loss per token
            seq_lens = self.attn_mask.sum(dim=-1)
            per_token_loss = (loss * seq_lens).sum() / seq_lens.sum()
        else:
            loss = torch.zeros(
                activation_vector.shape[0], device=activation_vector.device
            )
            per_token_loss = loss.mean()

        if self.sparsify and self.train:
            self.loss_val = per_token_loss
            self.train_metrics["sparsity_loss"] = per_token_loss.item()
        else:
            self.train_metrics["sparsity_loss"] = 0

        if self.monitor and self.evaluate:
            # Compute the metrics
            for metric in self.metrics:
                metric_val = metric(activation_vector, self.attn_mask)
                self.eval_metrics[metric.name] = metric_val.tolist()

            # Compute token counts per batch
            token_counts = self.attn_mask.sum(dim=-1)
            self.eval_metrics["sparsity_loss"] = loss.tolist()
            self.eval_metrics["token_counts"] = token_counts.tolist()
        else:
            self.eval_metrics = {k: None for k in self.eval_metrics.keys()}


class InputActivationSparstityHook(ActivationSparstityHook):
    def __call__(self, module, inputs):
        activation_vector = inputs[0]
        self.call(activation_vector)


class OutputActivationSparstityHook(ActivationSparstityHook):
    def __call__(self, module, inputs, outputs):
        activation_vector = outputs[0]
        self.call(activation_vector)


def create_hooks(
    model: torch.nn.Module,
    module_name: str,
    mode: str,
    loss: ActivationSparsityLoss,
    metrics: List[SparsityMetric],
    sparsify: bool,
    monitor: bool,
) -> List[ActivationSparstityHook]:
    modules_to_sparsify = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(module_name)
    ]
    modules_to_sparsify = sorted(modules_to_sparsify, key=lambda x: x[0])
    assert (
        len(modules_to_sparsify) > 0
    ), f"No modules with name containing '{module_name}' found in the model."

    hooks: List[ActivationSparstityHook] = []
    for name, module in modules_to_sparsify:
        if mode == "input":
            input_hook: ActivationSparstityHook = InputActivationSparstityHook(
                module_name=name,
                sparsity_loss=loss,
                metrics=metrics,
                sparsify=sparsify,
                monitor=monitor,
            )
            module.register_forward_pre_hook(input_hook)
            hooks.append(input_hook)
        elif mode == "output":
            output_hook: ActivationSparstityHook = OutputActivationSparstityHook(
                module_name=name,
                sparsity_loss=loss,
                metrics=metrics,
                sparsify=sparsify,
                monitor=monitor,
            )
            module.register_forward_hook(output_hook)
            hooks.append(output_hook)
        else:
            raise ValueError(
                f"Sparsification mode must be one of {SPARSIFICATION_MODES}"
            )
    return hooks
