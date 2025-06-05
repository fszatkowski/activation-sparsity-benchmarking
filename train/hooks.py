from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch

from train.sparsity_losses import ActivationSparsityLoss
from train.sparsity_metrics import SparsityMetric

LOSSES = ["l1", "l2", "hoyer", "none"]
SPARSIFICATION_MODES = ["input", "output"]


class BaseHook(ABC):
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.active = False
        self.attn_mask: Optional[torch.Tensor] = None

    @abstractmethod
    def _call(self, activation_vector: torch.Tensor):
        pass

    def set_attn_mask(self, attn_mask: torch.Tensor):
        self.attn_mask = attn_mask

    def enable(self):
        self.active = True

    @abstractmethod
    def disable(self):
        pass

    def __repr__(self):
        return f"({self.__class__.__name__ } - {self.module_name})"


class SparsificationHook(BaseHook):
    def __init__(
        self,
        module_name: str,
        sparsity_loss: ActivationSparsityLoss,
    ):
        super().__init__(module_name)

        self.loss_fn = sparsity_loss

        self.loss_val: Optional[torch.Tensor] = None
        self.train_metrics: Dict[str, Optional[float]] = {}

    def _call(self, activation_vector: torch.Tensor):
        # General hook call signature that can be applied to both forward and forward_pre hooks
        # activation_vector: (batch_size, seq_len, hidden_size)
        if not self.active:
            return

        assert (
            self.attn_mask is not None
        ), "Attention mask is not set. Please set it before calling the hook."

        # Compute the loss
        loss = self.loss_fn(tensor=activation_vector, attn_mask=self.attn_mask)
        # Distribute loss per token
        seq_lens = self.attn_mask.sum(dim=-1)
        per_token_loss = (loss * seq_lens).sum() / seq_lens.sum()
        self.loss_val = per_token_loss
        self.train_metrics["sparsity_loss"] = per_token_loss.item()

    def disable(self):
        self.active = False
        self.loss_val = None
        self.attn_mask = None
        self.train_metrics = {"sparsity_loss": None}


class InputSparsificationHook(SparsificationHook):
    def __call__(self, module, inputs):
        activation_vector = inputs[0]
        self._call(activation_vector)


class OutputSparsificationHook(SparsificationHook):
    def __call__(self, module, inputs, outputs):
        activation_vector = outputs[0]
        self._call(activation_vector)


class SparsityMonitorHook(BaseHook):
    def __init__(
        self,
        module_name: str,
        metrics: List[SparsityMetric],
    ):
        super().__init__(module_name)

        self._metrics = metrics

        self.eval_metrics: Dict[str, Optional[List[float]]] = {
            **{metric.name: None for metric in metrics},
            "token_counts": None,
        }

    @torch.inference_mode()
    def _call(self, activation_vector: torch.Tensor):
        # General hook call signature that can be applied to both forward and forward_pre hooks
        # activation_vector: (batch_size, seq_len, hidden_size)
        if not self.active:
            return

        assert (
            self.attn_mask is not None
        ), "Attention mask is not set. Please set it before calling the hook."

        with torch.no_grad():
            # Compute the metrics
            for metric in self._metrics:
                metric_val = metric(activation_vector, self.attn_mask)
                self.eval_metrics[metric.name] = metric_val.tolist()

            # Compute token counts per batch
            token_counts = self.attn_mask.sum(dim=-1)
            self.eval_metrics["token_counts"] = token_counts.tolist()

    def disable(self):
        self.active = False
        self.attn_mask = None
        self.eval_metrics = {
            **{metric.name: None for metric in self._metrics},
            "token_counts": None,
        }


class InputSparsityMonitorHook(SparsityMonitorHook):
    def __call__(self, module, inputs):
        activation_vector = inputs[0]
        self._call(activation_vector)


class OutputSparsityMonitorHook(SparsityMonitorHook):
    def __call__(self, module, inputs, outputs):
        activation_vector = outputs[0]
        self._call(activation_vector)


def create_hooks(
    model: torch.nn.Module,
    modules_to_sparsify: List[str],
    sparsification_modes: List[str],
    modules_to_monitor: List[str],
    monitor_modes: List[str],
    loss: Optional[ActivationSparsityLoss],
    metrics: List[SparsityMetric],
) -> Tuple[List[SparsificationHook], List[SparsityMonitorHook]]:
    sparsification_hooks = []
    if loss is not None:
        for module_name, mode in zip(modules_to_sparsify, sparsification_modes):
            module_name_and_module = [
                (name, module)
                for name, module in model.named_modules()
                if name.endswith(module_name)
            ]

            for name, module in module_name_and_module:
                sparsification_hook: SparsificationHook
                if mode == "input":
                    sparsification_hook = InputSparsificationHook(
                        module_name=name,
                        sparsity_loss=loss,
                    )
                    module.register_forward_pre_hook(sparsification_hook)
                elif mode == "output":
                    sparsification_hook = OutputSparsificationHook(
                        module_name=name,
                        sparsity_loss=loss,
                    )
                    module.register_forward_hook(sparsification_hook)
                else:
                    raise ValueError(f"Invalid sparsification mode: {mode}")
                sparsification_hooks.append(sparsification_hook)

    monitor_hooks = []
    for module_name, mode in zip(modules_to_monitor, monitor_modes):
        module_name_and_module = [
            (name, module)
            for name, module in model.named_modules()
            if name.endswith(module_name)
        ]

        for name, module in module_name_and_module:
            monitor_hook: SparsityMonitorHook
            if mode == "input":
                monitor_hook = InputSparsityMonitorHook(
                    module_name=name,
                    metrics=metrics,
                )
                module.register_forward_pre_hook(monitor_hook)
            elif mode == "output":
                monitor_hook = OutputSparsityMonitorHook(
                    module_name=name,
                    metrics=metrics,
                )
                module.register_forward_hook(monitor_hook)
            else:
                raise ValueError(f"Invalid monitor mode: {mode}")
            monitor_hooks.append(monitor_hook)

    return sparsification_hooks, monitor_hooks
