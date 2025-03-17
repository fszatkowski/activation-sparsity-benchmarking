import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset
from trl import SFTTrainer

from train.training_utils import wandb_log_if_enabled

LOSSES = ["l1", "l2", "hoyer"]
SPARSIFICATION_MODES = ["input", "output"]


class SparsificationHook:
    def __init__(self, module_name):
        self.enabled = True
        self.module_name = module_name
        self.value = None


class InputSparsificationHook(SparsificationHook):
    def __call__(self, module, inputs):
        if self.enabled:
            self.value = inputs[0]
        else:
            self.value = None


class OutputSparsificationHook(SparsificationHook):
    def __call__(self, module, inputs, outputs):
        if self.enabled:
            self.value = outputs[0]
        else:
            self.value = None


def l1_loss(tensor: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    # Tensor: (batch_size, seq_len, hidden_size)
    # attn_mask: (batch_size, seq_len)
    l1_norm = torch.linalg.norm(
        tensor, ord=1.0, dim=-1, dtype=torch.float32
    )  # (batch_size, seq_len)
    masked_norm = attn_mask * l1_norm
    return masked_norm.sum() / attn_mask.sum()


def l2_loss(tensor: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    # Tensor: (batch_size, seq_len, hidden_size)
    # attn_mask: (batch_size, seq_len)
    l2_norm = torch.linalg.norm(
        tensor, ord=2.0, dim=-1, dtype=torch.float32
    )  # (batch_size, seq_len)
    masked_norm = attn_mask * l2_norm
    return masked_norm.sum() / attn_mask.sum()


def hoyer_loss(tensor: torch.Tensor, attn_mask: torch.Tensor, eps=1e-6) -> torch.Tensor:
    # Tensor: (batch_size, seq_len, hidden_size)
    # attn_mask: (batch_size, seq_len)
    l1_norm = torch.linalg.norm(
        tensor, ord=1.0, dim=-1, dtype=torch.float32
    )  # (batch_size, seq_len)
    l2_norm = torch.linalg.norm(
        tensor, ord=2.0, dim=-1, dtype=torch.float32
    )  # (batch_size, seq_len)

    per_token_hoyer_loss = l1_norm**2 / (l2_norm**2 + eps)  # (batch_size, seq_len)
    masked_hoyer_loss = attn_mask * per_token_hoyer_loss
    return masked_hoyer_loss.sum() / attn_mask.sum()


class SparsityEnforcementTrainer(SFTTrainer):
    def __init__(
        self,
        loss_type: str,
        loss_weight: float,
        modules_to_sparsify: List[str],
        sparsification_modes: List[str],
        modules_to_monitor: List[str],
        monitor_modes: List[str],
        monitor_top_p: List[float],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.loss_weight = loss_weight
        if loss_type == "l1":
            self.sparsification_loss = l1_loss
        elif loss_type == "l2":
            self.sparsification_loss = l2_loss
        elif loss_type == "hoyer":
            self.sparsification_loss = hoyer_loss
        else:
            raise NotImplementedError(
                f"Loss type {loss_type} not implemented. Supported losses: {LOSSES}."
            )

        self.sparsification_hooks = []
        if self.loss_weight != 0:
            for module_name, sparsification_mode in zip(
                modules_to_sparsify, sparsification_modes
            ):
                modules_to_sparsify = [
                    (name, module)
                    for name, module in self.model.named_modules()
                    if name.endswith(module_name)
                ]
                modules_to_sparsify = sorted(modules_to_sparsify, key=lambda x: x[0])
                for name, module in modules_to_sparsify:
                    if sparsification_mode == "input":
                        hook = InputSparsificationHook(name)
                        module.register_forward_pre_hook(hook)
                    elif sparsification_mode == "output":
                        hook = OutputSparsificationHook(name)
                        module.register_forward_hook(hook)
                    else:
                        raise ValueError(
                            f"Sparsification mode must be one of {SPARSIFICATION_MODES}"
                        )
                    self.sparsification_hooks.append(hook)

        self.monitor_top_p = monitor_top_p
        self.monitor_hooks = []
        self.monitor_hooks_by_group = defaultdict(list)
        for module_name, monitor_mode in zip(modules_to_monitor, monitor_modes):
            modules_to_monitor = [
                (name, module)
                for name, module in self.model.named_modules()
                if name.endswith(module_name)
            ]
            modules_to_monitor = sorted(modules_to_monitor, key=lambda x: x[0])
            for name, module in modules_to_monitor:
                if monitor_mode == "input":
                    hook = InputSparsificationHook(name)
                    module.register_forward_pre_hook(hook)
                elif monitor_mode == "output":
                    hook = OutputSparsificationHook(name)
                    module.register_forward_hook(hook)
                else:
                    raise ValueError(
                        f"Monitor mode must be one of {SPARSIFICATION_MODES}"
                    )
                self.monitor_hooks.append(hook)
                self.monitor_hooks_by_group[module_name].append(hook)

        self._sparsity_metrics = defaultdict(list)
        self._total_num_neurons = defaultdict(list)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            (loss, outputs) = super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch
            )
        else:
            loss = super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch
            )

        self._metrics["loss_cross_entropy"].append(loss.mean().item())
        if self.loss_weight != 0:
            # Iterate over sparsification hooks and summarize the loss
            sparsification_loss = 0
            attn_mask = inputs["attention_mask"]
            activation_vals = [hook.value for hook in self.sparsification_hooks]
            sparsification_losses = [
                self.sparsification_loss(activation_val, attn_mask)
                for activation_val in activation_vals
            ]
            sparsification_loss = torch.stack(sparsification_losses).mean()
            self._metrics["loss_sparsity"].append(sparsification_loss.item())
            loss += self.loss_weight * sparsification_loss

        return (loss, outputs) if return_outputs else loss

    def enable_hooks(self, hooks: List[SparsificationHook]) -> None:
        for hook in hooks:
            hook.enabled = True

    def disable_hooks(self, hooks: List[SparsificationHook]) -> None:
        for hook in hooks:
            hook.enabled = False

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:  # Enable monitor hooks and disable sparsification hooks
        self.enable_hooks(self.monitor_hooks)

        # Restart sparsity metrics
        self._sparsity_metrics = defaultdict(list)
        self._total_num_neurons = defaultdict(list)

        eval_outputs = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Compute and log sparsity metrics
        if len(self.monitor_hooks) > 0:
            metrics = {}
            for group_name, group_hooks in self.monitor_hooks_by_group.items():
                for topp_val in self.monitor_top_p:
                    group_total_sparse_neurons = 0
                    group_total_neurons = 0
                    for hook in group_hooks:
                        num_neurons = sum(self._total_num_neurons[hook.module_name])
                        topp_sparsity_key = f"{hook.module_name}_{topp_val}"
                        sparsity_values = sum(self._sparsity_metrics[topp_sparsity_key])
                        sparsity = sparsity_values / num_neurons

                        module_name = hook.module_name.split(".")[-1]
                        # Find layer index with regex if possible
                        # Look for .<number>. in the module name
                        layer_index = None
                        match = re.search(r"\.(\d+)\.", hook.module_name)
                        if match is not None:
                            layer_index = match.group(1)
                        if layer_index is not None:
                            module_name = f"{module_name}_{layer_index}"

                        metrics[f"sparsity_topp_{topp_val}/{module_name}"] = float(
                            sparsity
                        )

                        group_total_sparse_neurons += sparsity_values
                        group_total_neurons += num_neurons
                    group_sparsity = group_total_sparse_neurons / group_total_neurons
                    group_name = group_name.replace(".", "_")
                    metrics[f"sparsity_topp_{topp_val}/avg_{group_name}"] = float(
                        group_sparsity
                    )

            # If wandb is enabled, log metrics
            wandb_log_if_enabled(metrics)

        # Disable monitor hooks and enable back training hooks
        self.disable_hooks(self.monitor_hooks)

        return eval_outputs

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        outputs = super().prediction_step(
            model=model,
            inputs=inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )

        # Compute sparsity over the monitored modules
        with torch.no_grad():
            if len(self.monitor_hooks) > 0:
                attn_mask = inputs["attention_mask"]
                total_num_neurons = [0 for _ in range(attn_mask.shape[0])]
                for hook in self.monitor_hooks:
                    per_token_norms = hook.value.abs().sum(dim=-1).unsqueeze(-1)
                    total_num_neurons = (
                        attn_mask.sum(-1) * hook.value.shape[-1]
                    ).tolist()

                    sorted_neuron_norms, _ = hook.value.abs().sort(
                        descending=True, dim=-1
                    )
                    activation_cumsum = sorted_neuron_norms.cumsum(
                        dim=-1, dtype=torch.float32
                    )
                    for topp_val in self.monitor_top_p:
                        sparse_neurons_mask = activation_cumsum >= (
                            per_token_norms * topp_val
                        )
                        num_sparse_neurons = (
                            (sparse_neurons_mask * attn_mask.unsqueeze(-1))
                            .sum(-1)
                            .sum(-1)
                        )
                        self._sparsity_metrics[f"{hook.module_name}_{topp_val}"].extend(
                            num_sparse_neurons.tolist()
                        )
                    self._total_num_neurons[hook.module_name].extend(total_num_neurons)
        return outputs
