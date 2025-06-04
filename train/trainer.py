from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from transformers import Trainer

from train.hooks import ActivationSparstityHook, create_hooks
from train.sparsity_losses import get_sparsity_loss
from train.sparsity_metrics import SparsityMetric, instantiate_sparsity_metrics
from train.wandb_utils import wandb_log_if_enabled

LOSSES = ["l1", "l2", "hoyer", "none"]
SPARSIFICATION_MODES = ["input", "output"]


class SparsityEnforcementTrainer(Trainer):
    def __init__(
        self,
        sparsity_loss_type: str,
        sparsity_loss_weight: float,
        sparsity_loss_shift: Optional[float],
        modules_to_sparsify: List[str],
        sparsification_modes: List[str],
        modules_to_monitor: List[str],
        monitor_modes: List[str],
        sparsity_metrics: List[str],
        kd_loss_weight: float = 0.0,
        kd_temperature: float = 1.0,
        teacher: Optional[torch.nn.Module] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if (sparsity_loss_type.lower() == "none" or sparsity_loss_weight == 0) and len(
            modules_to_sparsify
        ) > 0:
            print()
            print(
                "WARNING: Setting modules to sparsify with loss_function 'none' or loss_weight=0. "
                "This will not have any effect on the training updates, but will result in additonal memory consuption. "
                "Setting 'modules_to_sparsify' to empty list."
            )
            print()
            modules_to_sparsify = []

        self.sparsity_loss = get_sparsity_loss(
            sparsity_loss_type, sparsity_loss_weight, sparsity_loss_shift
        )

        # Find the overlap across the module, mode for sparsification and monitoring
        sparsify_modules_modes = [
            (module, mode)
            for module, mode in zip(modules_to_sparsify, sparsification_modes)
        ]
        monitor_modules_modes = [
            (module, mode) for module, mode in zip(modules_to_monitor, monitor_modes)
        ]
        all_modules_modes = list(set(sparsify_modules_modes + monitor_modules_modes))

        self.sparsity_metrics: List[SparsityMetric] = instantiate_sparsity_metrics(
            sparsity_metrics
        )
        self.eval_metrics: Dict[str, Dict[str, List[float]]] = {}
        self.sparsification_hooks: List[ActivationSparstityHook] = []

        for module, mode in all_modules_modes:
            sparsify = (module, mode) in sparsify_modules_modes
            monitor = (module, mode) in monitor_modules_modes
            hooks = create_hooks(
                model=self.model,
                module_name=module,
                mode=mode,
                loss=self.sparsity_loss,
                metrics=self.sparsity_metrics,
                sparsify=sparsify,
                monitor=monitor,
            )
            self.sparsification_hooks.extend(hooks)

        if kd_loss_weight:
            assert teacher is not None, "Teacher model must be provided for KD loss"
            # Remove the gradient tracking for the teacher model
            for param in teacher.parameters():
                param.requires_grad = False
            # Set the teacher model to evaluation mode
            teacher.eval()
        self.teacher = teacher
        self.kd_loss_weight = kd_loss_weight
        self.kd_temperature = kd_temperature

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Manually set the attention mask for sparsity hooks to enable masking
        for hook in self.sparsification_hooks:
            hook.set_train_mode()
            hook.set_attn_mask(inputs["attention_mask"])

        return super().training_step(model, inputs, num_items_in_batch)

    def _kld_loss(
        self,
        pred_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Compute KLD loss between the student and teacher logits,
        # Using targets to mask off the padded tokens
        # Compute probs including self.temperature
        if self.kd_temperature != 1.0:
            pred_logits = pred_logits / self.kd_temperature
            teacher_logits = teacher_logits / self.kd_temperature
        pred_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
        teacher_probs = torch.nn.functional.log_softmax(teacher_logits, dim=-1)

        kld_loss = torch.nn.functional.kl_div(
            pred_probs, teacher_probs, reduction="none", log_target=True
        )
        kld_loss = kld_loss.mean(dim=-1)
        # Mask the loss using the targets
        mask = targets != -100
        kld_loss = (kld_loss * mask).sum() / mask.sum()

        return kld_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs or self.teacher is not None:
            (ce_loss, outputs) = super().compute_loss(
                model, inputs, True, num_items_in_batch
            )
        else:
            ce_loss = super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch
            )

        # If any of the hooks are in training mode, also compute and log the sparsity loss
        if any(h.train for h in self.sparsification_hooks):
            sparsification_losses = [
                hook.loss_val
                for hook in self.sparsification_hooks
                if hook.loss_val is not None
            ]
            if len(sparsification_losses) == 0:
                sparsification_loss = torch.zeros_like(ce_loss)
                wandb_log_if_enabled({"train_ce_loss/total": ce_loss.item()})
            else:
                sparsification_loss = torch.mean(torch.stack(sparsification_losses))
                # Multiply by gradient accumulation steps to match the loss logged by default in HF trainer
                self._handle_train_sparsity_loss_metrics(
                    ce_loss * self.args.gradient_accumulation_steps
                )

            loss = ce_loss + sparsification_loss
        else:
            loss = ce_loss

        # Compute the knowledge distillation loss if using a teacher model
        if self.teacher is not None and self.kd_loss_weight > 0:
            with torch.no_grad():
                teacher_logits = self.teacher(**inputs).logits.detach()
            kld_loss = self._kld_loss(outputs.logits, teacher_logits, inputs["labels"])
            wandb_log_if_enabled({"train_kd_loss/total": kld_loss.item()})
            loss += kld_loss * self.kd_loss_weight

        return (loss, outputs) if return_outputs else loss

    def _handle_train_sparsity_loss_metrics(self, ce_loss: torch.Tensor) -> None:
        # Log sparsity losses
        train_metrics = {}
        for hook in self.sparsification_hooks:
            if hook.loss_val is not None:
                for metric, value in hook.train_metrics.items():
                    train_metrics[f"train_{metric}/{hook.module_name}"] = value
        # Compute average over all the primary keys
        train_metrics_keys = set(k.split("/")[0] for k in train_metrics.keys())
        for key in train_metrics_keys:
            key_vals = np.array(
                [train_metrics[k] for k in train_metrics.keys() if k.startswith(key)]
            )
            train_metrics[f"{key}/avg"] = float(np.mean(key_vals))
        train_metrics["train_ce_loss/total"] = ce_loss.item()

        wandb_log_if_enabled(train_metrics)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:  # Enable monitor hooks and disable sparsification hooks
        # Restart sparsity metrics
        self.eval_metrics = {
            metric_name: defaultdict(list)
            for metric_name in self.sparsification_hooks[0].eval_metrics.keys()
        }
        eval_outputs = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        self._handle_eval_sparsity_metrics()

        # Reset the eval metrics and hooks at the end
        for hook in self.sparsification_hooks:
            hook.reset()
        self.eval_metrics = {}

        # Free unused memory
        torch.cuda.empty_cache()

        return eval_outputs

    def _handle_eval_sparsity_metrics(self) -> None:
        # Collapse eval metrics across all modules
        token_counts = {
            k: np.array(v) for k, v in self.eval_metrics.pop("token_counts").items()
        }
        metric_names = list(self.eval_metrics.keys())
        module_names = list(self.eval_metrics[metric_names[0]].keys())
        aggregated_metrics: Dict[str, Dict[str, float]] = {
            metric_names: {} for metric_names in metric_names
        }
        for metric_name in metric_names:
            metric_vals: List[float] = []
            for module_name in module_names:
                per_sequence_vals = np.array(
                    self.eval_metrics[metric_name][module_name]
                )
                per_token_val = (
                    per_sequence_vals * token_counts[module_name]
                ).sum() / token_counts[module_name].sum()
                metric_vals.append(float(per_token_val))
            avg_val = float(np.mean(metric_vals))
            aggregated_metrics[metric_name] = {
                "avg": avg_val,
                **{
                    module_name: metric_val
                    for module_name, metric_val in zip(module_names, metric_vals)
                },
            }

        wandb_metrics = {
            f"eval_{metric_name}/{module_name}": metric_val
            for metric_name, module_metrics in aggregated_metrics.items()
            for module_name, metric_val in module_metrics.items()
        }
        wandb_log_if_enabled(wandb_metrics)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Manually set the attention mask for sparsity hooks to enable masking
        for hook in self.sparsification_hooks:
            hook.set_eval_mode()
            hook.set_attn_mask(inputs["attention_mask"])

        outputs = super().prediction_step(
            model=model,
            inputs=inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )
        for hook in self.sparsification_hooks:
            if hook.monitor:
                module_name = hook.module_name
                for metric_name, metric_value in hook.eval_metrics.items():
                    assert (
                        metric_value is not None
                    ), f"Value is None for {metric_name} at {module_name} hook."
                    self.eval_metrics[metric_name][module_name].extend(metric_value)
        return outputs
