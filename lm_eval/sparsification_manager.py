import json
from abc import ABC
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


def get_topp_mask(tensor: torch.Tensor, topp_val: float) -> torch.Tensor:
    abs_tensor = tensor.abs()

    # Sort the activation values, the biggest activations appear first
    sorted_activations, _ = abs_tensor.sort(descending=True, dim=-1)

    # Calculate the cumsum - the total sum of activations up to this position
    # dtype is float64 to avoid overflow
    activation_cumsum = sorted_activations.cumsum(
        dim=-1, dtype=torch.float64
    )  # shape: [batch_size, seq_len, hidden_dim]
    # Sum of all activations is the last element of the cumsum
    activation_sums = activation_cumsum[:, :, -1].unsqueeze(
        -1
    )  # shape: [batch_size, seq_len, 1]

    # Find which activations to use to maintain at least topp percent of the total activation sum
    target_sum = activation_sums * topp_val
    exceeds_target = activation_cumsum >= target_sum

    # Get the first index where we exceed the target (or last index if none)
    cumsum_threshold_idx = exceeds_target.int().argmax(dim=-1)

    # Handle edge case where no position exceeds target (shouldn't happen with valid inputs)
    # In this case, set to all zeros
    no_exceeds = ~exceeds_target.any(dim=-1)
    cumsum_threshold_idx = torch.where(
        no_exceeds, torch.zeros_like(cumsum_threshold_idx), cumsum_threshold_idx
    )

    # Get the minimum activation threshold
    activation_threshold_value = sorted_activations.gather(
        -1, cumsum_threshold_idx.unsqueeze(-1)
    ).type(tensor.dtype)
    non_sparse_mask = abs_tensor >= activation_threshold_value

    return non_sparse_mask


def get_maxp_mask(tensor: torch.Tensor, maxp_val: float) -> torch.Tensor:
    abs_tensor = tensor.abs()
    max_activation_vals = abs_tensor.max(dim=-1).values
    threshold_vals = max_activation_vals * maxp_val
    non_sparse_mask = abs_tensor >= threshold_vals.unsqueeze(-1)
    return non_sparse_mask


def get_topk_mask(tensor: torch.Tensor, topk_val: float) -> torch.Tensor:
    feature_dim = tensor.shape[-1]
    topk_int_val = int(feature_dim * topk_val)
    abs_tensor = tensor.abs()

    # Find minimum top-k activations for each token
    topk_threshold_vals = (
        torch.topk(abs_tensor, topk_int_val, dim=-1, sorted=True)
        .values.min(dim=-1)
        .values
    )
    # Create a mask for the top-k activations
    non_sparse_mask = abs_tensor >= topk_threshold_vals.unsqueeze(-1)
    # Ensure the mask is applied only to non-padded activations
    non_sparse_mask = non_sparse_mask.bool()
    return non_sparse_mask


@torch.inference_mode()
def sparsify_tensor(
    tensor: torch.Tensor,
    rule: str,
    th_val: Optional[float] = None,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    # Check for the device of the input mask and move it to the device of the tensor
    if rule == "topp":
        assert th_val is not None, "Topp value must be provided for topp rule"
        non_sparse_mask = get_topp_mask(tensor, th_val)
    elif rule == "maxp":
        assert th_val is not None, "Maxp value must be provided for maxp rule"
        non_sparse_mask = get_maxp_mask(tensor, th_val)
    elif rule == "topk":
        assert th_val is not None, "Topk value must be provided for topk rule"
        non_sparse_mask = get_topk_mask(tensor, th_val)
    else:
        raise ValueError(f"Invalid sparsification rule: {rule}")

    sparsified_tensor = torch.where(non_sparse_mask, tensor, torch.zeros_like(tensor))
    sparse_neurons_count = (~non_sparse_mask).sum(-1).detach().cpu().tolist()
    batch_size, seq_len, feature_dim = tensor.shape
    total_num_neurons = [[feature_dim] * seq_len] * batch_size

    return sparsified_tensor, sparse_neurons_count, total_num_neurons


class SparsificationHook(ABC):
    def __init__(
        self,
        layer_name: str,
        rule: str,
        th_val: Optional[float] = None,
        compute_effective_rank: bool = False,
    ):
        self.layer_name = layer_name
        self.rule = rule
        self.th_val = th_val
        self.compute_effective_rank = compute_effective_rank

        # Per-token activation sparsity values for the current batch without masking
        self.current_batch_sparse_counts = []
        self.current_batch_total_counts = []
        self.current_batch_svd_activations = []

        # Total counts across all batches seen so far, taking the masking into account
        self.sparse_counts = []
        self.total_counts = []
        self.num_generated_tokens = []
        self.per_batch_effective_rank = []
        self.act_dim = None

    def restart(self):
        self.current_batch_sparse_counts = []
        self.current_batch_total_counts = []
        self.current_batch_svd_activations = []

        self.sparse_counts = []
        self.total_counts = []
        self.num_generated_tokens = []
        self.per_batch_effective_rank = []
        self.act_dim = None

    @torch.inference_mode()
    def generic_call(self, value):
        if isinstance(value, tuple):
            value = value[
                0
            ]  # Handle tuple values in case model operates on multiple ins / outs
        assert isinstance(value, torch.Tensor), "Value must be a torch.Tensor."
        org_shape = value.shape

        if self.act_dim is None:
            self.act_dim = org_shape[-1]

        sparsified_value, num_sparse_neurons, num_all_neurons = sparsify_tensor(
            value, self.rule, self.th_val
        )
        assert (
            sparsified_value.shape == org_shape
        ), "Sparsified value has different shape than the original."

        if self.current_batch_sparse_counts == []:
            self.current_batch_sparse_counts = num_sparse_neurons
            self.current_batch_total_counts = num_all_neurons
        else:
            batch_size = value.shape[0]
            for batch_idx in range(batch_size):
                self.current_batch_sparse_counts[batch_idx].extend(
                    num_sparse_neurons[batch_idx]
                )
                self.current_batch_total_counts[batch_idx].extend(
                    num_all_neurons[batch_idx]
                )

        if self.compute_effective_rank:
            self.current_batch_svd_activations.append(value.detach().cpu())

        return sparsified_value

    def consolidate_batch(self, batch_start_indices, batch_end_indices):
        batch_size = len(batch_start_indices)

        if self.compute_effective_rank:
            # Concatenate the activations along the sequence dimension
            self.current_batch_svd_activations = torch.cat(
                self.current_batch_svd_activations, dim=1
            )

        svd_flat_activations = []
        for batch_idx in range(batch_size):
            sample_sparse_counts = self.current_batch_sparse_counts[batch_idx]
            sample_total_counts = self.current_batch_total_counts[batch_idx]
            sample_start_index = batch_start_indices[batch_idx]
            sample_end_index = batch_end_indices[batch_idx]

            sample_sparse_counts = sample_sparse_counts[
                sample_start_index:sample_end_index
            ]
            sample_total_counts = sample_total_counts[
                sample_start_index:sample_end_index
            ]
            self.sparse_counts.append(sum(sample_sparse_counts))
            self.total_counts.append(sum(sample_total_counts))

            if self.compute_effective_rank:
                svd_flat_activations.append(
                    self.current_batch_svd_activations[
                        batch_idx, sample_start_index:sample_end_index
                    ]
                )

        if self.compute_effective_rank:
            svd_flat_activations = torch.cat(svd_flat_activations, dim=0)
            eps = 1e-10
            _, S, _ = torch.linalg.svd(
                svd_flat_activations.float().cuda(), full_matrices=False
            )
            S = S + eps  # Add eps to avoid log(0)
            p = S / S.sum()
            effective_rank = torch.exp(-(p * p.log()).sum())

            # Normalize by the feature dimension
            self.per_batch_effective_rank.append(effective_rank.item() / self.act_dim)

        self.current_batch_sparse_counts = []
        self.current_batch_total_counts = []
        self.current_batch_svd_activations = []


# Pytorch hook to enforce sparsity on the output of the module
class OutputSparsificationHook(SparsificationHook):
    @torch.inference_mode()
    def __call__(self, module, input_tensor, output_tensor):
        value = output_tensor
        return self.generic_call(value)


# Pytorch hook to enforce sparsity on the input to the module
class InputSparsificationHook(SparsificationHook):
    @torch.inference_mode()
    def __call__(self, module, input_tensor):
        value = input_tensor
        return self.generic_call(value)


class SparsificationManager:
    def __init__(
        self,
        sparsification_config_path: str,
        output_dir: str,
        sparsification_rule: str,
        th_val: Optional[float] = None,
        save_outputs: bool = False,
        compute_effective_rank: bool = False,
    ):
        self.lm = None
        self.sparsification_rule = sparsification_rule
        assert (
            th_val is not None
        ), "Threshold value must be provided for sparsified infreence."
        assert th_val >= 0, "Threshold value must be greater than or equal to 0."
        assert th_val <= 1, "Threshold value must be less than or equal to 1."
        self.th_val = th_val
        self.save_outputs = save_outputs
        self.compute_effective_rank = compute_effective_rank

        with Path(sparsification_config_path).open("r") as f:
            sparsity_config = json.load(f)

            self.layer_names_to_sparsify = sparsity_config.get(
                "layers_to_sparsify", None
            )
            self.layer_hook_modes = sparsity_config.get("hook_mode", None)
            self.embedding_layer_name = sparsity_config.get(
                "embedding_layer_name", None
            )

        self.output_dir = Path(output_dir)
        self.sparsity_hooks: List[SparsificationHook] = []

        self.num_generated_tokens = []
        self.request_input_ids = []
        self.request_output_ids = []
        self.request_input_texts = []
        self.request_output_texts = []
        self.batch_lengths = []

    def restart(self):
        for hook in self.sparsity_hooks:
            hook.restart()

        if self.save_outputs:
            self.request_input_ids = []
            self.request_output_ids = []
            self.request_input_texts = []
            self.request_output_texts = []
            self.num_generated_tokens = []
            self.batch_lengths = []

    def consolidate_batch(self, batch_start_indices, batch_end_indices):
        for hook in self.sparsity_hooks:
            hook.consolidate_batch(batch_start_indices, batch_end_indices)

    def save_model_inputs_and_outputs(
        self,
        context_ids,
        response_ids,
        decoded_contexts,
        decoded_responses,
        num_generated_tokens,
        batch_lengths,
    ):
        # Store the inputs and outputs of the model for each request for logging purposes
        if self.save_outputs:
            self.request_input_ids.extend(context_ids)
            self.request_output_ids.extend(response_ids)
            self.request_input_texts.extend(decoded_contexts)
            self.request_output_texts.extend(decoded_responses)
            self.num_generated_tokens.extend(num_generated_tokens)
            self.batch_lengths.extend(batch_lengths)

    def initialize(self, lm):
        assert self.layer_names_to_sparsify is not None

        modules_to_hook = [
            lm.model.get_submodule(eval_layer_name)
            for eval_layer_name in self.layer_names_to_sparsify
        ]

        for layer_name, hook_module, hook_mode in zip(
            self.layer_names_to_sparsify, modules_to_hook, self.layer_hook_modes
        ):
            if hook_mode == "output":
                hook = OutputSparsificationHook(
                    layer_name=layer_name,
                    rule=self.sparsification_rule,
                    th_val=self.th_val,
                    compute_effective_rank=self.compute_effective_rank,
                )
                hook_module.register_forward_hook(hook)
            elif hook_mode == "input":
                hook = InputSparsificationHook(
                    layer_name=layer_name,
                    rule=self.sparsification_rule,
                    th_val=self.th_val,
                    compute_effective_rank=self.compute_effective_rank,
                )
                hook_module.register_forward_pre_hook(hook)
            else:
                raise ValueError(f"Invalid sparsification mode: {hook_mode}")
            self.sparsity_hooks.append(hook)

    def save_layer_sparsity_data(self):
        total_sparse_neurons = 0
        total_neurons = 0

        layer_sparisty_stats = {}
        for hook in self.sparsity_hooks:
            layer_name = hook.layer_name
            sparse_counts = sum(hook.sparse_counts)
            total_counts = sum(hook.total_counts)
            assert (
                total_counts != 0
            ), f"Total counts for {layer_name} is 0. Check the input mask."
            sparsity = sparse_counts / total_counts
            layer_sparisty_stats[layer_name] = sparsity
            total_sparse_neurons += sparse_counts
            total_neurons += total_counts

        layer_sparisty_stats["total"] = total_sparse_neurons / total_neurons

        sparsity_dict = {
            "rule": self.sparsification_rule,
            "th_val": self.th_val,
            "layers": self.layer_names_to_sparsify,
            "hook_modes": self.layer_hook_modes,
            "sparsity_stats": layer_sparisty_stats,
        }
        if len(self.num_generated_tokens) > 0:
            sparsity_dict["num_generated_tokens_per_sample"] = self.num_generated_tokens

        if self.compute_effective_rank:
            hook_names = [hook.layer_name for hook in self.sparsity_hooks]
            hook_dims = [hook.act_dim for hook in self.sparsity_hooks]
            effective_ranks = [
                np.mean(hook.per_batch_effective_rank) for hook in self.sparsity_hooks
            ]
            mean_effective_rank = np.mean(effective_ranks)
            effective_ranks = dict(zip(hook_names, effective_ranks))
            effective_ranks["mean"] = mean_effective_rank

            sparsity_dict["effective_ranks"] = effective_ranks
            sparsity_dict["hook_dims"] = dict(zip(hook_names, hook_dims))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "sparsity_stats.json").open("w") as f:
            json.dump(sparsity_dict, f, indent=2)

        if self.save_outputs:
            output_path = self.output_dir / "request_logs.json"
            output_dict = {
                "input_ids": self.request_input_ids,
                "output_ids": self.request_output_ids,
                "input_texts": self.request_input_texts,
                "output_texts": self.request_output_texts,
                "batch_lengths": self.batch_lengths,
                "num_generated_tokens": self.num_generated_tokens,
            }

            with output_path.open("w") as f:
                json.dump(output_dict, f, indent=2)
