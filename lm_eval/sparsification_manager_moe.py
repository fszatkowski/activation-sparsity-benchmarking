import json
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from lm_eval.sparsification_utils import (
    read_sparsification_config,
)


def get_topp_mask(tensor: torch.Tensor, topp_val: float) -> torch.Tensor:
    abs_tensor = tensor.abs()

    # Sort the activation values, the biggest activations appear first
    sorted_activations, _ = abs_tensor.sort(descending=True, dim=-1)

    # Calculate the cumsum - the total sum of activations up to this position
    # dtype is float64 to avoid overflow
    activation_cumsum = sorted_activations.cumsum(
        dim=-1, dtype=torch.float64
    )  # shape: [num_tokens, hidden_dim]
    # Sum of all activations is the last element of the cumsum
    activation_sums = activation_cumsum[:, -1].unsqueeze(-1)  # shape: [num_tokens, 1]

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
    # Outputs the sparsified tensor, the number of sparse neurons in each token, and the total number of neurons in each token
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
    num_tokens, feature_dim = tensor.shape
    total_num_neurons = [feature_dim] * num_tokens

    assert (
        len(sparse_neurons_count) == num_tokens
    ), "Sparse neurons count must have the same length as the number of tokens"
    assert (
        len(total_num_neurons) == num_tokens
    ), "Total neurons count must have the same length as the number of tokens"

    return sparsified_tensor, sparse_neurons_count, total_num_neurons


class MoESparsificationHook(ABC):
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

        # Per-token activation sparsity values for the currently processed sequence, without masking tokens
        self.running_sparse_count = 0
        self.running_total_count = 0
        self.running_prefill_sparse_count = None
        self.running_prefill_total_count = None

        # Total stats across all sequences seen so far, taking the masking into account
        self.total_sparse_count = 0
        self.total_total_count = 0
        self.total_prefill_sparse_count = 0
        self.total_prefill_total_count = 0

        if compute_effective_rank:
            raise NotImplementedError(
                "Effective rank computation is not supported for MoE models."
            )

    def restart(self):
        self.running_sparse_count = 0
        self.running_total_count = 0
        self.running_prefill_sparse_count = None
        self.running_prefill_total_count = None

        self.total_sparse_count = 0
        self.total_total_count = 0
        self.total_prefill_sparse_count = 0
        self.total_prefill_total_count = 0

    @torch.inference_mode()
    def generic_call(self, value):
        if isinstance(value, tuple):
            value = value[
                0
            ]  # Handle tuple values in case model operates on multiple ins / outs
        assert isinstance(value, torch.Tensor), "Value must be a torch.Tensor."
        num_tokens, feature_dim = value.shape

        sparsified_value, num_sparse_neurons, num_all_neurons = sparsify_tensor(
            value, self.rule, self.th_val
        )

        assert sparsified_value.shape == (
            num_tokens,
            feature_dim,
        ), "Sparsified value has different shape than the original."

        # Add prefill size if it's not yet present in the running stats
        if self.running_prefill_sparse_count is None:
            # We run with batch size 1, so the prefill size is the number of tokens for the first pass in the current sequence
            self.running_prefill_sparse_count = num_tokens
            self.running_prefill_total_count = num_all_neurons

        # Update the running sparsity stats
        self.running_sparse_count += num_sparse_neurons
        self.running_total_count += num_all_neurons

        return sparsified_value

    def consolidate_batch(self, batch_start_indices, batch_end_indices):
        # Update sequence level stats and restert the running stats
        self.total_sparse_count += self.running_sparse_count
        self.total_total_count += self.running_total_count
        self.total_prefill_sparse_count += self.running_prefill_sparse_count
        self.total_prefill_total_count += self.running_prefill_total_count

        self.running_sparse_count = 0
        self.running_total_count = 0
        self.running_prefill_sparse_count = None
        self.running_prefill_total_count = None


# Pytorch hook to enforce sparsity on the output of the module
class MoEOutputSparsificationHook(MoESparsificationHook):
    @torch.inference_mode()
    def __call__(self, module, input_tensor, output_tensor):
        value = output_tensor
        return self.generic_call(value)


# Pytorch hook to enforce sparsity on the input to the module
class MoEInputSparsificationHook(MoESparsificationHook):
    @torch.inference_mode()
    def __call__(self, module, input_tensor):
        value = input_tensor
        return self.generic_call(value)


class MoESparsificationManager:
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

        self.output_dir = Path(output_dir)
        (
            self.embedding_layer_name,
            self.layer_names_to_sparsify,
            self.layer_hook_modes,
        ) = read_sparsification_config(sparsification_config_path)
        self.sparsity_hooks: List[MoESparsificationHook] = []

        self.save_outputs = save_outputs
        self.compute_effective_rank = compute_effective_rank

        if save_outputs:
            raise NotImplementedError(
                "Saving inputs and outputs is not supported for MoE models."
            )

        if compute_effective_rank:
            raise NotImplementedError(
                "Effective rank computation is not supported for MoE models."
            )

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
                hook = MoEOutputSparsificationHook(
                    layer_name=layer_name,
                    rule=self.sparsification_rule,
                    th_val=self.th_val,
                    compute_effective_rank=self.compute_effective_rank,
                )
                hook_module.register_forward_hook(hook)
            elif hook_mode == "input":
                hook = MoEInputSparsificationHook(
                    layer_name=layer_name,
                    rule=self.sparsification_rule,
                    th_val=self.th_val,
                    compute_effective_rank=self.compute_effective_rank,
                )
                hook_module.register_forward_pre_hook(hook)
            else:
                raise ValueError(f"Invalid sparsification mode: {hook_mode}")
            self.sparsity_hooks.append(hook)

    def restart(self):
        for hook in self.sparsity_hooks:
            hook.restart()

        if self.save_outputs:
            self.requests_data = []

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
        pass

    def save_layer_sparsity_data(self):
        prefill_stats, generation_stats, total_stats = parse_hooks_sparsity_stats(
            self.sparsity_hooks
        )
        sparsity_dict = {
            "rule": self.sparsification_rule,
            "th_val": self.th_val,
            "layers": self.layer_names_to_sparsify,
            "hook_modes": self.layer_hook_modes,
            "sparsity_stats": total_stats,
            "sparsity_stats_prefill": prefill_stats,
            "sparsity_stats_generation": generation_stats,
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "sparsity_stats.json").open("w") as f:
            json.dump(sparsity_dict, f, indent=2)

        if self.save_outputs:
            raise NotImplementedError(
                "Saving inputs and outputs is not supported for MoE models."
            )


def parse_hooks_sparsity_stats(hooks: List["SparsificationHook"]) -> Dict[str, float]:
    layer_sparisty_stats_prefill = {}
    layer_sparisty_stats_generation = {}
    layer_sparisty_stats_total = {}

    sparse_neurons_prefill = 0
    all_neurons_prefill = 0

    sparse_neurons_generation = 0
    total_neurons_generation = 0

    sparse_neurons_total = 0
    neurons_total = 0

    for hook in hooks:
        layer_name = hook.layer_name

        total_sparse_counts = hook.total_sparse_count
        total_total_counts = hook.total_total_count
        prefill_sparse_counts = hook.total_prefill_sparse_count
        prefill_total_counts = hook.total_prefill_total_count
        generation_sparse_counts = total_sparse_counts - prefill_sparse_counts
        generation_total_counts = total_total_counts - prefill_total_counts

        assert (
            total_total_counts != 0
        ), f"Total counts for {layer_name} is 0. Check the input mask."

        prefill_sparsity = (
            prefill_sparse_counts / prefill_total_counts
            if prefill_total_counts != 0
            else 0
        )
        generation_sparsity = (
            generation_sparse_counts / generation_total_counts
            if generation_total_counts != 0
            else 0
        )
        total_sparsity = (
            total_sparse_counts / total_total_counts if total_total_counts != 0 else 0
        )

        layer_sparisty_stats_prefill[layer_name] = prefill_sparsity
        layer_sparisty_stats_generation[layer_name] = generation_sparsity
        layer_sparisty_stats_total[layer_name] = total_sparsity

        sparse_neurons_prefill += prefill_sparse_counts
        all_neurons_prefill += prefill_total_counts
        sparse_neurons_generation += generation_sparse_counts
        total_neurons_generation += generation_total_counts
        sparse_neurons_total += total_sparse_counts
        neurons_total += total_total_counts

    layer_sparisty_stats_prefill["average"] = (
        sparse_neurons_prefill / all_neurons_prefill
    )
    layer_sparisty_stats_generation["average"] = (
        sparse_neurons_generation / total_neurons_generation
        if total_neurons_generation != 0
        else 0
    )
    layer_sparisty_stats_total["average"] = sparse_neurons_total / neurons_total

    return (
        layer_sparisty_stats_prefill,
        layer_sparisty_stats_generation,
        layer_sparisty_stats_total,
    )
