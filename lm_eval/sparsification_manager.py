import json
from abc import ABC
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from lm_eval.sparsification_utils import (
    RunningEffectiveRankStats,
    RunningSparsityStats,
    SequenceSparsityStats,
    parse_hooks_effective_rank_stats,
    parse_hooks_sparsity_stats,
    read_sparsification_config,
)


def get_topp_mask(
    tensor: torch.Tensor, topp_val: float, topp_power: float = 1.0
) -> torch.Tensor:
    abs_tensor = tensor.abs()
    if topp_power != 1.0:
        abs_tensor = abs_tensor**topp_power

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
    topp_power: float = 1.0,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    # Check for the device of the input mask and move it to the device of the tensor
    # Outputs the sparsified tensor, the number of sparse neurons in each token, and the total number of neurons in each token
    if rule == "topp":
        assert th_val is not None, "Topp value must be provided for topp rule"
        non_sparse_mask = get_topp_mask(tensor, th_val, topp_power)
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
    total_num_neurons = [[feature_dim] * seq_len for _ in range(batch_size)]

    assert (
        len(sparse_neurons_count) == batch_size
    ), "Sparse neurons count must have the same length as the batch size"
    assert (
        len(total_num_neurons) == batch_size
    ), "Total neurons count must have the same length as the batch size"
    for i in range(batch_size):
        assert (
            len(sparse_neurons_count[i]) == seq_len
        ), "Sparse neurons count must have the same length as the sequence length"
        assert (
            len(total_num_neurons[i]) == seq_len
        ), "Total neurons count must have the same length as the sequence length"

    return sparsified_tensor, sparse_neurons_count, total_num_neurons


class SparsificationHook(ABC):
    def __init__(
        self,
        layer_name: str,
        rule: str,
        th_val: Optional[float] = None,
        topp_power: float = 1.0,
        compute_effective_rank: bool = False,
    ):
        self.layer_name = layer_name
        self.rule = rule
        self.th_val = th_val
        self.topp_power = topp_power

        # Per-token activation sparsity values for the currently processed sequence, without masking tokens
        self.running_stats = RunningSparsityStats()
        # Total stats across all sequences seen so far, taking the masking into account
        self.total_stats = SequenceSparsityStats()

        # Whether to compute the effective rank of the activations
        self.compute_effective_rank = compute_effective_rank
        # Per-batch stats for effective rank computations, used to compute the effective rank of the activations
        self.running_effective_rank_stats = RunningEffectiveRankStats()
        # Effective ranks of the activations stored per batch of samples processed so far
        # This is consolidated for the whole sequence, that is: either prefill or prefill+generation
        self.effective_ranks_per_sample_batches = []
        self.hook_dim = None

    def restart(self):
        self.running_stats = RunningSparsityStats()
        self.running_effective_rank_stats = RunningEffectiveRankStats()

    @torch.inference_mode()
    def generic_call(self, value):
        if isinstance(value, tuple):
            value = value[
                0
            ]  # Handle tuple values in case model operates on multiple ins / outs
        assert isinstance(value, torch.Tensor), "Value must be a torch.Tensor."
        batch_size, seq_len, feature_dim = value.shape

        sparsified_value, num_sparse_neurons, num_all_neurons = sparsify_tensor(
            value, self.rule, self.th_val, self.topp_power
        )

        assert sparsified_value.shape == (
            batch_size,
            seq_len,
            feature_dim,
        ), "Sparsified value has different shape than the original."

        # Add prefill size if it's not yet present in the running stats
        if self.running_stats.prefill_sizes == []:
            prefill_sizes = [seq_len] * batch_size
        else:
            prefill_sizes = None

        # Update the running sparsity stats
        self.running_stats.update(
            sparse_counts=num_sparse_neurons,
            total_counts=num_all_neurons,
            prefill_sizes=prefill_sizes,
        )

        # Store the activations for effective rank computation if required
        if self.compute_effective_rank:
            self.running_effective_rank_stats.update(
                activations=value, activations_dim=feature_dim
            )
            if self.hook_dim is None:
                self.hook_dim = feature_dim

        return sparsified_value

    def consolidate_batch(self, batch_start_indices, batch_end_indices):
        # Update sequence level stats and restert the running stats
        current_sequence_stats = self.running_stats.consolidate(
            batch_start_indices, batch_end_indices
        )
        self.total_stats.update(current_sequence_stats)
        self.running_stats = RunningSparsityStats()

        if self.compute_effective_rank:
            # Concatenate the activations along the sequence dimension
            effective_rank = self.running_effective_rank_stats.compute_effective_rank(
                batch_start_indices, batch_end_indices
            )
            self.effective_ranks_per_sample_batches.append(effective_rank)
            self.running_effective_rank_stats = RunningEffectiveRankStats()


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
        topp_power: float = 1.0,
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

        assert topp_power >= 1.0, "topp_power for sparsification must be >= 1.0"
        self.topp_power = topp_power

        self.output_dir = Path(output_dir)
        (
            self.embedding_layer_name,
            self.layer_names_to_sparsify,
            self.layer_hook_modes,
        ) = read_sparsification_config(sparsification_config_path)
        self.sparsity_hooks: List[SparsificationHook] = []

        self.save_outputs = save_outputs
        self.requests_data = []
        self.compute_effective_rank = compute_effective_rank

        self.num_generated_tokens = []
        self.request_input_ids = []
        self.request_output_ids = []
        self.request_input_texts = []
        self.request_output_texts = []
        self.batch_lengths = []

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
                    topp_power=self.topp_power,
                    compute_effective_rank=self.compute_effective_rank,
                )
                hook_module.register_forward_hook(hook)
            elif hook_mode == "input":
                hook = InputSparsificationHook(
                    layer_name=layer_name,
                    rule=self.sparsification_rule,
                    th_val=self.th_val,
                    topp_power=self.topp_power,
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
        if self.save_outputs:
            self.request_input_ids.extend(context_ids)
            self.request_output_ids.extend(response_ids)
            self.request_input_texts.extend(decoded_contexts)
            self.request_output_texts.extend(decoded_responses)
            self.num_generated_tokens.extend(num_generated_tokens)
            self.batch_lengths.extend(batch_lengths)

    def save_layer_sparsity_data(self):
        prefill_stats, generation_stats, total_stats = parse_hooks_sparsity_stats(
            self.sparsity_hooks
        )
        sparsity_dict = {
            "rule": self.sparsification_rule,
            "th_val": self.th_val,
            "topp_power": self.topp_power,
            "layers": self.layer_names_to_sparsify,
            "hook_modes": self.layer_hook_modes,
            "sparsity_stats": total_stats,
            "sparsity_stats_prefill": prefill_stats,
            "sparsity_stats_generation": generation_stats,
        }

        if len(self.num_generated_tokens) > 0:
            sparsity_dict["num_generated_tokens_per_sample"] = self.num_generated_tokens

        if self.compute_effective_rank:
            effective_ranks, hook_dims = parse_hooks_effective_rank_stats(
                self.sparsity_hooks
            )
            sparsity_dict["effective_ranks"] = effective_ranks
            sparsity_dict["hook_dims"] = hook_dims

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
