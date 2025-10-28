import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class SampleSparsityStats:
    sparse_counts: int
    total_counts: int
    num_tokens: int


@dataclass
class SequenceSparsityStats:
    prefill: List[SampleSparsityStats] = field(default_factory=list)
    generation: List[SampleSparsityStats] = field(default_factory=list)
    total: List[SampleSparsityStats] = field(default_factory=list)

    def update(self, new_stats: "SequenceSparsityStats"):
        self.prefill.extend(new_stats.prefill)
        self.generation.extend(new_stats.generation)
        self.total.extend(new_stats.total)


@dataclass
class RunningSparsityStats:
    sparse_counts: List[List[int]] = field(default_factory=list)
    total_counts: List[List[int]] = field(default_factory=list)
    prefill_sizes: List[int] = field(default_factory=list)

    def update(
        self,
        sparse_counts: List[List[int]],
        total_counts: List[List[int]],
        prefill_sizes: Optional[List[int]] = None,
    ):
        if self.sparse_counts == []:
            self.sparse_counts = sparse_counts
            self.total_counts = total_counts
        else:
            batch_size = len(sparse_counts)
            for batch_idx in range(batch_size):
                self.sparse_counts[batch_idx].extend(sparse_counts[batch_idx])
                self.total_counts[batch_idx].extend(total_counts[batch_idx])

        if prefill_sizes is not None:
            self.prefill_sizes.extend(prefill_sizes)

    def consolidate(self, batch_start_indices: List[int], batch_end_indices: List[int]):
        batch_size = len(batch_start_indices)

        stats_prefill = []
        stats_generation = []
        stats_total = []

        for batch_idx in range(batch_size):
            sample_sparse_counts = self.sparse_counts[batch_idx]
            sample_total_counts = self.total_counts[batch_idx]

            assert len(sample_sparse_counts) == len(
                sample_total_counts
            ), "Sparse and total counts must have the same length"

            start_index = batch_start_indices[batch_idx]
            end_index = batch_end_indices[batch_idx]
            prefill_idx = self.prefill_sizes[batch_idx]

            prefill_per_token_sparse_counts = sample_sparse_counts[
                start_index:prefill_idx
            ]
            prefill_per_token_total_counts = sample_total_counts[
                start_index:prefill_idx
            ]
            num_tokens_prefill = len(prefill_per_token_sparse_counts)
            sparse_counts_prefill = sum(prefill_per_token_sparse_counts)
            total_counts_prefill = sum(prefill_per_token_total_counts)

            generation_per_token_sparse_counts = sample_sparse_counts[
                prefill_idx:end_index
            ]
            generation_per_token_total_counts = sample_total_counts[
                prefill_idx:end_index
            ]
            num_tokens_generation = len(generation_per_token_sparse_counts)
            sparse_counts_generation = sum(generation_per_token_sparse_counts)
            total_counts_generation = sum(generation_per_token_total_counts)

            stats_prefill.append(
                SampleSparsityStats(
                    sparse_counts=sparse_counts_prefill,
                    total_counts=total_counts_prefill,
                    num_tokens=num_tokens_prefill,
                )
            )
            stats_generation.append(
                SampleSparsityStats(
                    sparse_counts=sparse_counts_generation,
                    total_counts=total_counts_generation,
                    num_tokens=num_tokens_generation,
                )
            )
            stats_total.append(
                SampleSparsityStats(
                    sparse_counts=sparse_counts_prefill + sparse_counts_generation,
                    total_counts=total_counts_prefill + total_counts_generation,
                    num_tokens=num_tokens_prefill + num_tokens_generation,
                )
            )

        return SequenceSparsityStats(
            prefill=stats_prefill, generation=stats_generation, total=stats_total
        )


@dataclass
class RunningEffectiveRankStats:
    activations: List[torch.Tensor] = field(default_factory=list)
    activations_dim: Optional[int] = None

    def update(
        self, activations: Optional[torch.Tensor], activations_dim: Optional[int] = None
    ):
        self.activations.append(activations.detach().cpu())
        if activations_dim is not None and self.activations_dim is None:
            self.activations_dim = activations_dim

    def compute_effective_rank(
        self, batch_start_indices: List[int], batch_end_indices: List[int]
    ) -> float:
        batch_size = len(batch_start_indices)

        concatenated_activations = torch.cat(self.activations, dim=1)
        unmasked_activations = []
        for batch_idx in range(batch_size):
            start_index = batch_start_indices[batch_idx]
            end_index = batch_end_indices[batch_idx]
            batch_activations = concatenated_activations[batch_idx][
                start_index:end_index
            ]
            unmasked_activations.append(batch_activations)

        return self.effective_rank(unmasked_activations)

    @staticmethod
    def effective_rank(activations: List[torch.Tensor]) -> float:
        flat_activations = torch.cat(activations, dim=0)
        eps = 1e-10
        _, S, _ = torch.linalg.svd(flat_activations.float().cuda(), full_matrices=False)
        S = S + eps  # Add eps to avoid log(0)
        p = S / S.sum()
        return torch.exp(-(p * p.log()).sum()).item()


def read_sparsification_config(config_path: str) -> Tuple[str, List[str], List[str]]:
    with Path(config_path).open("r") as f:
        sparsity_config = json.load(f)

        embedding_layer_name = sparsity_config["embedding_layer_name"]
        layers_to_sparsify = sparsity_config["layers_to_sparsify"]
        layers_to_sparsify = {
            key: parse_layers_to_sparsify(layer_indices_string)
            for key, layer_indices_string in layers_to_sparsify.items()
        }
        modules_to_sparsify = sparsity_config["modules_to_sparsify"]

        layer_names_to_sparsify = []
        layer_hook_modes = []

        for module_name_string, hook_mode in modules_to_sparsify.items():
            sparsify_indices = {
                key: indices
                for key, indices in layers_to_sparsify.items()
                if key in module_name_string
            }
            if len(sparsify_indices) > 0:
                # Create product consisting of all the sparsify key values combinations
                values_to_substitute = [
                    dict(zip(sparsify_indices.keys(), values))
                    for values in itertools.product(*sparsify_indices.values())
                ]
                for values in values_to_substitute:
                    new_string = module_name_string
                    for key, value in values.items():
                        new_string = new_string.replace(key, str(value))
                    layer_names_to_sparsify.append(new_string)
                    layer_hook_modes.append(hook_mode)
            else:
                layer_names_to_sparsify.append(module_name_string)
                layer_hook_modes.append(hook_mode)

        layer_names_to_sparsify = layer_names_to_sparsify
        layer_hook_modes = layer_hook_modes

    return embedding_layer_name, layer_names_to_sparsify, layer_hook_modes


def parse_layers_to_sparsify(layer_indices_string: str) -> List[int]:
    layer_indices_string = layer_indices_string.strip().replace(" ", "")
    if "," in layer_indices_string:
        substrings = layer_indices_string.split(",")
    else:
        substrings = [layer_indices_string]

    layer_indices = []
    for substring in substrings:
        if ":" in substring:
            start_index, end_index = substring.split(":")
            layer_indices.extend(list(range(int(start_index), int(end_index) + 1)))
        else:
            layer_indices.append(int(substring))
    return layer_indices


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
        hook_sparsity_stats: SequenceSparsityStats = hook.total_stats

        prefill_sparse_counts = sum(
            [s.sparse_counts for s in hook_sparsity_stats.prefill]
        )
        prefill_total_counts = sum(
            [s.total_counts for s in hook_sparsity_stats.prefill]
        )
        generation_sparse_counts = sum(
            [s.sparse_counts for s in hook_sparsity_stats.generation]
        )
        generation_total_counts = sum(
            [s.total_counts for s in hook_sparsity_stats.generation]
        )
        total_sparse_counts = prefill_sparse_counts + generation_sparse_counts
        total_total_counts = prefill_total_counts + generation_total_counts
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


def parse_hooks_effective_rank_stats(
    hooks: List["SparsificationHook"],
) -> Tuple[Dict[str, float], Dict[str, int]]:
    effective_ranks = {}
    hook_dims = {}
    for hook in hooks:
        effective_ranks[hook.layer_name] = np.mean(
            hook.effective_ranks_per_sample_batches
        ).item()
        hook_dims[hook.layer_name] = hook.hook_dim

    mean_effective_rank = np.sum(
        [r * d for r, d in zip(effective_ranks.values(), hook_dims.values())]
    ) / np.sum(list(hook_dims.values()))
    effective_ranks["mean"] = mean_effective_rank
    return effective_ranks, hook_dims
