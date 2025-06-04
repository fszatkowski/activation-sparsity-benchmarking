import json
from abc import ABC
from pathlib import Path
from typing import List, Optional, Tuple

import torch


class InputTokensHook:
    """
    Hook to register which tokens are padding tokens in the input tensor.
    """

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        self.token_mask = None

    @torch.inference_mode()
    def __call__(self, module, input_tensor, output_tensor):
        self.token_mask = input_tensor[0] != self.pad_token_id


def get_topp_mask(
    tensor: torch.Tensor, topp_val: float, input_mask: torch.Tensor
) -> torch.Tensor:
    non_padded_activations = (input_mask.unsqueeze(-1).float() * tensor).abs()

    sorted_activations, _ = non_padded_activations.sort(descending=True, dim=-1)

    # Use more precise dtype for the cumsum
    activation_cumsum = sorted_activations.cumsum(
        dim=-1, dtype=torch.float64
    )  # shape: [batch_size, seq_len, hidden_dim]
    activation_sums = activation_cumsum[:, :, -1].unsqueeze(
        -1
    )  # shape: [batch_size, seq_len]

    # Find which activations to use to maintain at least topp percent of the total activation sum
    cumsum_threshold_idx = (activation_cumsum < activation_sums * topp_val).sum(
        -1
    )  # shape: [batch_size, seq_len]
    # Get the minimum activation threshold
    activation_threshold_value = sorted_activations.gather(
        -1, cumsum_threshold_idx.unsqueeze(-1)
    ).type(tensor.dtype)
    non_sparse_mask = non_padded_activations >= activation_threshold_value
    return non_sparse_mask


def get_maxp_mask(
    tensor: torch.Tensor, maxp_val: float, input_mask: torch.Tensor
) -> torch.Tensor:
    non_padded_activations = (input_mask.unsqueeze(-1).float() * tensor).abs()
    max_activation_vals = non_padded_activations.max(dim=-1).values
    threshold_vals = max_activation_vals * maxp_val
    non_sparse_mask = non_padded_activations >= threshold_vals.unsqueeze(-1)
    return non_sparse_mask


def get_topk_mask(
    tensor: torch.Tensor, topk_val: float, input_mask: torch.Tensor
) -> torch.Tensor:
    feature_dim = tensor.shape[-1]
    topk_int_val = int(feature_dim * topk_val)
    non_padded_activations = (input_mask.unsqueeze(-1).float() * tensor).abs()

    # Find minimum top-k activations for each token
    topk_threshold_vals = (
        torch.topk(non_padded_activations, topk_int_val, dim=-1, sorted=True)
        .values.min(dim=-1)
        .values
    )
    # Create a mask for the top-k activations
    non_sparse_mask = non_padded_activations >= topk_threshold_vals.unsqueeze(-1)
    # Ensure the mask is applied only to non-padded activations
    non_sparse_mask = non_sparse_mask * input_mask.unsqueeze(-1).float()
    # Convert to boolean mask
    non_sparse_mask = non_sparse_mask.bool()
    return non_sparse_mask


def sparsify_tensor(
    tensor: torch.Tensor,
    input_mask: torch.Tensor,
    rule: str,
    th_val: Optional[float] = None,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    # If all activations are the same, this means the evaluation engine runs the initial batch size calibration

    if torch.all(tensor[0, 0] == tensor):
        return tensor, [], []

    if rule == "topp":
        assert th_val is not None, "Topp value must be provided for topp rule"
        non_sparse_mask = get_topp_mask(tensor, th_val, input_mask)
    elif rule == "maxp":
        assert th_val is not None, "Maxp value must be provided for maxp rule"
        non_sparse_mask = get_maxp_mask(tensor, th_val, input_mask)
    elif rule == "topk":
        assert th_val is not None, "Topk value must be provided for topk rule"
        non_sparse_mask = get_topk_mask(tensor, th_val, input_mask)
    else:
        raise ValueError(f"Invalid sparsification rule: {rule}")

    sparsified_tensor = torch.where(non_sparse_mask, tensor, torch.zeros_like(tensor))

    sparse_neurons = ((~non_sparse_mask).sum(-1) * input_mask).sum(-1).tolist()
    total_num_neurons = (input_mask.sum(-1) * tensor.shape[-1]).tolist()

    return sparsified_tensor, sparse_neurons, total_num_neurons


class SparsificationHook(ABC):
    def __init__(
        self,
        layer_name: str,
        rule: str,
        input_mask_hook: InputTokensHook,
        th_val: Optional[float] = None,
    ):
        self.layer_name = layer_name
        self.input_mask_hook = input_mask_hook
        self.rule = rule
        self.th_val = th_val

        self.sparse_counts = []
        self.total_counts = []

    def generic_call(self, value):
        input_mask = self.input_mask_hook.token_mask
        sparsified_value, num_sparse_neurons, num_all_neurons = sparsify_tensor(
            value, input_mask, rule=self.rule, th_val=self.th_val
        )
        self.sparse_counts.extend(num_sparse_neurons)
        self.total_counts.extend(num_all_neurons)
        return sparsified_value


# Pytorch hook to enforce sparsity on the output of the module
class OutputSparsificationHook(SparsificationHook):
    @torch.inference_mode()
    def __call__(self, module, input_tensor, output_tensor):
        value = output_tensor[0]
        return self.generic_call(value)


# Pytorch hook to enforce sparsity on the input to the module
class InputSparsificationHook(SparsificationHook):
    @torch.inference_mode()
    def __call__(self, module, input_tensor):
        value = input_tensor[0]
        return self.generic_call(value)


class SparsityMonitor:
    def __init__(
        self,
        sparsification_config_path: str,
        output_dir: str,
        sparsification_rule: str,
        th_val: Optional[float] = None,
    ):
        self.sparsification_rule = sparsification_rule
        assert (
            th_val is not None
        ), "Threshold value must be provided for sparsified infreence."
        assert th_val > 0, "Threshold value must be greater than 0."
        assert th_val <= 1, "Threshold value must be less than or equal to 1."
        self.th_val = th_val

        with Path(sparsification_config_path).open("r") as f:
            sparsity_config = json.load(f)

            self.layer_names_to_sparsify = sparsity_config.get(
                "layers_to_sparsify", None
            )
            self.layer_hook_modes = sparsity_config.get("hook_mode", None)

        self.output_dir = Path(output_dir)
        self.monitor_hooks = []
        self.sparsity_hooks = []

    def initialize(self, lm):
        assert self.layer_names_to_sparsify is not None

        input_hook = InputTokensHook(lm.tokenizer.pad_token_type_id)
        embedding_layer = lm.model.get_submodule("model.embed_tokens")
        embedding_layer.register_forward_hook(input_hook)

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
                    input_mask_hook=input_hook,
                    th_val=self.th_val,
                )
                hook_module.register_forward_hook(hook)
            elif hook_mode == "input":
                hook = InputSparsificationHook(
                    layer_name=layer_name,
                    rule=self.sparsification_rule,
                    input_mask_hook=input_hook,
                    th_val=self.th_val,
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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "sparsity_stats.json").open("w") as f:
            json.dump(sparsity_dict, f, indent=2)
