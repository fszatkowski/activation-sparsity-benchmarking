import json
from abc import ABC
from pathlib import Path
from typing import Counter, Dict, List

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
        if isinstance(input_tensor, tuple):
            input_tensor = input_tensor[0]  # Handle tuple input for some models
        self.token_mask = input_tensor != self.pad_token_id


def compute_tensor_stats(
    tensor: torch.Tensor, input_mask: torch.Tensor, granularity: float
) -> List[List[Dict[int, int]]]:
    # For [B, S, H] tensor, compute the activation stats for each token
    # Encode each tensor into dict which contains the activations stacked into buckets
    # Dict contains the bucket index, where the index of i means that the activation is between i * granularity and (i + 1) * granularity
    batch_size, seq_len, _ = tensor.shape
    bucket_indices = torch.floor(tensor / granularity).long()

    # For each bucket, transform to dict containing indices and number of counts
    # Store output as list of each element in the batch, where each element is a list of dicts
    batch_data = []
    for seq_idx in range(batch_size):
        seq_data = []
        for token_idx in range(seq_len):
            # If the token is masked out, skip
            if not input_mask[seq_idx, token_idx]:
                continue

            # Convert the bucket to python dict of counts
            token_bucket_indices = bucket_indices[seq_idx, token_idx]
            token_bucket_counts = Counter(token_bucket_indices.tolist())
            seq_data.append(token_bucket_counts)
        batch_data.append(seq_data)
    return batch_data


class ActivationsHook(ABC):
    def __init__(
        self,
        layer_name: str,
        input_mask_hook: InputTokensHook,
        output_dir: Path,
        granularity: float = 0.05,
    ):
        self.layer_name = layer_name
        self.input_mask_hook = input_mask_hook
        self.granularity = granularity

        # Create the file where the activation data will be stored
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = output_dir / f"{layer_name.replace('.', '_')}.jsonl"

    def generic_call(self, value, eps=1e-5):
        if isinstance(value, tuple):
            value = value[
                0
            ]  # Handle tuple values in case model operates on multiple ins / outs
        assert isinstance(value, torch.Tensor), "Value must be a torch.Tensor."

        input_mask = self.input_mask_hook.token_mask
        activation_bins = compute_tensor_stats(value, input_mask, self.granularity)
        activation_bins = [{"tokens": seq_tokens} for seq_tokens in activation_bins]

        # If all activations close to the first one, this means the evaluation engine runs the initial batch size calibration
        # Skip saving the data for such batches
        # The computation is still done to allow for more precise automatic batch size estimation
        diff = torch.abs(value[0, 0] - value)
        if torch.all(diff < eps):
            return value

        # For each sequence in the batch, append the dict to the output file
        with self.output_file.open("a+") as f:
            for seq_tokens in activation_bins:
                f.write(json.dumps(seq_tokens) + "\n")

        return value


# Pytorch hook to enforce sparsity on the output of the module
class OutputActivationsHook(ActivationsHook):
    @torch.inference_mode()
    def __call__(self, module, input_tensor, output_tensor):
        value = output_tensor
        return self.generic_call(value)


# Pytorch hook to enforce sparsity on the input to the module
class InputActivationsHook(ActivationsHook):
    @torch.inference_mode()
    def __call__(self, module, input_tensor):
        value = input_tensor
        return self.generic_call(value)


class ActivationsMonitor:
    def __init__(
        self,
        sparsity_monitor_config_path: str,
        output_dir: Path,
        granularity: float = 0.05,
    ):
        output_dir = Path(output_dir)
        with Path(sparsity_monitor_config_path).open("r") as f:
            config = json.load(f)

            self.layer_names_to_sparsify = config.get("layers_to_monitor", None)
            self.layer_hook_modes = config.get("hook_mode", None)
            self.embedding_layer_name = config.get("embedding_layer_name", None)

        self.granularity = granularity
        self.activations_hooks: List[ActivationsHook] = []

        self.output_dir = output_dir / "activation_stats"

    def initialize(self, lm):
        assert self.layer_names_to_sparsify is not None

        input_hook = InputTokensHook(lm.tokenizer.pad_token_id)
        embedding_layer = lm.model.get_submodule(self.embedding_layer_name)
        embedding_layer.register_forward_hook(input_hook)

        modules_to_hook = [
            lm.model.get_submodule(eval_layer_name)
            for eval_layer_name in self.layer_names_to_sparsify
        ]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for layer_name, hook_module, hook_mode in zip(
            self.layer_names_to_sparsify, modules_to_hook, self.layer_hook_modes
        ):
            if hook_mode == "output":
                hook = OutputActivationsHook(
                    layer_name=layer_name,
                    input_mask_hook=input_hook,
                    output_dir=self.output_dir,
                    granularity=self.granularity,
                )
                hook_module.register_forward_hook(hook)
            elif hook_mode == "input":
                hook = InputActivationsHook(
                    layer_name=layer_name,
                    input_mask_hook=input_hook,
                    output_dir=self.output_dir,
                    granularity=self.granularity,
                )
                hook_module.register_forward_pre_hook(hook)
            else:
                raise ValueError(f"Invalid sparsification mode: {hook_mode}")
            self.activations_hooks.append(hook)
