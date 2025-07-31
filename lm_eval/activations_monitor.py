import json
from abc import ABC
from pathlib import Path

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



class ActivationsHook(ABC):
    def __init__(
        self,
        layer_name: str,
        input_mask_hook: InputTokensHook,
        output_dir: Path,
    ):
        self.layer_name = layer_name
        self.input_mask_hook = input_mask_hook

        # Create a directory for to save the hooks activations
        self.output_file = output_dir / layer_name
        self.output_file.mkdir(parents=True, exist_ok=True)
        self.batch_idx = 0

        self.enabled = True

    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False

    def generic_call(self, value):
        if not self.enabled:
            return value
        
        if isinstance(value, tuple):
            value = value[
                0
            ]  # Handle tuple values in case model operates on multiple ins / outs
        assert isinstance(value, torch.Tensor), "Value must be a torch.Tensor."

        # If all activations are the same, this means the evaluation engine runs the initial batch size calibration
        if torch.all(value[0, 0] == value):
            return value

        input_mask = self.input_mask_hook.token_mask
        batch_size = value.shape[0]
        activations = []
        for i in range(batch_size):
            activations.append(value[i][input_mask[i]].detach().cpu())

        with (self.output_file / f"activations_{self.batch_idx}.pt").open("wb") as f:
            torch.save(activations, f)
        self.batch_idx += 1
        
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
        output_dir: str,
    ):
        raise NotImplementedError("Activations monitor is not implemented yet")

        with Path(sparsity_monitor_config_path).open("r") as f:
            config = json.load(f)

            self.layer_names_to_sparsify = config.get(
                "layers_to_monitor", None
            )
            self.layer_hook_modes = config.get("hook_mode", None)

        self.output_dir = Path(output_dir)
        self.activations_hooks = []

    def enable(self):
        for hook in self.activations_hooks:     
            hook.enable()

    def disable(self):
        for hook in self.activations_hooks:
            hook.disable()

    def initialize(self, lm):
        assert self.layer_names_to_sparsify is not None

        input_hook = InputTokensHook(lm.tokenizer.pad_token_type_id)
        embedding_layer = lm.model.get_submodule("model.embed_tokens")
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
                )
                hook_module.register_forward_hook(hook)
            elif hook_mode == "input":
                hook = InputActivationsHook(
                    layer_name=layer_name,
                    input_mask_hook=input_hook,
                    output_dir=self.output_dir,
                )
                hook_module.register_forward_pre_hook(hook)
            else:
                raise ValueError(f"Invalid sparsification mode: {hook_mode}")
            self.activations_hooks.append(hook)

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
