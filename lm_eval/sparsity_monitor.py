import json
from pathlib import Path
from typing import List, Optional
from lm_eval.plotting_utils import plot_heatmap
import torch

def find_sparse_subset(input_tensor: torch.Tensor, input_mask: torch.Tensor, topp_vals:List[float]):
    activations = input_tensor[0]

    # If all activations are the same, this means the evaluation engine runs the initial batch size calibration
    if torch.all(activations[0,0] == activations):
        mock = True
    else:
        mock = False
    
    non_padded_activations = input_mask.unsqueeze(-1).float() * activations

    sorted_activations, _ = non_padded_activations.abs().sort(descending=True, dim=-1)

    # Use more precise dtype for the cumsum
    activation_cumsum = sorted_activations.cumsum(dim=-1, dtype=torch.float32)
    activation_sums = activation_cumsum[:,:,-1].unsqueeze(-1)

    # TODO Optimize - vectorize to get rid of the for loop
    sparse_neurons_per_th = [((activation_cumsum >= activation_sums * topp_val).sum(-1) * input_mask).sum(-1).tolist() for topp_val in topp_vals]
    total_num_neurons = (input_mask.sum(-1) * activations.shape[-1]).tolist()
    if mock:
        return [], []
    else:
        return sparse_neurons_per_th, total_num_neurons

# Pytorch hook to monitor the input to the Linear layer
class InputSparsityMonitorHook:
    def __init__(self, layer_name, topp, input_mask_hook):
        self.layer_name = layer_name
        self.input_mask_hook = input_mask_hook
        self.topp = sorted(topp)
        self.sparse_counts = {topp: [] for topp in self.topp}
        self.total_counts = []

    @torch.inference_mode()
    def __call__(self, module, input_tensor, output_tensor):
        input_mask = self.input_mask_hook.token_mask
        num_sparse_neurons, num_all_neurons = find_sparse_subset(input_tensor, input_mask, self.topp )
        for idx, topp in enumerate(self.topp):
            self.sparse_counts[topp].extend(num_sparse_neurons[idx])
        self.total_counts.extend(num_all_neurons)


def sparsify_tensor(tensor: torch.Tensor, input_mask: torch.Tensor, topp_val: float):
    # If all activations are the same, this means the evaluation engine runs the initial batch size calibration
    if torch.all(tensor[0,0] == tensor):
        mock = True
    else:
        mock = False
    
    non_padded_activations = input_mask.unsqueeze(-1).float() * tensor

    sorted_activations, _ = non_padded_activations.abs().sort(descending=True, dim=-1)

    # Use more precise dtype for the cumsum
    activation_cumsum = sorted_activations.cumsum(dim=-1, dtype=torch.float32) # shape: [batch_size, seq_len, hidden_dim]
    activation_sums = activation_cumsum[:,:,-1].unsqueeze(-1) # shape: [batch_size, seq_len]

    # Find which activations to use to maintain at least topp percent of the total activation sum
    cumsum_threshold_idx = (activation_cumsum < activation_sums * topp_val).sum(-1) # shape: [batch_size, seq_len]
    # Get the minimum activation threshold    
    activation_threshold_value = sorted_activations.gather(-1, cumsum_threshold_idx.unsqueeze(-1)).type(tensor.dtype)
    non_sparse_mask = non_padded_activations.abs() >= activation_threshold_value
    sparsified_tensor = torch.where(non_sparse_mask, tensor, torch.zeros_like(tensor))

    # TODO Optimize - vectorize to get rid of the for loop
    sparse_neurons = ((activation_cumsum >= activation_sums * topp_val).sum(-1) * input_mask).sum(-1).tolist() 
    total_num_neurons = (input_mask.sum(-1) * tensor.shape[-1]).tolist()
    
    if mock:
        return tensor, [], []
    else:
        return sparsified_tensor, sparse_neurons, total_num_neurons


# Pytorch hook to enforce sparsity on the output of the module
class OutputSparsificationHook:
    def __init__(self, layer_name, topp, input_mask_hook):
        self.layer_name = layer_name
        self.input_mask_hook = input_mask_hook
        self.topp = topp
        self.sparse_counts = []
        self.total_counts = []

    @torch.inference_mode()
    def __call__(self, module, input_tensor, output_tensor):
        input_mask = self.input_mask_hook.token_mask
        sparsified_output, num_sparse_neurons, num_all_neurons = sparsify_tensor(output_tensor[0], input_mask, self.topp )
        self.sparse_counts.extend(num_sparse_neurons)
        self.total_counts.extend(num_all_neurons)

        return sparsified_output

# Pytorch hook to enforce sparsity on the input to the module
class InputSparsificationHook:
    def __init__(self, layer_name, topp, input_mask_hook):
        self.layer_name = layer_name
        self.input_mask_hook = input_mask_hook
        self.topp = topp
        self.sparse_counts = []
        self.total_counts = []

    @torch.inference_mode()
    def __call__(self, module, input_tensor):
        input_mask = self.input_mask_hook.token_mask
        sparsified_input, num_sparse_neurons, num_all_neurons = sparsify_tensor(input_tensor[0], input_mask, self.topp )
        self.sparse_counts.extend(num_sparse_neurons)
        self.total_counts.extend(num_all_neurons)

        return sparsified_input


class InputTokensHook:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        self.token_mask = None

    @torch.inference_mode()
    def __call__(self, module, input_tensor, output_tensor):
        self.token_mask = (input_tensor[0] != self.pad_token_id)


class SparsityMonitor:
    def __init__(self, sparsity_config_path: str, output_dir: str, topp: Optional[float] = None, sparsification_mode: Optional[str] = None):
        with Path(sparsity_config_path).open("r") as f:
            sparsity_config = json.load(f)

            self.layers_to_evaluate = sparsity_config.get("layers_to_monitor", None)
            self.layers_to_sparsify = sparsity_config.get("layers_to_sparsify", None)
            self.topp = sparsity_config.get("topp", None)
            self.sparsification_mode = sparsity_config.get("sparsification_mode", None)

            # Overwrite some values from configs
            if topp is not None:
                self.topp = topp
            if sparsification_mode is not None:
                self.sparsification_mode = sparsification_mode

            self.output_dir = Path(output_dir)

        self.monitor_hooks = []
        self.sparsity_hooks = []

    def initialize(self, lm):
        input_hook = InputTokensHook(lm.tokenizer.pad_token_type_id)
        embedding_layer = lm.model.get_submodule('model.embed_tokens')
        embedding_layer.register_forward_hook(input_hook)

        if self.layers_to_evaluate is not None:
            hook_layers = [lm.model.get_submodule(eval_layer_name) for eval_layer_name in self.layers_to_evaluate]
            for layer_name, hook_layer in zip(self.layers_to_evaluate, hook_layers):
                hook = InputSparsityMonitorHook(layer_name, self.topp, input_hook)
                hook_layer.register_forward_hook(hook)
                self.monitor_hooks.append(hook)
        
        if self.layers_to_sparsify is not None:
            hook_layers = [lm.model.get_submodule(eval_layer_name) for eval_layer_name in self.layers_to_sparsify]
            for layer_name, hook_layer in zip(self.layers_to_sparsify, hook_layers):
                if self.sparsification_mode == "output":
                    hook = OutputSparsificationHook(layer_name, self.topp, input_hook)
                    hook_layer.register_forward_hook(hook)
                elif self.sparsification_mode == "input":
                    hook = InputSparsificationHook(layer_name, self.topp, input_hook)
                    hook_layer.register_forward_pre_hook(hook)
                else:
                    raise ValueError(f"Invalid sparsification mode: {self.sparsification_mode}")
                self.sparsity_hooks.append(hook)


    def plot_input_sparsity_data(self):
        if self.layers_to_evaluate is None:
            return
        
        layer_names = self.layers_to_evaluate
        topp_vals = self.topp

        # TODO differentiate between tasks?
        sparsity_values = torch.zeros((len(layer_names), len(topp_vals)))
        for layer_idx, hook in enumerate(self.monitor_hooks):
            for topp_idx, topp in enumerate(topp_vals):
                sparsity_values[layer_idx, topp_idx] = sum(hook.sparse_counts[topp]) / sum(hook.total_counts)

        # Make it percent
        sparsity_values = sparsity_values * 100
        layer_names = ['mlp.' + str(i) for i in range(len(layer_names))]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "sparsity_heatmap.png"
        plot_heatmap(layer_names, topp_vals, sparsity_values.T, output_path)
        

    def save_layer_sparsity_data(self):
        if self.layers_to_sparsify is None:
            return

        total_sparse_neurons = 0
        total_neurons = 0
        layer_sparisty_stats = {}
        for hook in self.sparsity_hooks:
            layer_name = hook.layer_name
            sparse_counts = sum(hook.sparse_counts)  
            total_counts = sum(hook.total_counts)
            sparsity = sparse_counts / total_counts
            layer_sparisty_stats[layer_name] = sparsity
            total_sparse_neurons += sparse_counts
            total_neurons += total_counts

        layer_sparisty_stats["total"] = total_sparse_neurons / total_neurons

        sparsity_dict = {'topp': self.topp, 'sparsity_stats': layer_sparisty_stats}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "sparsity_stats.json").open("w") as f:
            json.dump(sparsity_dict, f, indent=2)

    def register_task(self, task_name, model):
        self.task = task_name
        self.model = model
        self.data_buffers = self._prepare_data_buffers()