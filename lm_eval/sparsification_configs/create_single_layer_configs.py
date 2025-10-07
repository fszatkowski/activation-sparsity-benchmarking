from pathlib import Path
import json
from typing import Tuple, List


def qwen_module_name_fn(module_type: str, layer_idx: int) -> Tuple[List[str], List[str]]:
    if module_type == "input":
        return [f"model.layers.{layer_idx}.mlp"], ["input"]
    elif module_type == "intermediate":
        return [f"model.layers.{layer_idx}.mlp.down_proj"], ["input"]
    elif module_type == "gate":
        return [f"model.layers.{layer_idx}.mlp.act_fn"], ["output"]
    elif module_type == "up_proj":
        return [f"model.layers.{layer_idx}.mlp.up_proj"], ["output"]
    elif module_type == "all_inputs":
        return [f"model.layers.{layer_idx}.mlp", f"model.layers.{layer_idx}.mlp.down_proj"], ["input", "input"]
    else:
        raise ValueError(f"Invalid module type: {module_type}")


def llama_module_name_fn(module_type: str, layer_idx: int) -> Tuple[List[str], List[str]]:
    if module_type == "input":
        return [f"model.layers.{layer_idx}.mlp"], ["input"]
    elif module_type == "intermediate":
        return [f"model.layers.{layer_idx}.mlp.down_proj"], ["input"]       
    elif module_type == "gate":
        return [f"model.layers.{layer_idx}.mlp.act_fn"], ["output"]
    elif module_type == "up_proj":
        return [f"model.layers.{layer_idx}.mlp.up_proj"], ["output"]
    elif module_type == "all_inputs":
        return [f"model.layers.{layer_idx}.mlp", f"model.layers.{layer_idx}.mlp.down_proj"], ["input", "input"]
    else:
        raise ValueError(f"Invalid module type: {module_type}")


def gemma_lm_module_name_fn(module_type: str, layer_idx: int) -> Tuple[List[str], List[str]]:
    if module_type == "input":
        return [f"model.layers.{layer_idx}.mlp"], ["input"]
    elif module_type == "intermediate":
        return [f"model.layers.{layer_idx}.mlp.down_proj"], ["input"]
    elif module_type == "gate":
        return [f"model.layers.{layer_idx}.mlp.act_fn"], ["output"]
    elif module_type == "up_proj":
        return [f"model.layers.{layer_idx}.mlp.up_proj"], ["output"]
    elif module_type == "all_inputs":
        return [f"model.layers.{layer_idx}.mlp", f"model.layers.{layer_idx}.mlp.down_proj"], ["input", "input"]
    else:
        raise ValueError(f"Invalid module type: {module_type}")


def gemma_multimodal_module_name_fn(module_type: str, layer_idx: int) -> Tuple[List[str], List[str]]:
    if module_type == "input":
        return [f"language_model.layers.{layer_idx}.mlp"], ["input"]
    elif module_type == "intermediate":
        return [f"language_model.layers.{layer_idx}.mlp.down_proj"], ["input"]
    elif module_type == "gate":
        return [f"language_model.layers.{layer_idx}.mlp.act_fn"], ["output"]
    elif module_type == "up_proj":
        return [f"language_model.layers.{layer_idx}.mlp.up_proj"], ["output"]
    elif module_type == "all_inputs":
        return [f"language_model.layers.{layer_idx}.mlp", f"language_model.layers.{layer_idx}.mlp.down_proj"], ["input", "input"]
    else:
        raise ValueError(f"Invalid module type: {module_type}")


def get_model_layer_config(
    model_name: str, module_type: str, layer_idx: int
) -> Tuple[List[str], List[str]]:
    if "qwen" in model_name:
        return qwen_module_name_fn(module_type, layer_idx)
    elif "llama" in model_name:
        return llama_module_name_fn(module_type, layer_idx)
    elif "gemma3-1b" in model_name:
        return gemma_lm_module_name_fn(module_type, layer_idx)
    elif "gemma" in model_name:
        return gemma_multimodal_module_name_fn(module_type, layer_idx)
    else:
        raise ValueError(f"Invalid model name: {model_name}")


if __name__ == "__main__":
    root = Path(__file__).parent

    models_n_layers_embedding_names = [
        # ("llama3-8b", 32, "model.embed_tokens"),
        ("qwen2_5-7b", 28, "model.embed_tokens"),
    ]

    for model_name, n_layers, embedding_name in models_n_layers_embedding_names:
        for module_type in ["input", "intermediate", "gate", "up_proj", "all_inputs"]:
            for layer_idx in range(n_layers):
                module_names, hook_types = get_model_layer_config(
                    model_name, module_type, layer_idx
                )
                config = {
                    "embedding_layer_name": embedding_name,
                    "layers_to_sparsify": module_names,
                    "hook_mode": hook_types,
                }
                config_path = root / 'single_layer' / f"{model_name}_{module_type}_layer_{layer_idx}.json"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with config_path.open("w+") as f:
                    json.dump(config, f, indent=4)
