from pathlib import Path
import json
from typing import Tuple


def qwen_module_name_fn(module_type: str, layer_idx: int) -> Tuple[str, str]:
    if module_type == "input":
        return f"model.layers.{layer_idx}.mlp", "input"
    elif module_type == "intermediate":
        return f"model.layers.{layer_idx}.mlp.down_proj", "input"
    elif module_type == "gate":
        return f"model.layers.{layer_idx}.mlp.act_fn", "output"
    elif module_type == "up_proj":
        return f"model.layers.{layer_idx}.mlp.up_proj", "output"
    else:
        raise ValueError(f"Invalid module type: {module_type}")


def llama_module_name_fn(module_type: str, layer_idx: int) -> str:
    if module_type == "input":
        return f"model.layers.{layer_idx}.mlp", "input"
    elif module_type == "intermediate":
        return f"model.layers.{layer_idx}.mlp.down_proj", "input"
    elif module_type == "gate":
        return f"model.layers.{layer_idx}.mlp.act_fn", "output"
    elif module_type == "up_proj":
        return f"model.layers.{layer_idx}.mlp.up_proj", "output"
    else:
        raise ValueError(f"Invalid module type: {module_type}")


def gemma_lm_module_name_fn(module_type: str, layer_idx: int) -> str:
    if module_type == "input":
        return f"model.layers.{layer_idx}.mlp", "input"
    elif module_type == "intermediate":
        return f"model.layers.{layer_idx}.mlp.down_proj", "input"
    elif module_type == "gate":
        return f"model.layers.{layer_idx}.mlp.act_fn", "output"
    elif module_type == "up_proj":
        return f"model.layers.{layer_idx}.mlp.up_proj", "output"
    else:
        raise ValueError(f"Invalid module type: {module_type}")


def gemma_multimodal_module_name_fn(module_type: str, layer_idx: int) -> str:
    if module_type == "input":
        return f"language_model.layers.{layer_idx}.mlp", "input"
    elif module_type == "intermediate":
        return f"language_model.layers.{layer_idx}.mlp.down_proj", "input"
    elif module_type == "gate":
        return f"language_model.layers.{layer_idx}.mlp.act_fn", "output"
    elif module_type == "up_proj":
        return f"language_model.layers.{layer_idx}.mlp.up_proj", "output"
    else:
        raise ValueError(f"Invalid module type: {module_type}")


def get_model_layer_config(
    model_name: str, module_type: str, layer_idx: int
) -> Tuple[str, str]:
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
        ("gemma3-1b", 26, "model.embed_tokens"),
        ("gemma3-4b", 34, "language_model.embed_tokens"),
        ("gemma3-12b", 48, "language_model.embed_tokens"),
        ("gemma3-27b", 62, "language_model.embed_tokens"),
        ("llama3-1b", 16, "model.embed_tokens"),
        ("llama3-3b", 28, "model.embed_tokens"),
        ("llama3-8b", 32, "model.embed_tokens"),
        ("llama3-70b", 80, "model.embed_tokens"),
        ("qwen2_5-0_5b", 24, "model.embed_tokens"),
        ("qwen2_5-1_5b", 28, "model.embed_tokens"),
        ("qwen2_5-3b", 36, "model.embed_tokens"),
        ("qwen2_5-7b", 28, "model.embed_tokens"),
        ("qwen2_5-14b", 48, "model.embed_tokens"),
        ("qwen2_5-32b", 64, "model.embed_tokens"),
        ("qwen2_5-72b", 80, "model.embed_tokens"),
    ]

    for model_name, n_layers, embedding_name in models_n_layers_embedding_names:
        for module_type in ["input", "intermediate", "gate", "up_proj"]:
            layer_names, hook_modes = [], []
            for layer_idx in range(n_layers):
                module_name, hook_type = get_model_layer_config(
                    model_name, module_type, layer_idx
                )
                layer_names.append(module_name)
                hook_modes.append(hook_type)
            config = {
                "embedding_layer_name": embedding_name,
                "layers_to_sparsify": layer_names,
                "hook_mode": hook_modes,
            }
            config_path = root / f"{model_name}_{module_type}.json"
            with config_path.open("w+") as f:
                json.dump(config, f, indent=4)
