import json
from pathlib import Path
from typing import Dict


def qwen_module_name_fn(module_type: str) -> Dict[str, str]:
    if module_type == "input":
        return {f"model.layers.L.mlp": "input"}
    elif module_type == "intermediate":
        return {"model.layers.L.mlp.down_proj": "input"}
    elif module_type == "gate":
        return {"model.layers.L.mlp.act_fn": "output"}
    elif module_type == "up_proj":
        return {"model.layers.L.mlp.up_proj": "output"}
    elif module_type == "all_inputs":
        return {"model.layers.L.mlp": "input", "model.layers.L.mlp.down_proj": "input"}
    else:
        raise ValueError(f"Invalid module type: {module_type}")


def llama_module_name_fn(module_type: str) -> Dict[str, str]:
    if module_type == "input":
        return {f"model.layers.L.mlp": "input"}
    elif module_type == "intermediate":
        return {f"model.layers.L.mlp.down_proj": "input"}
    elif module_type == "gate":
        return {f"model.layers.L.mlp.act_fn": "output"}
    elif module_type == "up_proj":
        return {f"model.layers.L.mlp.up_proj": "output"}
    elif module_type == "all_inputs":
        return {
            f"model.layers.L.mlp": "input",
            f"model.layers.L.mlp.down_proj": "input",
        }
    else:
        raise ValueError(f"Invalid module type: {module_type}")


def gemma_lm_module_name_fn(module_type: str) -> Dict[str, str]:
    if module_type == "input":
        return {f"model.layers.L.mlp": "input"}
    elif module_type == "intermediate":
        return {f"model.layers.L.mlp.down_proj": "input"}
    elif module_type == "gate":
        return {f"model.layers.L.mlp.act_fn": "output"}
    elif module_type == "up_proj":
        return {f"model.layers.L.mlp.up_proj": "output"}
    elif module_type == "all_inputs":
        return {
            f"model.layers.L.mlp": "input",
            f"model.layers.L.mlp.down_proj": "input",
        }
    else:
        raise ValueError(f"Invalid module type: {module_type}")


def gemma_multimodal_module_name_fn(module_type: str) -> Dict[str, str]:
    if module_type == "input":
        return {f"language_model.layers.L.mlp": "input"}
    elif module_type == "intermediate":
        return {f"language_model.layers.L.mlp.down_proj": "input"}
    elif module_type == "gate":
        return {f"language_model.layers.L.mlp.act_fn": "output"}
    elif module_type == "up_proj":
        return {f"language_model.layers.L.mlp.up_proj": "output"}
    elif module_type == "all_inputs":
        return {
            f"language_model.layers.L.mlp": "input",
            f"language_model.layers.L.mlp.down_proj": "input",
        }
    else:
        raise ValueError(f"Invalid module type: {module_type}")


def get_model_layer_config(model_name: str, module_type: str) -> Dict[str, str]:
    if "qwen" in model_name:
        return qwen_module_name_fn(module_type)
    elif "llama" in model_name:
        return llama_module_name_fn(module_type)
    elif "gemma3-1b" in model_name:
        return gemma_lm_module_name_fn(module_type)
    elif "gemma" in model_name:
        return gemma_multimodal_module_name_fn(module_type)
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
        ("qwen3-4b", 36, "model.embed_tokens"),
    ]

    for model_name, n_layers, embedding_name in models_n_layers_embedding_names:
        for module_type in ["input", "intermediate", "gate", "up_proj", "all_inputs"]:
            module_names = get_model_layer_config(model_name, module_type)
            layers_to_sparsify = {"L": f"0:{n_layers-1}"}
            config = {
                "embedding_layer_name": embedding_name,
                "modules_to_sparsify": module_names,
                "layers_to_sparsify": layers_to_sparsify,
            }
            config_path = root / f"{model_name}_{module_type}.json"
            with config_path.open("w+") as f:
                json.dump(config, f, indent=4)
