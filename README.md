# Activation Sparsity Benchmarking

Code for the paper **"Universal Properties of Activation Sparsity in Modern Large Language Models"** published at **ICLR 2026** 🎉

**Authors:** Filip Szatkowski, Patryk Będkowski, Alessio Devoto, Jan Dubiński, Pasquale Minervini, Mikołaj Piórczyński, Simone Scardapane, Bartosz Wójcik

[![arXiv](https://img.shields.io/badge/arXiv-2509.00454-b31b1b.svg)](https://arxiv.org/abs/2509.00454)

## 🔍 Overview

This repository provides tools for systematically evaluating activation sparsity robustness in modern Large Language Models. We extend the [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) with sparsification capabilities that allow zeroing out activations in FFN layers according to configurable rules and thresholds, and then measuring the impact on downstream task performance.

## 📁 Repository Structure

```
activation_sparsity_benchmarking/
├── sparsified_harness/    # Main evaluation codebase (based on lm-evaluation-harness)
│   ├── lm_eval/
│   │   ├── sparsification_manager.py       # Activation sparsification for dense models
│   │   ├── sparsification_manager_moe.py   # Activation sparsification for MoE models
│   │   ├── sparsification_utils.py         # Sparsification utilities and statistics
│   │   ├── activations_monitor.py          # Activation monitoring during evaluation
│   │   ├── sparsification_configs/         # Pre-built JSON configs for all evaluated models
│   │   └── tasks/                          # Evaluation task definitions
│   └── slurm_scripts/                      # SLURM job scripts for running experiments
│       └── harness/
│           ├── sparse_eval.sh              # Main sparsified evaluation script
│           ├── sparse_eval_moe.sh          # MoE model evaluation script
│           └── model_grids/                # Grid search scripts per model family
└── sparsified_llada/      # LLaDA (diffusion LLM) evaluation code (coming soon)
```

## 🔧 Sparsified Harness

The `sparsified_harness` directory contains a modified version of the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (v0.4.7) with added activation sparsification support. The key additions are:

- **Sparsification Manager** — hooks into model forward passes to zero out activations in FFN layers based on a chosen rule (`topp`, `topk`, `maxp`) and threshold.
- **Sparsification Configs** — JSON configuration files that specify which modules to sparsify for each model architecture (Llama 3, Qwen 2.5, Gemma 3, OLMoE, etc.).
- **Activation Monitoring** — optional recording of activation statistics and effective ranks.

### 📦 Installation

```bash
cd sparsified_harness
pip install -e .
```

You will also need to set the following environment variables:

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Your Hugging Face access token |
| `HF_HOME` | Directory for caching models and datasets |

### 🚀 Usage

Run a sparsified evaluation directly:

```bash
python -m lm_eval.__main__ \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B,dtype=bfloat16 \
    --tasks arc_easy,hellaswag,winogrande \
    --batch_size auto \
    --output_path results/ \
    --sparsification_config lm_eval/sparsification_configs/llama3-1b_input.json \
    --sparsification_rule topp \
    --sparsification_th_val 0.90
```

The sparsification configs define which layers and modules to target. You can generate new configs or modify existing ones via `lm_eval/sparsification_configs/create_configs.py`.

The sparsification code should work with most HuggingFace transformer models out of the box — just make sure to provide a matching sparsification config that correctly maps the module names for your architecture.

> **Tip:** If you're evaluating instruction-tuned or chat models, double-check that the chat template and generation settings (e.g., `max_gen_toks`, system prompts) are configured correctly for your model. Mismatched templates can significantly affect results.

### 🧪 Reproducing Paper Experiments (SLURM)

We provide SLURM grid search scripts under `slurm_scripts/harness/model_grids/` as **skeletons** for reproducing the experiments from the paper. These scripts sweep over models, sparsification targets, and thresholds, and submit jobs via `sbatch`. Example:

```bash
export SLURM_ACC=<your_account>
export SLURM_PARTITION=<your_partition>

bash slurm_scripts/harness/model_grids/llama_base.sh
```

> ⚠️ **Important:** These scripts are **not** turnkey reproducibility scripts — they were written for specific compute clusters (with particular conda environments, module systems, and GPU setups) and should be treated as **templates**. You will likely need to adapt environment activation, partition names, GPU resource requests, and paths to match your infrastructure. See `sparsified_harness/SETUP.md` for details on required environment variables.

## 🧬 LLaDA Evaluations

The `sparsified_llada` directories:

- **`scripts/`** — Contains modified source files from the [LLaDA repository](https://github.com/ML-GSAI/LLaDA). We integrated custom hooks to apply layer-wise sparsity and record sparsity logs using configurable rules (`topp`, `topk`) and thresholds.
- **`slurm_scripts/`** — Contains SLURM batch scripts for executing grid parameter sweeps on the LLaDA 8B model across various sparsity rules and thresholds, alongside scripts to reproduce our sparsity results.
- 
### 📦 Installation

```bash
cd sparsified_llada
pip install -e .
```

### 🧪 Reproducing Paper Experiments (SLURM)

We provide SLURM grid search scripts under `slurm_scripts/harness/model_grids/` as **skeletons** for reproducing the experiments from the paper. These scripts sweep over models, sparsification targets, and thresholds, and submit jobs via `sbatch`. Example:

Modify the scripts in `sparsified_llada/scripts/` to set the desired sparsification rule and threshold, and then run sparsified inference directly:

```bash
export SLURM_ACC=<your_account>
export SLURM_PARTITION=<your_partition>

bash llada_input_intermediate.sh
bash llada_intermediate.sh
```

For each of the setting, the run will start the evaluation and generate sparsity logs with the activation statistics and effective ranks for each layer. You can modify the scripts to run on different datasets or with different generation settings.

> ⚠️ **Important:** These scripts are **not** turnkey reproducibility scripts — they were written for specific compute clusters (with particular conda environments, module systems, and GPU setups) and should be treated as **templates**. You will likely need to adapt environment activation, partition names, GPU resource requests, and paths to match your infrastructure. See `sparsified_harness/SETUP.md` for details on required environment variables.

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{szatkowski2026universal,
    title={Universal Properties of Activation Sparsity in Modern Large Language Models},
    author={Szatkowski, Filip and Będkowski, Patryk and Devoto, Alessio and Dubiński, Jan and Minervini, Pasquale and Piórczyński, Mikołaj and Scardapane, Simone and Wójcik, Bartosz},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://arxiv.org/abs/2509.00454}
}
```
