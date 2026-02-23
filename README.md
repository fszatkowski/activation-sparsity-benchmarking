# Activation Sparsity Benchmarking

Code for the paper **"Universal Properties of Activation Sparsity in Modern Large Language Models"** published at **ICLR 2026**.

**Authors:** Filip Szatkowski, Patryk Będkowski, Alessio Devoto, Jan Dubiński, Pasquale Minervini, Mikołaj Piórczyński, Simone Scardapane, Bartosz Wójcik

[[Paper (arXiv)]](https://arxiv.org/abs/2509.00454)

## Overview

This repository provides tools for systematically evaluating activation sparsity robustness in modern Large Language Models. We extend the [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) with sparsification capabilities that allow zeroing out activations in FFN layers according to configurable rules and thresholds, and then measuring the impact on downstream task performance.

## Repository Structure

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

## Sparsified Harness

The `sparsified_harness` directory contains a modified version of the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (v0.4.7) with added activation sparsification support. The key additions are:

- **Sparsification Manager** — hooks into model forward passes to zero out activations in FFN layers based on a chosen rule (`topp`, `topk`, `maxp`) and threshold.
- **Sparsification Configs** — JSON configuration files that specify which modules to sparsify for each model architecture (Llama 3, Qwen 2.5, Gemma 3, OLMoE, etc.).
- **Activation Monitoring** — optional recording of activation statistics and effective ranks.

### Installation

```bash
cd sparsified_harness
pip install -e .
```

You will also need to set the following environment variables:

- `HF_TOKEN` — your Hugging Face access token
- `HF_HOME` — directory for caching models and datasets

### Usage

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

### Reproducing Paper Experiments

The grid search scripts under `slurm_scripts/harness/model_grids/` automate the full sweep of models, sparsification targets, and thresholds reported in the paper. For example:

```bash
# Set required SLURM variables
export SLURM_ACC=<your_account>
export SLURM_PARTITION=<your_partition>

# Launch the Llama evaluation grid
bash slurm_scripts/harness/model_grids/llama_base.sh
```

> **Note:** The SLURM scripts were developed for use on specific compute clusters and may require adaptation for your environment (e.g., conda environment names, module loads, GPU configurations, partition names). Refer to `sparsified_harness/SETUP.md` for environment variable details.

## LLaDA Evaluations

The `sparsified_llada/` directory will contain evaluation code for activation sparsity experiments on diffusion-based LLMs (LLaDA). This code is coming soon.

## Citation

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
