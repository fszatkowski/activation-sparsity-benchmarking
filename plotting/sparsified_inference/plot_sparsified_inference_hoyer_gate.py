from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plotting.harness_parsing import read_sparsified_harness_results

model_dict = {
    "llama3-1b": "Llama3.1-1B",
    "gemma2-2b": "Gemma2-2B",
    "qwen2_5-1_5b": "Qwen2.5-1.5B",
}
fontsize = 16
fontsize_ticks = 12

if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent

    results_dir = root / "sparsified_evaluation_hoyer" / "llama3-1b-hoyer-gate"
    assert results_dir.exists(), f"Results directory {results_dir} does not exist."

    all_data_abs = []
    all_data_norm = []

    setups = sorted(
        list(set([child for child in results_dir.glob("*") if child.is_dir()]))
    )
    for setup in setups:
        sparsification_loss = setup.name
        model = setup.parent.name
        print(f"Processing: {sparsification_loss} sparsification for {model}")
        sparsity_stat_files = list(setup.rglob("sparsity_stats.json"))
        try:
            df = read_sparsified_harness_results(sparsity_stat_files)
        except Exception as e:
            print(f"Error processing sparsity stats files for {setup}")
            continue
        df["sparsification_loss"] = sparsification_loss
        df["model"] = model
        all_data_abs.append(df)

        df_norm = df.copy()
        # Within task normalize by max acc
        for task in df_norm["task"].unique():
            task_df = df_norm[df_norm["task"] == task]
            max_acc = task_df["acc"].max()
            df_norm.loc[df_norm["task"] == task, "acc"] = task_df["acc"] / max_acc
        all_data_norm.append(df_norm)

    setups = sorted(
        list(
            set(
                [
                    child
                    for child in (root / "sparsified_evaluation").glob("*/*")
                    if child.is_dir()
                ]
            )
        )
    )
    for setup in setups:
        sparsification_module = setup.name
        model = setup.parent.name
        if "llama" not in model:
            continue
        if sparsification_module != "gate":
            continue
        sparsity_stat_files = list(setup.rglob("sparsity_stats.json"))
        sparsity_stat_files = [f for f in sparsity_stat_files if "topp" in str(f)]
        df = read_sparsified_harness_results(sparsity_stat_files)
        df["sparsification_loss"] = "baseline"
        df["model"] = model
        all_data_abs.append(df)

        df_norm = df.copy()
        # Within task normalize by max acc
        for task in df_norm["task"].unique():
            task_df = df_norm[df_norm["task"] == task]
            max_acc = task_df["acc"].max()
            df_norm.loc[df_norm["task"] == task, "acc"] = task_df["acc"] / max_acc
        all_data_norm.append(df_norm)

    # Now plot everything together with style depending on the sparsification_module
    all_data_abs_df = pd.concat(all_data_abs, ignore_index=True)
    all_data_norm_df = pd.concat(all_data_norm, ignore_index=True)

    for key, data_df in [("abs", all_data_abs_df), ("norm", all_data_norm_df)]:
        tasks = sorted(data_df["task"].unique().tolist())
        for task in tasks:
            sparsification_losses = sorted(
                data_df["sparsification_loss"].unique().tolist()
            )
            plt.cla()
            plt.clf()
            plt.figure()

            task_df = data_df[data_df["task"] == task]
            plot = sns.lineplot(
                data=task_df,
                x="sparsity",
                y="acc",
                hue="sparsification_loss",
                dashes=True,
                markers=True,
            )
            plot.set_ylabel("Accuracy", fontsize=fontsize)
            plot.set_xlabel("Sparsity", fontsize=fontsize)
            plot.set_title(
                f"Hoyer sparsification comparison ({key}), {task}",
                fontsize=fontsize,
            )
            plot.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
            # Set legend title and fontsize
            handles, labels = plot.get_legend_handles_labels()
            plot.legend(
                handles=handles,
                labels=labels,
                title=None,
                title_fontsize=fontsize,
                fontsize=fontsize_ticks,
                loc="lower left",
            )
            plt.tight_layout()
            save_path = results_dir / f"{key}_sparsity_{task}.png"
            print(f"Saving plot to {save_path}")
            plot.get_figure().savefig(save_path)
            plt.close()
