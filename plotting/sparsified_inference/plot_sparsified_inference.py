import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plotting.harness_parsing import read_sparsified_harness_results

model_dict = {
    "llama3-1b": "Llama3.2-1B",
    "llama3-1b-instruct": "Llama3.2-1B-Instruct",
    "llama3-3b": "Llama3.2-3B",
    "llama3-3b-instruct": "Llama3.2-3B-Instruct",
    "gemma2-2b": "Gemma2-2B",
    "qwen2_5-1_5b": "Qwen2.5-1.5B",
}
fontsize = 16
fontsize_ticks = 12

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    root = Path(__file__).parent.parent.parent

    results_dir = root / args.results_dir
    assert results_dir.exists(), f"Results directory {results_dir} does not exist."

    all_data = []
    setups = sorted(
        list(set([child for child in results_dir.glob("*/*") if child.is_dir()]))
    )
    for setup in setups:
        sparsification_module = setup.name
        model = setup.parent.name
        print(f"Processing: {sparsification_module} sparsification for {model}")
        sparsity_stat_files = list(setup.rglob("sparsity_stats.json"))
        df = read_sparsified_harness_results(sparsity_stat_files)
        df["sparsification_module"] = sparsification_module
        df["model"] = model

        all_data.append(df)
        df.to_csv(setup / "sparsification_stats.csv", index=False)

        tasks = sorted(df["task"].unique().tolist())
        for task in tasks:
            task_df = df[df["task"] == task]

            plt.cla()
            plt.clf()
            plt.figure()

            plot = sns.lineplot(
                data=task_df,
                x="sparsity",
                y="acc",
                hue="sparsification_rule",
                style="sparsification_rule",
                markers=True,
                dashes=False,
            )
            plot.set_ylabel("Accuracy", fontsize=fontsize)
            plot.set_xlabel("Sparsity", fontsize=fontsize)
            plot.set_title(
                f"{model}, {sparsification_module} sparsification, {task}",
                fontsize=fontsize,
            )

            plot.tick_params(axis="both", which="major", labelsize=fontsize_ticks)

            # Set legend title and fontsize
            handles, labels = plot.get_legend_handles_labels()
            plot.legend(
                handles=handles,
                labels=labels,
                title="Rule",
                title_fontsize=fontsize,
                fontsize=fontsize_ticks,
                loc="lower left",
            )

            plt.tight_layout()
            save_path = setup / f"{task}_sparsity.png"
            print(f"Saving plot to {save_path}")
            plot.get_figure().savefig(save_path)
            plt.close()

    # Now plot everything together with style depending on the sparsification_module
    all_data_df = pd.concat(all_data, ignore_index=True)
    models = sorted(
        list(set([child.name for child in results_dir.glob("*") if child.is_dir()]))
    )
    tasks = sorted(all_data_df["task"].unique().tolist())
    for model in models:
        model_df = all_data_df[all_data_df["model"] == model]
        for task in tasks:
            sparsification_modules = sorted(
                model_df["sparsification_module"].unique().tolist()
            )
            plt.cla()
            plt.clf()
            plt.figure()

            task_df = model_df[model_df["task"] == task]
            plot = sns.lineplot(
                data=task_df,
                x="sparsity",
                y="acc",
                hue="sparsification_rule",
                style="sparsification_module",
                dashes=True,
                markers=True,
            )
            plot.set_ylabel("Accuracy", fontsize=fontsize)
            plot.set_xlabel("Sparsity", fontsize=fontsize)
            plot.set_title(
                f"{model_dict[model]}, Sparsification comparison, {task}",
                fontsize=fontsize,
            )
            plot.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
            # Set legend title and fontsize
            handles, labels = plot.get_legend_handles_labels()
            # Replace "sparsification_module" with "Module" in labels
            labels = [
                label.replace("sparsification_module", "Module") for label in labels
            ]
            labels = [label.replace("sparsification_rule", "Rule") for label in labels]
            plot.legend(
                handles=handles,
                labels=labels,
                title=None,
                title_fontsize=fontsize,
                fontsize=fontsize_ticks,
                loc="lower left",
            )
            plt.tight_layout()
            save_path = results_dir / model / f"sparsity_{task}.png"
            print(f"Saving plot to {save_path}")
            plot.get_figure().savefig(save_path)
            plt.close()

    # Now for all setups, lets plot 4 setups side by side and compare topp inference curves on average loglikelihood tasks
    plt.cla()
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5), sharey=True)
    colormap = {
        "llama3-1b": "blue",
        "gemma2-2b": "orange",
        "qwen2_5-1_5b": "green",
    }
    for i, setup_name in enumerate(["input", "up_proj", "gate", "intermediate"]):
        setup_df = all_data_df[all_data_df["sparsification_module"] == setup_name]
        rule_df = setup_df[setup_df["sparsification_rule"] == "topp"]
        avg_likelihood_df = rule_df[rule_df["task"].isin(["average_likelihood"])]

        plot = sns.lineplot(
            data=avg_likelihood_df,
            x="sparsity",
            y="acc",
            hue="model",
            palette=colormap,
            markers=True,
            ax=axes[i],
            legend=i == 0,
        )
        plot.set_title(f"{setup_name} sparsification (topp)", fontsize=fontsize)
        plot.set_xlabel("Sparsity", fontsize=fontsize)
        if i == 0:
            plot.set_ylabel("Average Accuracy", fontsize=fontsize)
        plot.tick_params(axis="both", which="major", labelsize=fontsize_ticks)

        # Set legend title and fontsize on the leftmost plot, delete legend in all other
        if i == 0:
            handles, labels = plot.get_legend_handles_labels()
            plot.legend(
                handles=handles,
                labels=labels,
                title="Model",
                title_fontsize=fontsize,
                fontsize=fontsize_ticks,
                loc="lower left",
            )
    plt.tight_layout()
    save_path = results_dir / "average_likelihood_sparsity_per_model.png"
    print(f"Saving merged plot to {save_path}")
    fig.savefig(save_path)
    plt.close(fig)

    # Now plot the same, but noramalize the accuracy for each model with the max within the setup
    plt.cla()
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5), sharey=True)
    for i, setup_name in enumerate(["input", "up_proj", "gate", "intermediate"]):
        setup_df = all_data_df[all_data_df["sparsification_module"] == setup_name]
        rule_df = setup_df[setup_df["sparsification_rule"] == "topp"]
        avg_likelihood_df = rule_df[rule_df["task"].isin(["average_likelihood"])]

        # Normalize accuracy by the max accuracy for each model
        model_dfs = []
        for model in avg_likelihood_df["model"].unique():
            model_df = avg_likelihood_df[avg_likelihood_df["model"] == model]
            max_acc = model_df["acc"].max()
            model_df["acc"] = model_df["acc"] / max_acc
            model_dfs.append(model_df)
        avg_likelihood_df = pd.concat(model_dfs, ignore_index=True)

        plot = sns.lineplot(
            data=avg_likelihood_df,
            x="sparsity",
            y="acc",
            hue="model",
            palette=colormap,
            markers=True,
            ax=axes[i],
            legend=i == 0,
        )
        plot.set_title(f"{setup_name} sparsification (topp)", fontsize=fontsize)
        plot.set_xlabel("Sparsity", fontsize=fontsize)
        if i == 0:
            plot.set_ylabel("Normalized Accuracy", fontsize=fontsize)
        plot.tick_params(axis="both", which="major", labelsize=fontsize_ticks)

        # Set legend title and fontsize on the leftmost plot, delete legend in all other
        if i == 0:
            handles, labels = plot.get_legend_handles_labels()
            plot.legend(
                handles=handles,
                labels=labels,
                title="Model",
                title_fontsize=fontsize,
                fontsize=fontsize_ticks,
                loc="lower left",
            )
    plt.tight_layout()
    save_path = results_dir / "average_likelihood_sparsity_per_model_normalized.png"
    print(f"Saving normalized merged plot to {save_path}")
    fig.savefig(save_path)
