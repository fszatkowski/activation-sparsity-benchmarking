from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plotting.harness_parsing import read_standard_harness_results

sns.set_style("whitegrid")


config_keys: List[str] = ["lr", "sparsity_loss", "sparsity_weight"]


def parser(path: Path) -> Dict:
    run_name = path.parent.parent.name
    if "sparsity-loss" in run_name:
        sparsity_loss = run_name.split("sparsity-loss-")[1].split("-")[0]
        if sparsity_loss != "none":
            sparsity_weight = float(run_name.split("-w")[1].split("-")[0])
        else:
            sparsity_weight = 0.0
        lr = float(run_name.split("lr-")[1])
    else:
        lr = float(run_name.split("lr-")[1])
        sparsity_loss = "none"
        sparsity_weight = 0.0
    output = {
        "lr": lr,
        "sparsity_loss": sparsity_loss,
        "sparsity_weight": sparsity_weight,
    }
    return output


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent / "results" / "continued_ft_fineweb_200m"
    csv_file = Path(root) / "parsed_results.csv"
    if not csv_file.exists():
        result_files = list(root.rglob("**/results*.json"))
        result_files = [p for p in result_files if not "sparsity-loss-none" in str(p)]

        df = read_standard_harness_results(result_files, parser, config_keys)
        df.to_csv(root / "parsed_results.csv", index=False)
    else:
        print(f"File {csv_file} already exists. Reading already parsed results.")
        # Read the parsed results
        df = pd.read_csv(root / "parsed_results.csv")

    columns = config_keys + ["task", "acc"]
    df = df[columns]

    # Plot baseline results
    task_order = sorted(list(set(df["task"])))
    task_order.remove("average")
    task_order.remove("average_likelihood")
    task_order.remove("average_generative")
    task_order = task_order + ["average", "average_likelihood", "average_generative"]

    baseline_df = df[df["sparsity_loss"] == "none"]
    # Sort the df by task name according to order in the dict
    baseline_df["task"] = pd.Categorical(
        baseline_df["task"], categories=task_order, ordered=True
    )
    baseline_df = baseline_df.sort_values(by=["task"])
    baseline_df["label"] = baseline_df.apply(lambda row: f"LR={row['lr']:.0e}", axis=1)
    baseline_df["task"] = (
        baseline_df["task"]
        .str.replace("_easy", "_e")
        .str.replace("_challenge", "_c")
        .str.replace("mmlu_redux_generative", "mmlu_r")
        .str.replace("average", "avg")
        .str.replace("loglikelihood", "llh")
        .str.replace("generative", "gen")
    )

    plt.figure()
    plt.cla()
    plt.clf()
    plot = sns.barplot(
        data=baseline_df,
        x="task",
        y="acc",
        hue="label",
    )

    # Make x ticks for tasks rotated
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    plot.set_xlabel(None)
    plot.set_ylabel("Accuracy")
    plot.set_title("Baseline results for 200M tokens on FineWeb", fontsize=16)

    # Set legend title and fontsize
    handles, labels = plot.get_legend_handles_labels()
    plot.legend(
        handles=handles,
        labels=labels,
        title=None,
        fontsize=8,
    )
    plt.tight_layout()
    plot.get_figure().savefig(root / "baseline_results.png")
    plt.close()

    # Plot results for sparsity enforcement losses - hoyer, l1, l2
    baseline_result = df[(df["lr"] == 5e-6) & (df["sparsity_loss"] == "none")]
    baseline_result["label"] = "Baseline"
    for loss_type in ["hoyer", "l1", "l2"]:
        loss_df = df[df["sparsity_loss"] == loss_type]
        loss_df["label"] = loss_df.apply(
            lambda row: f"{row['sparsity_loss'].title()}, w={row['sparsity_weight']}",
            axis=1,
        )
        if loss_type == "hoyer" or loss_type == "l1":
            loss_df = loss_df[loss_df["sparsity_weight"] <= 0.00001]
        elif loss_type == "l2":
            loss_df = loss_df[loss_df["sparsity_weight"] <= 0.001]

        # Merge loss df and baseline
        loss_df = pd.concat([baseline_result, loss_df], ignore_index=True)
        loss_df["task"] = pd.Categorical(
            loss_df["task"], categories=task_order, ordered=True
        )
        loss_df = loss_df.sort_values(by=["task", "sparsity_weight"])
        loss_df["task"] = (
            loss_df["task"]
            .str.replace("_easy", "_e")
            .str.replace("_challenge", "_c")
            .str.replace("mmlu_redux_generative", "mmlu_r")
            .str.replace("average", "avg")
            .str.replace("loglikelihood", "llh")
            .str.replace("generative", "gen")
        )

        plt.figure()
        plt.cla()
        plt.clf()

        plot = sns.barplot(
            data=loss_df,
            x="task",
            y="acc",
            hue="label",
        )
        # Make x ticks for tasks rotated
        plot.set_xticklabels(
            plot.get_xticklabels(), rotation=45, ha="right", fontsize=10
        )
        plot.set_xlabel(None)
        plot.set_ylabel("Accuracy")
        plot.set_title(
            f"{loss_type.title()} results for 200M tokens on FineWeb", fontsize=16
        )

        # Set legend title and fontsize
        handles, labels = plot.get_legend_handles_labels()
        plot.legend(
            handles=handles,
            labels=labels,
            title=None,
            fontsize=8,
        )
        plt.tight_layout()
        plot.get_figure().savefig(root / f"{loss_type}_results.png")
        plt.close()
