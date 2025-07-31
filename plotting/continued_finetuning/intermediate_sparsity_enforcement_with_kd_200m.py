from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plotting.harness_parsing import read_standard_harness_results

sns.set_style("whitegrid")


config_keys: List[str] = ["lr", "sparsity_loss", "sparsity_weight", "kd_weight"]


def parser(path: Path) -> Dict:
    run_name = path.parent.parent.name
    if "sparsity-loss" in run_name:
        sparsity_loss = run_name.split("sparsity-loss-")[1].split("-")[0]
        if sparsity_loss != "none":
            sparsity_weight = float(run_name.split("-w")[1].split("-")[0])
        else:
            sparsity_weight = 0.0
        lr = float(run_name.split("lr-")[1].split('-kd')[0])
        if 'kd' in run_name:
            kd_weight = float(run_name.split("kd-w-")[1].split('-')[0])
        else:
            kd_weight = 0.0
    else:
        lr = float(run_name.split("lr-")[1])
        sparsity_loss = "none"
        sparsity_weight = 0.0
        kd_weight = 0.0
    output = {
        "lr": lr,
        "sparsity_loss": sparsity_loss,
        "sparsity_weight": sparsity_weight,
        "kd_weight": kd_weight,
    }
    return output


if __name__ == "__main__":
    root = (
        Path(__file__).parent.parent.parent
        / "results"
        / "intermediate_sparsification_kd_fineweb_200m"
    )
    csv_file = Path(root) / "parsed_results.csv"
    result_files = list(root.rglob("**/results*.json"))
    df = read_standard_harness_results(result_files, parser, config_keys)
    df.to_csv(root / "parsed_results.csv", index=False)

    columns = config_keys + ["task", "acc"]
    df = df[columns]

    # Plot baseline results
    task_order = sorted(list(set(df["task"])))
    task_order.remove("average")
    task_order.remove("average_likelihood")
    # task_order.remove("average_generative")
    task_order = task_order + ["average", "average_likelihood"] # , "average_generative"]

    loss_type = "hoyer"
    loss_df = df[df["sparsity_loss"] == loss_type]
    loss_df["label"] = loss_df.apply(
        lambda row: f"KD_w={row['kd_weight']}",
        axis=1,
    )

    # Merge loss df and baseline
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
        f"{loss_type.title()} KD results for 200M tokens on FineWeb", fontsize=16
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
