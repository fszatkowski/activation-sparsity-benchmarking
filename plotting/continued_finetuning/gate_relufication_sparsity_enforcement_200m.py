from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plotting.harness_parsing import read_standard_harness_results

sns.set_style("whitegrid")


config_keys: List[str] = ["label"]


def parser(path: Path) -> Dict:
    run_name = path.parent.parent.name
    if "relufication" in run_name:
        label = run_name.split('relufication-')[1]
    else:
        label = "baseline"
    
    output = {
        "label": label
    }
    return output


if __name__ == "__main__":
    root = (
        Path(__file__).parent.parent.parent
        / "results"
        / "gate_relufication_fineweb_200m"
    )
    baseline_dir = Path(__file__).parent.parent.parent / 'results' / 'intermediate_sparsification_fineweb_200m'
    csv_file = Path(root) / "parsed_results.csv"
    if not csv_file.exists():
        result_files = list(root.rglob("**/results*.json"))
        df = read_standard_harness_results(result_files, parser, config_keys)

        baseline_files = list(f for f in baseline_dir.rglob("**/results*.json"))
        baseline_files = [f for f in baseline_files if 'sparsity-loss-none' in str(f)]
        baseline_df = read_standard_harness_results(baseline_files, parser, config_keys)

        df = pd.concat([df, baseline_df], ignore_index=True)
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

    # Sort the df by task name according to order in the dict
    df["task"] = pd.Categorical(
        df["task"], categories=task_order, ordered=True
    )
    df = df.sort_values(by=["task"])
    df["task"] = (
        df["task"]
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
        data=df,
        x="task",
        y="acc",
        hue="label",
    )

    # Make x ticks for tasks rotated
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    plot.set_xlabel(None)
    plot.set_ylabel("Accuracy")
    plot.set_title("Relufication results for 200M tokens on FineWeb", fontsize=16)

    # Set legend title and fontsize
    handles, labels = plot.get_legend_handles_labels()
    plot.legend(
        handles=handles,
        labels=labels,
        title=None,
        fontsize=8,
    )
    plt.tight_layout()
    plot.get_figure().savefig(root / "results.png")
    plt.close()


    # Plot separate plot that only shows likelihood tasks
    likelihood_tasks = ['arc_c', 'arc_e', 'boolq', 'hellaswag', 'lambada', 'piqa', 'sciq', 'triviaqa', 'winogrande', 'avg_likelihood']
    df = df[df["task"].isin(likelihood_tasks)]
    plt.cla()
    plt.clf()
    plt.figure()

    plot = sns.barplot(
        data=df,
        x="task",
        y="acc",
        hue="label",
    )

    # Make x ticks for tasks rotated
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    plot.set_xlabel(None)
    plot.set_ylabel("Accuracy")
    plot.set_title("Relufication results for 200M tokens on FineWeb", fontsize=16)

    # Set legend title and fontsize
    handles, labels = plot.get_legend_handles_labels()
    plot.legend(
        handles=handles,
        labels=labels,
        title=None,
        fontsize=8,
    )
    plt.tight_layout()
    plot.get_figure().savefig(root / "results_loglikelihood.png")
    plt.close()
