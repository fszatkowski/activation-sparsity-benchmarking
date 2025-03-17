import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

task_names = [
    "arc_easy",
    "arc_challenge",
    "boolq",
    "triviaqa",
    "gsm8k",
    "mmlu_redux_generative",
    "hellaswag",
    "piqa",
    "sciq",
]


def plot(
    outputs,
    output_path,
    filter_tasks: list = None,
    style=None,
    model_name="Llama3.2-1B",
):
    if filter_tasks:
        outputs = [o for o in outputs if o["task"] in filter_tasks]

    df = pd.DataFrame(outputs)
    df = df.sort_values(by=["task", "sparsity"], ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot = sns.lineplot(
        data=df,
        x="sparsity",
        y="accuracy",
        hue="task",
        markers=True,
        ax=axes[0],
        legend=False,
        style=style,
    )
    plot.set(xlabel="Sparsity", ylabel="Performance")
    plot.set_title(f"{model_name} - absolute performance")

    # Normalize accuracy for each task with the accuracy for the minimal sparsity
    df["accuracy_normalized"] = df.groupby("task")["accuracy"].transform(
        lambda x: x / x.max()
    )
    plot = sns.lineplot(
        data=df,
        x="sparsity",
        y="accuracy_normalized",
        hue="task",
        markers=True,
        ax=axes[1],
        legend=True,
        style=style,
    )
    plot.set(xlabel="Sparsity", ylabel="Normalized Performance")
    plot.set_title(f"{model_name} - normalized performance")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)


if __name__ == "__main__":
    for model_key, model_name in ("llama3-3b", "Llama3.2-3B-Instruct"), (
        "llama3-8b",
        "Llama3.1-8B-Instruct",
    ):
        results_dir = Path(__file__).parent.parent / "results" / model_key

        outputs = []
        for dirname in ["input_sparsification", "intermediate_sparsification"]:
            root = results_dir / dirname
            task_dirs = [d for d in sorted(list(root.glob("*"))) if d.is_dir()]
            for task_dir in task_dirs:
                task_name = task_dir.stem
                if task_name not in task_names:
                    continue
                sparsity_subdirs = [
                    d for d in sorted(list(task_dir.glob("*"))) if d.is_dir()
                ]

                for sparsity_subdir in sparsity_subdirs:
                    results_path = list(sparsity_subdir.rglob("results*.json"))[0]
                    sparsity_stats_path = sparsity_subdir / "sparsity_stats.json"

                    with results_path.open("r") as f:
                        results = json.load(f)["results"]
                    with sparsity_stats_path.open("r") as f:
                        sparsity_stats = json.load(f)

                    sparsity = sparsity_stats["sparsity_stats"]["total"] * 100
                    topp = sparsity_stats["topp"]

                    subtask_names = results.keys()
                    if "mmlu_redux_generative" in subtask_names:
                        subtask_names = [
                            n for n in subtask_names if "mmlu" in n and "redux" not in n
                        ]
                        acc = [results[n]["exact_match,default"] for n in subtask_names]
                        acc = sum(acc) / len(acc)
                        outputs.append(
                            {
                                "task": "mmlu_redux",
                                "sparsity": sparsity,
                                "topp": topp,
                                "accuracy": acc,
                                "sparsity_mode": dirname.replace("_sparsification", ""),
                            }
                        )
                    else:
                        if "mmlu_pro" in subtask_names:
                            subtask_names = ["mmlu_pro"]

                        for subtask_name in subtask_names:
                            if "exact_match,remove_whitespace" in results[subtask_name]:
                                accuracy = results[subtask_name][
                                    "exact_match,remove_whitespace"
                                ]
                            elif "exact_match,strict-match" in results[subtask_name]:
                                accuracy = results[subtask_name][
                                    "exact_match,strict-match"
                                ]
                            elif "acc,none" in results[subtask_name]:
                                accuracy = results[subtask_name]["acc,none"]
                            elif "prompt_level_loose_acc,none" in results[subtask_name]:
                                accuracy = results[subtask_name][
                                    "prompt_level_loose_acc,none"
                                ]
                            elif "exact_match,custom-extract" in results[subtask_name]:
                                accuracy = results[subtask_name][
                                    "exact_match,custom-extract"
                                ]
                            elif "exact_match,default" in results[subtask_name]:
                                accuracy = results[subtask_name]["exact_match,default"]
                            else:
                                print(
                                    f"No accuracy key found in results for task {task_name} in {sparsity_subdir}. Possible keys: {results[subtask_name].keys()}"
                                )
                                continue

                            outputs.append(
                                {
                                    "task": subtask_name,
                                    "sparsity": sparsity,
                                    "topp": topp,
                                    "accuracy": accuracy,
                                    "sparsity_mode": dirname.replace(
                                        "_sparsification", ""
                                    ),
                                }
                            )

        plot(
            [o for o in outputs if o["sparsity_mode"] == "input"],
            results_dir / "input_sparsity_per_task.png",
            model_name=model_name,
            style="task",
        )
        plot(
            [o for o in outputs if o["sparsity_mode"] == "intermediate"],
            results_dir / "itermediate_sparsity_per_task.png",
            model_name=model_name,
            style="task",
        )
        plot(
            outputs,
            results_dir / "agg_sparsity_per_task.png",
            style="sparsity_mode",
            model_name=model_name,
        )
