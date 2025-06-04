import json
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd

# Add 'average_likelihood' and 'average_generative' for specific tasks
generative_tasks = ["gsm8k", "ifeval", "mmlu_redux_generative"]
likelihood_tasks = [
    "boolq",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "sciq",
    "winogrande",
    "lambada",
    "triviaqa",
]
all_tasks = generative_tasks + likelihood_tasks


def _extract_metrics(results: Dict) -> Dict:
    tasks = list(results.keys())
    if len(tasks) == 1:
        task = tasks[0]
    elif len(tasks) > 1:
        if any(["lambada" in k for k in tasks]):
            task = "lambada"
        elif "mmlu_redux_generative" in tasks:
            task = "mmlu_redux_generative"
        else:
            raise ValueError(
                f"Expected one task in results, found {len(results)}: {list(results.keys())}"
            )

    if task in ("boolq", "winogrande"):
        output = {"task": task, "acc": results[task]["acc,none"] * 100}
    elif task in ("arc_easy", "arc_challenge", "hellaswag", "piqa", "sciq"):
        output = {
            "task": task,
            "acc": results[task]["acc,none"] * 100,
            "acc_norm": results[task]["acc_norm,none"] * 100,
        }
    elif task in ("gsm8k"):
        output = {
            "task": task,
            "acc": results[task]["exact_match,strict-match"] * 100,
            "acc_flex": results[task]["exact_match,flexible-extract"] * 100,
        }
    elif task in ("triviaqa"):
        output = {
            "task": task,
            "acc": results[task]["exact_match,remove_whitespace"] * 100,
        }
    elif task in ("ifeval"):
        # Use average of prompt and inst accuracies as a main metric as per:
        # https://github.com/EleutherAI/lm-evaluation-harness/issues/2200?utm_source=chatgpt.com
        output = {
            "task": task,
            "acc": 100
            * (
                results[task]["prompt_level_strict_acc,none"]
                + results[task]["inst_level_strict_acc,none"]
            )
            / 2,
            "acc_prompt_strict": results[task]["prompt_level_strict_acc,none"] * 100,
            "acc_inst_strict": results[task]["inst_level_strict_acc,none"] * 100,
            "acc_prompt_loose": results[task]["prompt_level_loose_acc,none"] * 100,
            "acc_inst_loose": results[task]["inst_level_loose_acc,none"] * 100,
        }
    elif task in ("lambada"):
        results = results["lambada_standard"]
        output = {
            "task": task,
            "acc": results["acc,none"] * 100,
        }
    elif task in ("mmlu_redux_generative"):
        scores = [
            results[k]["exact_match,default"]
            for k, v in results.items()
            if "exact_match,default" in v
        ]
        avg = sum(scores) / len(scores) if scores else 0
        output = {
            "task": task,
            "acc": avg * 100,
        }
    else:
        raise ValueError(f"Unknown alias {results['alias']} in results.")

    assert "task" in output, f"Task not found in output: {output}"
    assert "acc" in output, f"Accuracy not found in output: {output}"
    assert (
        output["task"] in all_tasks
    ), f"Task {output['task']} not in expected tasks: {all_tasks}"
    return output


def _read_sparsified_harness_result(result_path: Path) -> Dict:
    """
    Read the results of evaluation from lm-eval-harness run with sparsification.
    """
    with result_path.open("r") as f:
        results = json.load(f)
        rule = results["rule"]
        th_val = results["th_val"]
        sparsity = results["sparsity_stats"]["total"]

    parent_file = result_path.parent
    eval_scores = list(parent_file.glob("*/*.json"))
    assert (
        len(eval_scores) == 1
    ), f"Expected one eval score file, found {len(eval_scores)}"

    with eval_scores[0].open("r") as f:
        eval_results = json.load(f)["results"]

    results = _extract_metrics(eval_results)
    results["sparsification_rule"] = rule
    results["threshold"] = th_val
    results["sparsity"] = sparsity * 100

    return results


def read_sparsified_harness_results(
    sparsity_stat_files: List[Path], add_average: bool = True
) -> pd.DataFrame:
    sparsity_results = []
    for s in sparsity_stat_files:
        try:
            sparsity_results.append(_read_sparsified_harness_result(s))
        except Exception as e:
            print(f"Error processing {s}: {e}")
    df = pd.DataFrame(sparsity_results)

    if add_average:
        # Add "average" task that averages sparsity and acc across all tasks for given rule and threshold
        avg_df = (
            df.groupby(["sparsification_rule", "threshold"])
            .agg(sparsity=("sparsity", "mean"), acc=("acc", "mean"))
            .reset_index()
        )
        avg_df["task"] = "average"
        df = pd.concat([df, avg_df], ignore_index=True)

        avg_likelihood = (
            df[df["task"].isin(likelihood_tasks)]
            .groupby(["sparsification_rule", "threshold"])
            .agg(sparsity=("sparsity", "mean"), acc=("acc", "mean"))
            .reset_index()
        )
        avg_likelihood["task"] = "average_likelihood"
        avg_generative = (
            df[df["task"].isin(generative_tasks)]
            .groupby(["sparsification_rule", "threshold"])
            .agg(sparsity=("sparsity", "mean"), acc=("acc", "mean"))
            .reset_index()
        )
        avg_generative["task"] = "average_generative"
        df = pd.concat([df, avg_likelihood, avg_generative], ignore_index=True)

    df = df.sort_values(by=["task", "sparsification_rule", "threshold"])
    return df


def _read_standard_harness_result(
    result_path: Path, path_parser: Callable[[Path], Dict]
) -> Dict:
    with result_path.open("r") as f:
        results = json.load(f)["results"]
    results = _extract_metrics(results)
    result_data = path_parser(result_path)
    results.update(result_data)
    return results


def read_standard_harness_results(
    sparsity_stat_files: List[Path],
    path_parser: Callable[[Path], Dict],
    group_metrics: List[str],
    add_average: bool = True,
) -> pd.DataFrame:
    sparsity_results = []
    for s in sparsity_stat_files:
        try:
            sparsity_results.append(_read_standard_harness_result(s, path_parser))
        except Exception as e:
            print(f"Error processing {s}: {e}")
    df = pd.DataFrame(sparsity_results)

    if add_average:
        # Add "average" task that averages sparsity and acc across all tasks for given rule and threshold
        avg_df = df.groupby(group_metrics).agg(acc=("acc", "mean")).reset_index()
        avg_df["task"] = "average"
        df = pd.concat([df, avg_df], ignore_index=True)

        avg_likelihood = (
            df[df["task"].isin(likelihood_tasks)]
            .groupby(group_metrics)
            .agg(acc=("acc", "mean"))
            .reset_index()
        )
        avg_likelihood["task"] = "average_likelihood"
        avg_generative = (
            df[df["task"].isin(generative_tasks)]
            .groupby(group_metrics)
            .agg(acc=("acc", "mean"))
            .reset_index()
        )
        avg_generative["task"] = "average_generative"
        df = pd.concat([df, avg_likelihood, avg_generative], ignore_index=True)

    df = df.sort_values(by=["task"] + group_metrics)
    return df
