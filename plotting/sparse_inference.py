import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import itertools

sns.set_style("whitegrid")

if __name__ == "__main__":
    root = Path(__file__).parent.parent / "sparse_inference_results"

    outputs = []
    task_dirs = [d for d in sorted(list(root.glob("*"))) if d.is_dir()]
    for task_dir in task_dirs:
        task_name = task_dir.stem
        sparsity_subdirs = [d for d in sorted(list(task_dir.glob("*"))) if d.is_dir()]

        for sparsity_subdir in sparsity_subdirs:
            results_path = list(sparsity_subdir.rglob("results*.json"))[0]
            sparsity_stats_path = sparsity_subdir / "sparsity_stats.json"

            with results_path.open("r") as f:
                results = json.load(f)['results']
            with sparsity_stats_path.open("r") as f:
                sparsity_stats = json.load(f)

            subtask_names = results.keys()
            if 'mmlu_pro' in subtask_names:
                subtask_names = ['mmlu_pro']

            sparsity = sparsity_stats['sparsity_stats']['total'] * 100
            topp = sparsity_stats['topp']

            for subtask_name in subtask_names: 
                if 'exact_match,remove_whitespace' in results[subtask_name]:
                    accuracy = results[subtask_name]['exact_match,remove_whitespace']
                elif "exact_match,strict-match" in results[subtask_name]:
                    accuracy = results[subtask_name]["exact_match,strict-match"]
                elif 'acc,none' in results[subtask_name]:
                    accuracy = results[subtask_name]['acc,none']
                elif "prompt_level_loose_acc,none" in results[subtask_name]:
                    accuracy = results[subtask_name]["prompt_level_loose_acc,none"]
                elif 'exact_match,custom-extract' in results[subtask_name]:
                    accuracy = results[subtask_name]['exact_match,custom-extract']
                else:
                    raise ValueError(f"No accuracy key found in results for task {task_name}. Possible keys: {results[subtask_name].keys()}")
                
                outputs.append({
                    "task": subtask_name,
                    "sparsity": sparsity,
                    "topp": topp,
                    "accuracy": accuracy
                })           

    df = pd.DataFrame(outputs)
    df = df.sort_values(by=["task","sparsity"], ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot = sns.lineplot(data=df, x="sparsity", y="accuracy", hue="task", markers=True, ax=axes[0], legend=False, style="task")  
    plot.set(xlabel="Sparsity", ylabel="Performance")
    plot.set_title("Llama3.2-1B - absolute performance numbers")

    # Normalize accuracy for each task with the accuracy for the minimal sparsity
    df['accuracy_normalized'] = df.groupby('task')['accuracy'].transform(lambda x: x / x.max())
    plot = sns.lineplot(data=df, x="sparsity", y="accuracy_normalized", hue="task", markers=True, ax=axes[1], legend=True, style="task")
    plot.set(xlabel="Sparsity", ylabel="Normalized Performance")
    plot.set_title("Llama3.2-1B - normalized performance")
    
    plt.tight_layout()
    fig.savefig(root / "sparsity_per_task.png")