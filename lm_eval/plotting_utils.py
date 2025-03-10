from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import torch


sns.set_style("whitegrid")


def plot_heatmap(x_ticks, y_ticks, data, save_path):
    plt.cla()
    plt.clf()
    plt.figure()

    plot = sns.heatmap(data, xticklabels=x_ticks, yticklabels=y_ticks, cmap="viridis", cbar=False, annot=True, fmt=".1f")
    # Rotate ticks
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    parent_dir = Path(save_path).parent  
    parent_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plot.get_figure().savefig(save_path)
    torch.save(data, parent_dir / "src_data.pt")
