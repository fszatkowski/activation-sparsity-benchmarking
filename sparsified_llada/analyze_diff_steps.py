#!/usr/bin/env python3


"""The “Token×Neuron frequency” section builds a frequency matrix over tokens and neuron indices for selected steps
and saves heatmaps of log10(frequency+1) per layer, per interval (e.g. layerX_freq_interval1.png), but never converts
that into a sparsity fraction per step.

stability_vs_interval computes per‑token and global Jaccard between masks at step t t and t + Δ t+Δ for various
intervals, then you plot a heatmap of mean per‑token Jaccard over (layer_num, step) and one time‑series for a mid
layer; this is “mask stability”, not sparsity.

The remaining parts compute averaged‑mask quality and top‑k frequency overlap with the final mask and save CSVs and
one more time‑series plot, again using Jaccard and overlap metrics, not fraction of active units per diffusion step.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import os
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ========================
# CONFIGURATION (EDIT HERE)
# ========================

# Path to the directory from the SCRIPT_DIR variable from the llada_*.sh file that runs the slurm jobs.
# SCRIPT_DIR = "${WORKDIR}/${JOB_NAME}"
DIR = r""

# filename of the *rank0_mask_snapshots.npz file with an extension created in the DIR by the run
npz_file = os.path.join(DIR, "*_rank0_mask_snapshots.npz")

# filename of the *_rank0_sparsity_per_step.json file with an extension created in the DIR by the run
sparsity_per_step_file = os.path.join(DIR, "*_rank0_sparsity_per_step.json")

# filename of the *_rank0_sparsity_per_step.csv file with an extension created in the DIR by the run
sparsity_csv = os.path.join(DIR, r"*_rank0_sparsity_per_step.csv")

# ========================
# Load sparsity dataframe
# ========================
df_sparsity = pd.read_csv(sparsity_csv)

# If layer_num is not in the CSV, add it
if 'layer_num' not in df_sparsity.columns:
    df_sparsity['layer_num'] = df_sparsity['layer'].apply(layer_num_from_name)

print("Columns:", df_sparsity.columns.tolist())
print(df_sparsity.head())

PLOT_DENSITY = False  # if True, plots 100 - mean_sparsity

output_dir = Path(DIR) / "analysis_results"
intervals = [32]
interval_sizes = [32]

# ========================
# END CONFIGURATION
# ========================

out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)


def layer_num(layer_name: str) -> int:
    """
    Extract numeric transformer block index from layer_name.
    Falls back to large number to push unparsed names to the end.
    """
    m = re.search(r"(?:blocks|layers)\.(\d+)", layer_name)
    return int(m.group(1)) if m else 10**9


def load_sparsity_json(path: Path) -> dict:
    print(f"Loading sparsity per step from: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def sparsity_json_to_df(data: dict) -> pd.DataFrame:
    """
    Flatten the 'layers' -> 'step' structure into a DataFrame:
    columns: layer, layer_num, step, mean_sparsity, count, total_zeros, total_elements
    """
    rows = []
    layers_dict = data.get("layers", {})

    for layer_name, steps in layers_dict.items():
        ln = layer_num(layer_name)
        for step_str, s in steps.items():
            step = int(step_str)
            rows.append({
                "layer": layer_name,
                "layer_num": ln,
                "step": step,
                "mean_sparsity": float(s["mean_sparsity"]),
                "count": int(s["count"]),
                "total_zeros": int(s["total_zeros"]),
                "total_elements": int(s["total_elements"]),
            })

    df = pd.DataFrame(rows)
    df.sort_values(["layer_num", "step"], inplace=True)
    return df


# ---------------------------
# Load snapshots (indices/mags)
# ---------------------------
def load_snapshots(npz_path: str) -> Dict[Tuple[str, int], dict]:
    data = np.load(npz_path, allow_pickle=True)
    snaps = {}
    print(f"Loading snapshots from: {npz_path}")
    print(f"Total keys: {len(data.files)}")

    for key in data.files:
        if key.startswith('metadata_'):
            continue

        if key.endswith('_indices'):
            base = key[:-8]
            m = re.match(r'^(.*)_step_(\d+)$', base)
            if not m:
                print(f"  [warn] Could not parse key: {key}")
                continue
            layer, step = m.group(1), int(m.group(2))
            snaps.setdefault((layer, step), {})['indices'] = list(data[key])

        elif key.endswith('_magnitudes'):
            base = key[:-11]
            m = re.match(r'^(.*)_step_(\d+)$', base)
            if not m:
                print(f"  [warn] Could not parse key: {key}")
                continue
            layer, step = m.group(1), int(m.group(2))
            snaps.setdefault((layer, step), {})['magnitudes'] = list(data[key])

    print(f"\nLoaded {len(snaps)} snapshots")
    if snaps:
        for k in list(snaps.keys())[:3]:
            print(f"  sample: {k}")
    return snaps


def layer_num(layer_name: str) -> int:
    m = re.search(r'(?:blocks|layers)\.(\d+)', layer_name)
    return int(m.group(1)) if m else 10**9


def global_hidden_size_for_layer(snaps: dict, layer: str) -> int:
    gmax = -1
    for (lay, step), v in snaps.items():
        if lay != layer:
            continue
        for arr in v['indices']:
            if isinstance(arr, np.ndarray) and arr.size:
                gmax = max(gmax, int(arr.max()))
    return gmax + 1 if gmax >= 0 else 4096


# ---------------------------
# Token×Neuron frequency plots
# ---------------------------
def build_token_neuron_frequency(snaps: dict, layer: str, interval: int = 0) -> np.ndarray:
    all_steps = sorted([s for (lay, s) in snaps if lay == layer])
    if not all_steps:
        return None
    steps = [all_steps[0]] if interval == 0 else [s for s in all_steps if s % interval == 0]
    if not steps:
        return None
    H = global_hidden_size_for_layer(snaps, layer)
    freq_dict = defaultdict(lambda: np.zeros(H, dtype=np.int64))
    for step in steps:
        indices_list = snaps[(layer, step)]['indices']
        for token_idx, arr in enumerate(indices_list):
            if not isinstance(arr, np.ndarray) or arr.size == 0:
                continue
            valid = arr[(arr >= 0) & (arr < H)]
            if valid.size == 0:
                continue
            freq_dict[token_idx][valid] += 1
    if not freq_dict:
        return None
    T = max(freq_dict.keys()) + 1
    freq_mat = np.zeros((T, H), dtype=np.int64)
    for t, row in freq_dict.items():
        freq_mat[t, :] = row
    return freq_mat


def save_frequency_heatmap(mat: np.ndarray, title: str, ylabel: str, out_path: Path):
    if mat is None or mat.size == 0:
        print(f"[warn] Empty matrix for {title}, skipping")
        return
    fig, ax = plt.subplots(figsize=(16, 10))
    mat_vis = np.log10(mat + 1)
    im = ax.imshow(mat_vis, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Neuron Index', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, label='log10(Frequency + 1)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------
# Stability computations
# ---------------------------
def jaccard_per_token(A_list, B_list):
    n = min(len(A_list), len(B_list))
    outv = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = A_list[i]; b = B_list[i]
        a_set = set(a.tolist() if hasattr(a, 'tolist') else a)
        b_set = set(b.tolist() if hasattr(b, 'tolist') else b)
        if not a_set and not b_set:
            outv[i] = 1.0
        else:
            inter = len(a_set & b_set)
            uni = len(a_set | b_set)
            outv[i] = inter / uni if uni > 0 else 0.0
    return outv


def jaccard_global(A_list, B_list):
    A = set(); B = set()
    for a in A_list:
        A.update(a.tolist() if hasattr(a, 'tolist') else a)
    for b in B_list:
        B.update(b.tolist() if hasattr(b, 'tolist') else b)
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))


def stability_vs_interval(snaps, intervals=(4,8,16,32)):
    rows = []
    steps_by_layer = defaultdict(list)
    for (lay, s) in snaps.keys():
        steps_by_layer[lay].append(s)
    for lay in steps_by_layer:
        steps_by_layer[lay] = sorted(set(steps_by_layer[lay]))
    for lay, steps in steps_by_layer.items():
        for Δ in intervals:
            for t in steps:
                t2 = t + Δ
                if (lay, t) not in snaps or (lay, t2) not in snaps:
                    continue
                A = snaps[(lay, t)]['indices']
                B = snaps[(lay, t2)]['indices']
                Jtok = jaccard_per_token(A, B)
                Jg = jaccard_global(A, B)
                rows.append({
                    'layer': lay,
                    'layer_num': layer_num(lay),
                    'interval': Δ,
                    't': t,
                    'mean_tokenJ': float(np.mean(Jtok)),
                    'median_tokenJ': float(np.median(Jtok)),
                    'globalJ': float(Jg),
                    'count_tokens': int(len(Jtok)),
                })
    return pd.DataFrame(rows)


def averaged_mask(A_lists, method='union'):
    if method == 'union':
        S = set()
        for L in A_lists:
            for arr in L:
                S.update(arr.tolist() if hasattr(arr, 'tolist') else arr)
        return S
    elif method == 'intersection':
        init = None
        for L in A_lists:
            step_set = set()
            for arr in L:
                step_set.update(arr.tolist() if hasattr(arr, 'tolist') else arr)
            init = step_set if init is None else (init & step_set)
        return init if init is not None else set()
    else:
        raise ValueError("method must be 'union' or 'intersection'")


def averaged_mask_quality(snaps, layer, interval_steps, method='union'):
    sets = []
    for s in interval_steps:
        if (layer, s) in snaps:
            sets.append(snaps[(layer, s)]['indices'])
    if not sets:
        return pd.DataFrame()
    avg_mask = averaged_mask(sets, method=method)
    rows = []
    for s in interval_steps:
        if (layer, s) not in snaps:
            continue
        L = snaps[(layer, s)]['indices']
        jtok = []
        # tokenwise Jaccard to avg
        for arr in L:
            a_set = set(arr.tolist() if hasattr(arr, 'tolist') else arr)
            if not a_set and not avg_mask:
                jtok.append(1.0)
            else:
                inter = len(a_set & avg_mask)
                uni = len(a_set | avg_mask)
                jtok.append(inter / uni if uni > 0 else 0.0)
        # global Jaccard to avg
        cur = set()
        for a in L:
            cur.update(a.tolist() if hasattr(a, 'tolist') else a)
        if not cur and not avg_mask:
            Jg = 1.0
        else:
            Jg = len(cur & avg_mask) / max(1, len(cur | avg_mask))
        rows.append({'step': s, 'mean_tokenJ_to_avg': float(np.mean(jtok)), 'globalJ_to_avg': float(Jg)})
    return pd.DataFrame(rows)


def precision_topk_frequency_vs_final(snaps, layer, k_ratio=0.2):
    steps = sorted([s for (lay,s) in snaps if lay==layer])
    if not steps:
        return None
    last = steps[-1]
    H = 0
    for s in steps:
        for arr in snaps[(layer,s)]['indices']:
            if isinstance(arr, np.ndarray) and arr.size:
                H = max(H, int(arr.max())+1)
    if H == 0:
        return None
    freq = np.zeros(H, dtype=np.int64)
    for s in steps:
        for arr in snaps[(layer,s)]['indices']:
            if isinstance(arr, np.ndarray) and arr.size:
                v = arr[(arr>=0)&(arr<H)]
                freq[v] += 1
    k = max(1, int(k_ratio * H))
    topk = np.argpartition(freq, -k)[-k:]
    final_set = set()
    for arr in snaps[(layer,last)]['indices']:
        final_set.update(arr.tolist() if hasattr(arr,'tolist') else arr)
    return float(len(set(topk) & final_set) / max(1, len(set(topk))))

# ---------------------------
# RUN
# ---------------------------
snaps = load_snapshots(npz_file)

# 1) Token×Neuron frequency plots per layer
layers = sorted({k[0] for k in snaps.keys()}, key=layer_num)
print(f"\nFound {len(layers)} layers")
for layer in layers:
    lname_short = f"layer{layer_num(layer)}"
    for interval in intervals:
        mat = build_token_neuron_frequency(snaps, layer, interval=interval)
        if mat is None:
            print(f"[warn] No data for {layer} interval={interval}")
            continue
        title = f"{lname_short}: Token×Neuron Frequency (Step 0 only)" if interval==0 \
                else f"{lname_short}: Token×Neuron Frequency (Steps: 0, {interval}, {2*interval}, ...)"
        fname = f"{lname_short}_freq_step0.png" if interval==0 else f"{lname_short}_freq_interval{interval}.png"
        save_frequency_heatmap(mat, title, "Token Position", out / fname)

# 2) Stability vs interval
stab = stability_vs_interval(snaps, intervals=tuple(interval_sizes))
stab.to_csv(out / "stability_vs_interval.csv", index=False)

# Heatmap for Δ=32 (or closest present)
target_delta = 32 if 32 in interval_sizes else interval_sizes[-1]
stab_delta = stab[stab["interval"] == target_delta]

# if stab_delta may be empty, guard it:
if stab_delta.empty:
    print(f"No stability rows for interval Δ={target_delta}, skipping heatmap.")
else:
    pivot = stab_delta.pivot_table(
        index="layer_num",
        columns="t",
        values="mean_tokenJ",
        aggfunc="mean"  # or 'median', 'max', etc.
    )
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        pivot,
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        cbar_kws={"label": "Mean per-token Jaccard"}
    )
    plt.title(f"Stability Heatmap (Δ={target_delta})", fontweight="bold")
    plt.xlabel("Step")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(out / f"heatmap_interval{target_delta}.png", dpi=300)
    plt.close()


# 3) Averaged/frozen mask proxy
rows = []
steps_by_layer = defaultdict(list)
for (lay,s) in snaps.keys():
    steps_by_layer[lay].append(s)
for lay in layers:
    s_all = sorted(set(steps_by_layer[lay]))
    for Δ in interval_sizes:
        if not s_all or Δ <= 0:
            continue
        for start in range(0, max(s_all)+1, Δ):
            interval = [s for s in s_all if start <= s < start+Δ]
            if len(interval) < 2:
                continue
            q = averaged_mask_quality(snaps, lay, interval, method='union')
            if not q.empty:
                rows.append({
                    'layer': lay, 'layer_num': layer_num(lay),
                    'interval': Δ,
                    'mean_of_means': float(q['mean_tokenJ_to_avg'].mean()),
                    'n_steps': len(interval)
                })
avg_df = pd.DataFrame(rows)
avg_df.to_csv(out / "averaged_mask_quality.csv", index=False)

# 4) Frequency vs final overlap (+ optional t0 magnitude later)
freq_rows = []
for lay in layers:
    prec = precision_topk_frequency_vs_final(snaps, lay)
    freq_rows.append({'layer': lay, 'layer_num': layer_num(lay), 'precision_topk_vs_final': prec})
pd.DataFrame(freq_rows).to_csv(out / "neuron_frequency_correlation.csv", index=False)

# 5) Example time series (mid layer)
if not stab.empty and 'layer_num' in stab:
    mids = sorted(stab['layer_num'].unique())
    mid_layer_num = mids[len(mids)//2]
    one = stab[stab['layer_num']==mid_layer_num]
    plt.figure(figsize=(12,5))
    for Δ in interval_sizes:
        sub = one[one['interval']==Δ].sort_values('t')
        if not sub.empty:
            plt.plot(sub['t'], sub['mean_tokenJ'], label=f'Δ={Δ}')
    plt.xlabel('Step'); plt.ylabel('Mean per‑token Jaccard')
    plt.title(f'Layer {mid_layer_num}: stability vs interval'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out / f"layer{mid_layer_num}_stability_timeseries.png", dpi=300); plt.close()
    print(f"- {out/f'layer{mid_layer_num}_stability_timeseries.png'}")

print("\nSaved:")
print(f"- {out/'stability_vs_interval.csv'}")
print(f"- {out/f'heatmap_interval{target_delta}.png'}")
print(f"- {out/'averaged_mask_quality.csv'}")
print(f"- {out/'neuron_frequency_correlation.csv'}")

"""
Sparsity per diffusion step plots
"""

data = load_sparsity_json(sparsity_per_step_file)
df = sparsity_json_to_df(data)

# Optionally convert to density (1 - sparsity)
if PLOT_DENSITY:
    df["mean_density"] = 100.0 - df["mean_sparsity"]  # percent of active units
    value_col = "mean_density"
    value_label = "Mean Density (%)"
    base_fname = "density"
else:
    value_col = "mean_sparsity"
    value_label = "Mean Sparsity (%)"
    base_fname = "sparsity"

# 1) Global sparsity / density vs diffusion step (token-weighted)
#    Use total_zeros / total_elements across layers per step.
grouped = df.groupby("step", as_index=False).agg(
    sum_zeros=("total_zeros", "sum"),
    sum_elems=("total_elements", "sum")
)
print("after grouping")
grouped[value_col] = np.where(
    grouped["sum_elems"] > 0,
    (grouped["sum_zeros"] / grouped["sum_elems"]) * 100.0 if not PLOT_DENSITY
    else 100.0 - (grouped["sum_zeros"] / grouped["sum_elems"]) * 100.0,
    0.0
)
print("after np where")

plt.figure(figsize=(10, 5))
plt.plot(grouped["step"], grouped[value_col], label="global", linewidth=2)
plt.xlabel("Diffusion step")
plt.ylabel(value_label)
plt.title(f"Global {value_label} vs diffusion step")
plt.grid(True, alpha=0.3)
plt.tight_layout()
out_path = output_dir / f"global_{base_fname}_timeseries.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved global timeseries: {out_path}")

# 2) Heatmap: layer_num × step of mean sparsity/density
pivot = df.pivot_table(
    index="layer_num",
    columns="step",
    values=value_col,
    aggfunc="mean"
)
plt.figure(figsize=(14, 8))
sns.heatmap(
    pivot,
    cmap="viridis" if not PLOT_DENSITY else "magma",
    cbar_kws={"label": value_label}
)
plt.xlabel("Diffusion step")
plt.ylabel("Layer (block index)")
plt.title(f"{value_label} heatmap (layer × step)")
plt.tight_layout()
out_path = output_dir / f"{base_fname}_heatmap_layer_step.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved heatmap: {out_path}")

# 3) Example per-layer sparsity/density time series (mid layer)
unique_layers = sorted(df["layer_num"].unique())
if unique_layers:
    mid_layer_num = unique_layers[len(unique_layers) // 2]
    sub = df[df["layer_num"] == mid_layer_num].sort_values("step")

    plt.figure(figsize=(10, 5))
    plt.plot(sub["step"], sub[value_col], label=f"layer {mid_layer_num}", linewidth=2)
    plt.xlabel("Diffusion step")
    plt.ylabel(value_label)
    plt.title(f"Layer {mid_layer_num}: {value_label} vs diffusion step")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / f"layer{mid_layer_num}_{base_fname}_timeseries.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved mid-layer timeseries: {out_path}")

# 4) Optional: average sparsity/density across steps per layer
layer_stats = df.groupby("layer_num", as_index=False).agg(
    avg_value=(value_col, "mean"),
    min_value=(value_col, "min"),
    max_value=(value_col, "max")
)
csv_out = output_dir / f"{base_fname}_summary_per_layer.csv"
layer_stats.to_csv(csv_out, index=False)
print(f"Saved per-layer summary CSV: {csv_out}")


# ========================
# 1) Global mean sparsity per step (averaged over layers)
# ========================
global_df = (
    df_sparsity.groupby('step', as_index=False)['mean_sparsity_pct']
      .mean()
      .rename(columns={'mean_sparsity_pct': 'global_mean_sparsity_pct'})
)

plt.figure(figsize=(10, 5))
plt.plot(global_df['step'], global_df['global_mean_sparsity_pct'], marker='o', linewidth=1.5)
plt.xlabel('Diffusion step')
plt.ylabel('Global mean sparsity (%)')
plt.title('Global mean sparsity vs diffusion step')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outdir / "global_mean_sparsity_vs_step.png", dpi=300)
plt.close()

# ========================
# 2) Per-layer mean sparsity curves vs step
# ========================
plt.figure(figsize=(12, 6))

# Choose whether to plot all layers or a subset
layers = sorted(df_sparsity['layer_num'].unique())
# e.g., plot all:
for ln in layers:
    sub = df_sparsity[df_sparsity['layer_num'] == ln].sort_values('step')
    plt.plot(sub['step'], sub['mean_sparsity_pct'], alpha=0.7, label=f"L{ln}")

plt.xlabel('Diffusion step')
plt.ylabel('Mean sparsity (%)')
plt.title('Per-layer mean sparsity vs diffusion step')
plt.grid(True, alpha=0.3)

# If too many layers, make legend outside
plt.legend(ncol=4, fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(outdir / "per_layer_mean_sparsity_vs_step.png", dpi=300, bbox_inches='tight')
plt.close()

print("Saved:")
print(f"- {outdir/'global_mean_sparsity_vs_step.png'}")
print(f"- {outdir/'per_layer_mean_sparsity_vs_step.png'}")
