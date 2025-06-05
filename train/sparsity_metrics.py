from abc import ABC, abstractmethod
from typing import List

import torch

LOSSES = ["l1", "l2", "hoyer", "none"]
SPARSIFICATION_MODES = ["input", "output"]


class SparsityMetric(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, tensor: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        pass


class MaxActivationValue(SparsityMetric):
    def __init__(self):
        super().__init__("max_activation_value")

    @torch.inference_mode()
    def __call__(self, tensor: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Tensor: (batch_size, seq_len, hidden_size)
        # attn_mask: (batch_size, seq_len)
        max_activation_values = tensor.abs().max(dim=-1)[0]
        # (batch_size, seq_len)
        token_level_max = max_activation_values.masked_fill(attn_mask == 0, 0)
        sequence_level_max = token_level_max.sum(dim=-1) / attn_mask.sum(dim=-1)
        # Set value as the mean of the max activation vals across the sequence
        return sequence_level_max


class MeanActivationValue(SparsityMetric):
    def __init__(self):
        super().__init__("mean_activation_value")

    @torch.inference_mode()
    def __call__(self, tensor: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Tensor: (batch_size, seq_len, hidden_size)
        # attn_mask: (batch_size, seq_len)
        mean_activation_values = tensor.abs().mean(dim=-1)
        # (batch_size, seq_len)
        token_level_mean = mean_activation_values.masked_fill(attn_mask == 0, 0)
        sequence_level_mean = token_level_mean.sum(dim=-1) / attn_mask.sum(dim=-1)
        # Set value as the mean of the mean activation vals across the sequence
        return sequence_level_mean


class PPercMaxSparsity(SparsityMetric):
    """Measures how many activations are under certaing percentage of the max activation"""

    def __init__(self, p: float):
        super().__init__(f"{p*100}_perc_max_sparsity")
        self.p = p

    @torch.inference_mode()
    def __call__(self, tensor: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Tensor: (batch_size, seq_len, hidden_size)
        # attn_mask: (batch_size, seq_len)
        hidden_dim = tensor.shape[-1]
        abs_tensor = tensor.abs()
        max_activation_values = abs_tensor.max(dim=-1)[0]  # (batch_size, seq_len)
        p_val = self.p * max_activation_values
        # (batch_size, seq_len)
        sparsity_mask = abs_tensor < p_val.unsqueeze(-1)
        token_level_sparsity = (sparsity_mask.sum(dim=-1) * attn_mask) / hidden_dim
        sequence_level_sparsity = (
            100 * token_level_sparsity.sum(dim=-1) / attn_mask.sum(dim=-1)
        )
        # Set value as the mean of the sparsity vals across the sequence
        return sequence_level_sparsity


def instantiate_sparsity_metrics(metric_names: List[str]) -> List[SparsityMetric]:
    metrics: List[SparsityMetric] = []
    for metric_name in metric_names:
        if metric_name == "max":
            metrics.append(MaxActivationValue())
        elif metric_name == "mean":
            metrics.append(MeanActivationValue())
        elif metric_name.endswith("%max"):
            p = float(metric_name.split("%")[0]) / 100
            metrics.append(PPercMaxSparsity(p))
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")
    return metrics
