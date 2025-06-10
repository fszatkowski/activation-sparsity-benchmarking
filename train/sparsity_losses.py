from abc import ABC, abstractmethod
from typing import Optional

import torch

LOSSES = ["l1", "l2", "hoyer", "none"]
SPARSIFICATION_MODES = ["input", "output"]


class ActivationSparsityLoss(torch.nn.Module, ABC):
    def __init__(self, name: str, weight: float = 1.0, shift: Optional[float] = None):
        self.name = name
        self.weight = weight
        self.shift = shift

    @abstractmethod
    def __call__(self, tensor: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Should return loss per sequence in the batch, so the shape would be (batch_size,)
        pass


def _masked_l1_loss(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (mask * tensor.abs().sum(dim=-1)).sum(dim=-1) / mask.sum(dim=-1)


class L1Loss(ActivationSparsityLoss):
    def __init__(self, weight: float = 1.0, shift: Optional[float] = None):
        super().__init__("l1", weight, shift)

    def __call__(self, tensor: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Tensor: (batch_size, seq_len, hidden_size)
        # attn_mask: (batch_size, seq_len)
        if self.shift is not None:
            tensor = torch.clamp(tensor, min=self.shift) - self.shift

        return self.weight * _masked_l1_loss(tensor, attn_mask)


def _masked_l2_norm(
    tensor: torch.Tensor, mask: torch.Tensor, eps: float
) -> torch.Tensor:
    return (mask * torch.sqrt((tensor**2).sum(dim=-1) + eps)).sum(dim=-1) / mask.sum(
        dim=-1
    )


class L2Loss(ActivationSparsityLoss):
    def __init__(self, weight: float = 1.0, shift: Optional[float] = None):
        super().__init__("l2", weight, shift)

    def __call__(
        self, tensor: torch.Tensor, attn_mask: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        # Tensor: (batch_size, seq_len, hidden_size)
        # attn_mask: (batch_size, seq_len)
        if self.shift is not None:
            tensor = torch.clamp(tensor, min=self.shift) - self.shift

        return self.weight * _masked_l2_norm(tensor, attn_mask, eps)


def _masked_hoyer_loss(
    tensor: torch.Tensor, mask: torch.Tensor, eps: float
) -> torch.Tensor:
    return (
        mask * ((tensor.abs().sum(dim=-1) ** 2) / ((tensor**2).sum(dim=-1) + eps))
    ).sum(dim=-1) / mask.sum(dim=-1)


class HoyerLoss(ActivationSparsityLoss):
    def __init__(self, weight: float = 1.0, shift: Optional[float] = None):
        super().__init__("hoyer", weight, shift)

    def __call__(
        self, tensor: torch.Tensor, attn_mask: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        # Tensor: (batch_size, seq_len, hidden_size)
        # attn_mask: (batch_size, seq_len)
        if self.shift is not None:
            tensor = torch.clamp(tensor, min=self.shift) - self.shift

        return self.weight * _masked_hoyer_loss(tensor, attn_mask, eps)


def get_sparsity_loss(
    loss_name: str, weight: float, shift: Optional[float] = None
) -> Optional[ActivationSparsityLoss]:
    loss_name = loss_name.lower()
    if loss_name == "l1":
        return L1Loss(weight, shift)
    elif loss_name == "l2":
        return L2Loss(weight, shift)
    elif loss_name == "hoyer":
        return HoyerLoss(weight, shift)
    elif loss_name == "none":
        return None
    else:
        raise ValueError(f"Unknown sparsity loss: {loss_name}")
