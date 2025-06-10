from abc import abstractmethod
from typing import List

import torch


class ReLUfiedActication(torch.nn.Module):
    def __init__(self, max_steps: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_ctr = 0
        self.max_steps = max_steps

    def increment_step_ctr(self):
        self.step_ctr += 1

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class HardReLU(ReLUfiedActication):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(x)


class SoftLinearReLU(ReLUfiedActication):
    def __init__(self, org_act_cls: torch.nn.Module, max_steps: int, *args, **kwargs):
        super().__init__(max_steps, *args, **kwargs)
        self.org_act_cls = org_act_cls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return linar combination of the original activation and ReLU based on the number of steps,
        # going progressively from original activation to ReLU
        alpha = (self.step_ctr + 1) / (self.max_steps)
        alpha = max(min(alpha, 1.0), 0.0)
        return alpha * torch.nn.functional.relu(x) + (1 - alpha) * self.org_act_cls(x)


class SoftCosineReLU(ReLUfiedActication):
    def __init__(self, org_act_cls: torch.nn.Module, max_steps: int, *args, **kwargs):
        super().__init__(max_steps, *args, **kwargs)
        self.org_act_cls = org_act_cls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return cosine combination of the original activation and ReLU based on the number of steps,
        # going progressively from original activation to ReLU
        progress = max(min((self.step_ctr + 1) / (self.max_steps), 1.0), 0.0)
        alpha = torch.cos(0.5 * torch.pi * torch.tensor(progress))
        # Make alpha between 0 and 1
        alpha = torch.clamp(alpha, 0.0, 1.0)
        return (1 - alpha) * torch.nn.functional.relu(x) + alpha * self.org_act_cls(x)


class SoftInverseCosineReLU(ReLUfiedActication):
    def __init__(self, org_act_cls: torch.nn.Module, max_steps: int, *args, **kwargs):
        super().__init__(max_steps, *args, **kwargs)
        self.org_act_cls = org_act_cls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return cosine combination of the original activation and ReLU based on the number of steps,
        # going progressively from original activation to ReLU
        progress = max(min((self.step_ctr + 1) / (self.max_steps), 1.0), 0.0)
        alpha = torch.cos(0.5 * torch.pi * torch.tensor(1 - progress))
        # Make alpha between 0 and 1
        alpha = torch.clamp(alpha, 0.0, 1.0)
        return alpha * torch.nn.functional.relu(x) + (1 - alpha) * self.org_act_cls(x)


def set_module_by_name(
    model: torch.nn.Module, module_name: str, new_module: torch.nn.Module
):
    """
    Replaces a module in the model with a new module using the module name.

    Args:
        model (torch.nn.Module): The parent model.
        module_name (str): The name of the module to replace (e.g., "layer1.block1.relu").
        new_module (torch.nn.Module): The new module to set.
    """
    # Split the module name into parts (e.g., "layer1.block1.relu" -> ["layer1", "block1", "relu"])
    module_parts = module_name.split(".")

    # Traverse the hierarchy to get the parent module
    parent_module = model
    for part in module_parts[:-1]:  # Exclude the last part
        parent_module = getattr(parent_module, part)

    # Replace the target module
    setattr(parent_module, module_parts[-1], new_module)


def apply_relufication(
    model: torch.nn.Module,
    modules_to_relufy: List[str],
    relufication_mode: str = "hard",
    max_steps: int = -1,
) -> List[ReLUfiedActication]:
    relufied_activation_handles = []
    for module_name in modules_to_relufy:
        module_name_and_module = [
            (name, module)
            for name, module in model.named_modules()
            if name.endswith(module_name)
        ]
        for org_module_name, org_module in module_name_and_module:
            new_activation: ReLUfiedActication
            if relufication_mode == "hard":
                new_activation = HardReLU(max_steps=max_steps)
            elif relufication_mode == "soft":
                new_activation = SoftLinearReLU(org_module, max_steps=max_steps)
            elif relufication_mode == "soft_cosine":
                new_activation = SoftCosineReLU(org_module, max_steps=max_steps)
            elif relufication_mode == "soft_inverse_cosine":
                new_activation = SoftInverseCosineReLU(org_module, max_steps=max_steps)
            else:
                raise ValueError(f"Invalid relufication mode: {relufication_mode}")
            set_module_by_name(model, org_module_name, new_activation)
            relufied_activation_handles.append(new_activation)
    return relufied_activation_handles
