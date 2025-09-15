import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A linear layer with Low-Rank Adaptation (LoRA).

    The original weight is frozen and two low-rank matrices are learned
    to provide an additive update. This is the basic LoRA formulation and
    does not rely on any online-specific variants.
    """

    def __init__(self, linear: nn.Linear, r: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scaling = alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, linear.in_features))
            self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            # freeze original weights
            self.linear.weight.requires_grad = False
            if self.linear.bias is not None:
                self.linear.bias.requires_grad = False
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        if self.r > 0:
            lora = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
            result = result + lora * self.scaling
        return result


def add_lora_to_linear_layers(model: nn.Module, r: int = 4, alpha: float = 1.0, dropout: float = 0.0) -> nn.Module:
    """Recursively replace ``nn.Linear`` modules in ``model`` with ``LoRALinear``.

    Args:
        model: The module to modify in-place.
        r: Rank of the low-rank update matrices.
        alpha: Scaling factor for the LoRA updates.
        dropout: Dropout to apply to inputs before the LoRA layers.
    """

    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        else:
            add_lora_to_linear_layers(module, r=r, alpha=alpha, dropout=dropout)
    return model
