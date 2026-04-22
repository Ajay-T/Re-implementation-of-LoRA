import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a low-rank adapter.

    Forward: h = W_0 @ x + (alpha/r) * B @ A @ x
    Only lora_A and lora_B are trainable; the base weight stays frozen.
    """

    def __init__(self, base_linear: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_linear = base_linear
        self.r = r
        self.scaling = alpha / r

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # A projects input down to rank r; B projects back up to output dim
        # B is zeros so delta_W = B @ A = 0 at init — training starts from pretrained weights
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Freeze the original pretrained weights
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        if self.merged:
            return base_out
        # Low-rank path: x -> dropout -> A -> B, scaled by alpha/r
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling

    def merge(self):
        """Fold LoRA weights into the base weight for zero-overhead inference."""
        if not self.merged:
            self.base_linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def unmerge(self):
        """Reverse merge to resume training or switch adapters."""
        if self.merged:
            self.base_linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
