from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    # Rank of the low-rank decomposition (paper tests r = 1, 2, 4, 8, 64)
    r: int = 8
    # Scaling numerator — output is multiplied by alpha/r to keep magnitude stable across ranks
    alpha: float = 8.0
    # Dropout applied to input before the low-rank path
    dropout: float = 0.05
    # Which linear layers to replace with LoRA (matched by substring in module name)
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])

    @property
    def scaling(self) -> float:
        return self.alpha / self.r
