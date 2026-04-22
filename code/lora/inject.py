import torch.nn as nn
from .config import LoRAConfig
from .layers import LoRALinear


def inject_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Walk the model and replace matching nn.Linear layers with LoRALinear.

    Matches by substring: e.g. target_modules=["query", "value"] will match
    modules named "...attention.self.query", "...attention.self.value", etc.
    """
    for name, module in model.named_modules():
        for target in config.target_modules:
            if target not in name:
                continue
            if not isinstance(module, nn.Linear):
                continue

            lora_layer = LoRALinear(
                base_linear=module,
                r=config.r,
                alpha=config.alpha,
                dropout=config.dropout,
            )

            # Navigate to the parent module and replace the target layer in-place
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], lora_layer)

    return model


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none"):
    """Freeze everything, then selectively unfreeze LoRA params (and optionally biases).

    bias="none": only lora_A/lora_B are trainable
    bias="all": all biases in the model are also trainable
    bias="lora_only": only biases in LoRA-injected layers are trainable
    """
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        if bias == "all" and "bias" in name:
            param.requires_grad = True
        elif bias == "lora_only":
            if "bias" in name and any(
                lora_name in name.replace("bias", "")
                for lora_name in ["lora_A", "lora_B"]
            ):
                param.requires_grad = True


def lora_state_dict(model: nn.Module) -> dict:
    """Extract only the LoRA parameters for lightweight saving."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}
