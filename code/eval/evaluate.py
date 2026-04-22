import torch
import json
import os


def count_parameters(model):
    """Count total, trainable, and frozen parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def measure_gpu_memory():
    """Snapshot current GPU memory usage."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024 ** 2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024 ** 2,
    }


def save_results(results: dict, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


def compare_results(full_ft_results: dict, lora_results: dict):
    """Print a side-by-side comparison of full fine-tuning vs LoRA results."""
    print(f"\n{'='*60}")
    print("COMPARISON: Full Fine-Tuning vs LoRA")
    print(f"{'='*60}")

    print(f"\nTrainable params — Full FT: {full_ft_results['params']['trainable']:,} | "
          f"LoRA: {lora_results['params']['trainable']:,} | "
          f"Reduction: {(1 - lora_results['params']['trainable'] / full_ft_results['params']['trainable']) * 100:.1f}%")

    if full_ft_results.get("peak_memory_mb") and lora_results.get("peak_memory_mb"):
        print(f"Peak VRAM — Full FT: {full_ft_results['peak_memory_mb']:.1f} MB | "
              f"LoRA: {lora_results['peak_memory_mb']:.1f} MB")

    print("\nMetrics:")
    for key in full_ft_results.get("val_metrics", {}):
        ft_val = full_ft_results["val_metrics"][key]
        lora_val = lora_results["val_metrics"][key]
        print(f"  {key}: Full FT = {ft_val:.4f} | LoRA = {lora_val:.4f} | Δ = {lora_val - ft_val:+.4f}")
