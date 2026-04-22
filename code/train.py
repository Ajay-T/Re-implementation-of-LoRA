"""Main entry point for training. Supports both full fine-tuning and LoRA modes.

Usage:
    python train.py --model_name roberta-base --task sst2 --mode lora --r 8
    python train.py --model_name roberta-base --task sst2 --mode full
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from lora import LoRAConfig, inject_lora, mark_only_lora_as_trainable, lora_state_dict
from data.glue import load_glue_dataset, get_dataloaders, TASK_TO_NUM_LABELS
from training.trainer import Trainer
from eval.evaluate import count_parameters, measure_gpu_memory, save_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--task", type=str, default="sst2", choices=list(TASK_TO_NUM_LABELS.keys()))
    parser.add_argument("--mode", type=str, default="lora", choices=["lora", "full"])

    # LoRA hyperparameters
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=["query", "value"])

    # Training hyperparameters (defaults follow the LoRA paper)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--output_dir", type=str, default="../results")
    return parser.parse_args()


def main():
    args = parse_args()
    num_labels = TASK_TO_NUM_LABELS[args.task]

    print(f"Loading {args.model_name} for {args.task} ({args.mode} mode)")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    if args.mode == "lora":
        lora_config = LoRAConfig(
            r=args.r,
            alpha=args.alpha,
            dropout=args.lora_dropout,
            target_modules=args.target_modules,
        )
        # Replace target linear layers with LoRA-wrapped versions
        model = inject_lora(model, lora_config)
        # Freeze everything except lora_A and lora_B
        mark_only_lora_as_trainable(model, bias="none")

        # The classification head must remain trainable (it's randomly initialized)
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True

    param_info = count_parameters(model)
    print(f"Parameters — Total: {param_info['total']:,} | Trainable: {param_info['trainable']:,} "
          f"({param_info['trainable']/param_info['total']*100:.2f}%)")

    dataset = load_glue_dataset(args.task, tokenizer, args.max_length)
    train_loader, val_loader = get_dataloaders(dataset, args.batch_size)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # LoRA uses higher LR than full fine-tuning (paper sweeps 1e-4 to 5e-4 for LoRA, ~2e-5 for full FT)
    training_config = {
        "lr": args.lr if args.mode == "lora" else 2e-5,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "patience": args.patience,
        "fp16": args.fp16,
    }

    trainer = Trainer(model, train_loader, val_loader, args.task, training_config)
    history = trainer.train()

    final_metrics = trainer.evaluate()
    print(f"\nFinal metrics: {final_metrics}")

    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0

    results = {
        "model": args.model_name,
        "task": args.task,
        "mode": args.mode,
        "params": param_info,
        "val_metrics": final_metrics,
        "peak_memory_mb": peak_mem,
        "epoch_times": history["epoch_times"],
        "train_losses": history["train_loss"],
    }

    if args.mode == "lora":
        results["lora_config"] = {"r": args.r, "alpha": args.alpha, "dropout": args.lora_dropout,
                                  "target_modules": args.target_modules}
        # Save only the LoRA weights (tiny compared to the full model)
        torch.save(lora_state_dict(model), f"{args.output_dir}/lora_weights_{args.task}.pt")

    save_results(results, args.output_dir, f"{args.mode}_{args.task}_{args.model_name.replace('/', '_')}.json")


if __name__ == "__main__":
    main()
