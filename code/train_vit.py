"""Entry point for ViT image classification with LoRA or full fine-tuning.

Usage:
    python train_vit.py --dataset cifar10 --mode lora --r 8 --fp16
    python train_vit.py --dataset cifar10 --mode full --fp16
"""

import argparse
import torch
from transformers import ViTForImageClassification

from lora import LoRAConfig, inject_lora, mark_only_lora_as_trainable, lora_state_dict
from data.vision import load_vision_dataset, get_vision_dataloaders, DATASET_NUM_LABELS
from training.vit_trainer import ViTTrainer
from eval.evaluate import count_parameters, save_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=list(DATASET_NUM_LABELS.keys()))
    parser.add_argument("--mode", type=str, default="lora", choices=["lora", "full"])

    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=["query", "value"])

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--output_dir", type=str, default="../results")
    return parser.parse_args()


def main():
    args = parse_args()
    num_labels = DATASET_NUM_LABELS[args.dataset]

    print(f"Loading {args.model_name} for {args.dataset} ({args.mode} mode)")
    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    if args.mode == "lora":
        lora_config = LoRAConfig(
            r=args.r,
            alpha=args.alpha,
            dropout=args.lora_dropout,
            target_modules=args.target_modules,
        )
        model = inject_lora(model, lora_config)
        mark_only_lora_as_trainable(model, bias="none")

        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True

    param_info = count_parameters(model)
    print(f"Parameters — Total: {param_info['total']:,} | Trainable: {param_info['trainable']:,} "
          f"({param_info['trainable']/param_info['total']*100:.2f}%)")

    train_set, val_set = load_vision_dataset(args.dataset)
    train_loader, val_loader = get_vision_dataloaders(train_set, val_set, args.batch_size)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    training_config = {
        "lr": args.lr if args.mode == "lora" else 2e-5,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "patience": args.patience,
        "fp16": args.fp16,
    }

    trainer = ViTTrainer(model, train_loader, val_loader, training_config)
    history = trainer.train()

    final_metrics = trainer.evaluate()
    print(f"\nFinal metrics: {final_metrics}")

    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0

    results = {
        "model": args.model_name,
        "dataset": args.dataset,
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
        torch.save(lora_state_dict(model), f"{args.output_dir}/lora_weights_vit_{args.dataset}.pt")

    save_results(results, args.output_dir, f"{args.mode}_{args.dataset}_{args.model_name.replace('/', '_')}.json")


if __name__ == "__main__":
    main()
