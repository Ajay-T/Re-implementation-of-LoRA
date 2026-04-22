import time
import torch
import numpy as np
from torch.cuda.amp import GradScaler
from transformers import get_linear_schedule_with_warmup
from .metrics import compute_metrics


class Trainer:
    """Handles the training loop, evaluation, and early stopping.

    Works for both full fine-tuning and LoRA — the only difference is which
    parameters have requires_grad=True, which is handled before this class.
    """

    def __init__(self, model, train_loader, val_loader, task_name, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_name = task_name
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.is_regression = task_name == "stsb"

        # Only optimize parameters that are unfrozen
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 0.1),
        )

        # Linear warmup then linear decay, following the paper's setup
        total_steps = len(train_loader) * config["epochs"]
        warmup_steps = int(total_steps * config.get("warmup_ratio", 0.06))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        self.scaler = GradScaler() if config.get("fp16", False) else None
        self.history = {"train_loss": [], "val_metrics": [], "epoch_times": []}

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        all_preds, all_labels = [], []

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            logits = outputs.logits

            if self.is_regression:
                preds = logits.squeeze(-1).cpu().numpy()
            else:
                preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(batch["labels"].cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        return compute_metrics(self.task_name, all_preds, all_labels)

    def train(self):
        best_metric = -float("inf")
        best_state = None
        patience = self.config.get("patience", 5)
        patience_counter = 0

        for epoch in range(self.config["epochs"]):
            start = time.time()
            train_loss = self.train_epoch()
            elapsed = time.time() - start

            val_metrics = self.evaluate()

            self.history["train_loss"].append(train_loss)
            self.history["val_metrics"].append(val_metrics)
            self.history["epoch_times"].append(elapsed)

            # Use the first metric in the dict as the primary for early stopping
            primary = list(val_metrics.values())[0]
            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Val: {val_metrics} | "
                  f"Time: {elapsed:.1f}s")

            if primary > best_metric:
                best_metric = primary
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best checkpoint
        if best_state:
            self.model.load_state_dict(best_state)

        return self.history
