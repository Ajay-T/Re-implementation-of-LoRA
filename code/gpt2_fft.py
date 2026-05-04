import io
import json
import math
import os
import time
import urllib.request
from collections import defaultdict

import evaluate
import nltk
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import corpus_nist
from pycocoevalcap.cider.cider import Cider
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

MODEL_NAME = "gpt2-medium"
MAX_SEQ_LEN = 512
TARGET_BATCH_SIZE = 8
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = TARGET_BATCH_SIZE // PER_DEVICE_BATCH_SIZE
LR = 2e-4
EPOCHS = 10
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
MAX_GRAD_NORM = 1.0
MAX_NEW_TOKENS = 80
NUM_BEAMS = 10
LENGTH_PENALTY = 0.9
NO_REPEAT_NGRAM = 4
USE_FP16 = torch.cuda.is_available()
NUM_WORKERS = 2
PATIENCE = 3  # stop after 3 epochs without improvement
early_stop_counter = 0

assert TARGET_BATCH_SIZE % PER_DEVICE_BATCH_SIZE == 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def causal_lm_loss(logits, labels, label_smoothing=0.0):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.size(-1)

    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1)
    valid_mask = flat_labels.ne(-100)

    flat_labels = flat_labels.masked_fill(~valid_mask, 0)
    log_probs = F.log_softmax(flat_logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=flat_labels.unsqueeze(-1)).squeeze(-1)

    if label_smoothing > 0.0:
        smooth = -log_probs.mean(dim=-1)
        per_token_loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    else:
        per_token_loss = nll

    return per_token_loss[valid_mask].mean()

from huggingface_hub import snapshot_download

from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(
    f"{MODEL_NAME}: {total_params/1e6:.1f}M total | "
    f"{trainable_params/1e6:.1f}M trainable ({trainable_params/total_params*100:.1f}%)"
)
print(
    f"Per-device batch size: {PER_DEVICE_BATCH_SIZE} | "
    f"Gradient accumulation: {GRAD_ACCUM_STEPS} | "
    f"Effective batch size: {PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS}"
)

def _fetch_e2e_csv(url):
    with urllib.request.urlopen(url) as response:
        df = pd.read_csv(io.StringIO(response.read().decode("utf-8")))
    return df.rename(columns={"mr": "meaning_representation", "ref": "human_reference"})


BASE = "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/"
splits = {
    "train": _fetch_e2e_csv(BASE + "trainset.csv"),
    "validation": _fetch_e2e_csv(BASE + "devset.csv"),
    "test": _fetch_e2e_csv(BASE + "testset_w_refs.csv"),
}
raw_dataset = DatasetDict(
    {
        split: Dataset.from_pandas(
            df[["meaning_representation", "human_reference"]].reset_index(drop=True)
        )
        for split, df in splits.items()
    }
)
print(raw_dataset)
print(
    f"Train: {len(raw_dataset['train'])} | "
    f"Val: {len(raw_dataset['validation'])} | "
    f"Test: {len(raw_dataset['test'])}"
)


def linearize_mr(mr: str) -> str:
    parts = []
    for item in mr.split(","):
        key, val = item.strip().split("[", maxsplit=1)
        parts.append(f"{key} : {val.rstrip(']')}")
    return " | ".join(parts)


def build_prompt(mr: str) -> str:
    return "Input: " + linearize_mr(mr) + "\n" + "Output:"


def tokenize_example(example):
    prompt_text = build_prompt(example["meaning_representation"])
    target_text = " " + example["human_reference"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    input_ids = input_ids[:MAX_SEQ_LEN]
    labels = labels[:MAX_SEQ_LEN]
    pad_len = MAX_SEQ_LEN - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
    labels = labels + [-100] * pad_len
    attention_mask = [1] * len(input_ids[:-pad_len]) + [0] * pad_len


    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


tokenized = raw_dataset.map(tokenize_example, remove_columns=raw_dataset["train"].column_names)
tokenized.set_format("torch")
print(
    f"Train: {len(tokenized['train'])} | "
    f"Val: {len(tokenized['validation'])} | "
    f"Test: {len(tokenized['test'])}"
)

loader_kwargs = {
    "batch_size": PER_DEVICE_BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "pin_memory": torch.cuda.is_available(),
}
train_loader = DataLoader(tokenized["train"], shuffle=True, **loader_kwargs)
val_loader = DataLoader(tokenized["validation"], shuffle=False, **loader_kwargs)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
total_steps = steps_per_epoch * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps,
)
print(f"Optimizer steps per epoch: {steps_per_epoch}")
print(f"Total optimizer steps: {total_steps} | Warmup steps: {WARMUP_STEPS}")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

scaler = torch.amp.GradScaler("cuda", enabled=USE_FP16)
epoch_times = []
train_losses = []
val_losses = []
best_val_loss = float("inf")
best_epoch = 0
best_model_state = None
train_start = time.time()

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    epoch_loss_sum = 0.0
    epoch_start = time.time()

    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=USE_FP16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = causal_lm_loss(outputs.logits, labels, label_smoothing=LABEL_SMOOTHING)
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        running_loss += loss.item() * GRAD_ACCUM_STEPS
        epoch_loss_sum += loss.item() * GRAD_ACCUM_STEPS

        should_step = step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= scale_before:
                scheduler.step()

        if step % (200 * GRAD_ACCUM_STEPS) == 0 or step == len(train_loader):
            print(
                f"Epoch {epoch + 1} step {step}/{len(train_loader)} "
                f"| train loss {running_loss / step:.4f}"
            )

    train_loss = epoch_loss_sum / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=USE_FP16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                val_loss = causal_lm_loss(outputs.logits, labels, label_smoothing=0.0)

            val_loss_sum += val_loss.item()

    avg_val_loss = val_loss_sum / len(val_loader)
    val_losses.append(avg_val_loss)
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        best_model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        early_stop_counter = 0  # reset patience
    else:
        early_stop_counter += 1

    print(
        f"Epoch {epoch + 1} | train {train_loss:.4f} | val {avg_val_loss:.4f} "
        f"| best val {best_val_loss:.4f} | time {epoch_time / 60:.1f} min"
    )
    if early_stop_counter >= PATIENCE:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

total_train_time = time.time() - train_start
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Restored best checkpoint from epoch {best_epoch}.")

model.config.use_cache = True
peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
print(f"Total training time: {total_train_time / 60:.1f} min")
print(f"Peak VRAM: {peak_vram_mb:.1f} MB")

# save_dir = "/kaggle/working/gpt2_small_ft_e2e"
# os.makedirs(save_dir, exist_ok=True)
# model.save_pretrained(save_dir)
# tokenizer.save_pretrained(save_dir)

mr_to_refs = defaultdict(list)
for ex in raw_dataset["test"]:
    mr_to_refs[ex["meaning_representation"]].append(ex["human_reference"])

model.eval()
predictions, references = [], []
with torch.no_grad():
    for idx, (mr, refs) in enumerate(mr_to_refs.items(), start=1):
        prompt = build_prompt(mr)
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            pad_token_id=tokenizer.eos_token_id,
        )

        gen_ids = output_ids[0][input_ids.shape[1]:]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        predictions.append(pred)
        references.append(refs)

        if idx % 100 == 0:
            print(f"Generated {idx}/{len(mr_to_refs)} examples")

print(f"Generated {len(predictions)} predictions for {len(predictions)} unique MRs")
print("Example prediction:", predictions[0])
print("References:", references[0])

sacrebleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

max_refs = max(len(refs) for refs in references)
padded_references = [refs + [refs[-1]] * (max_refs - len(refs)) for refs in references]
bleu = sacrebleu.compute(predictions=predictions, references=padded_references)["score"]
rouge_l = rouge.compute(
    predictions=predictions,
    references=[list(refs) for refs in references],
    use_aggregator=True,
)["rougeL"] * 100
meteor = 100 * sum(
    meteor_score([ref.split() for ref in refs], pred.split())
    for pred, refs in zip(predictions, references)
) / len(predictions)

tok_preds = [pred.split() for pred in predictions]
tok_refs = [[ref.split() for ref in refs] for refs in references]
nist = corpus_nist(tok_refs, tok_preds, n=5)

gts = {idx: refs for idx, refs in enumerate(references)}
res = {idx: [pred] for idx, pred in enumerate(predictions)}
cider, _ = Cider().compute_score(gts, res)

print("\n" + "=" * 50)
print(f"{'Metric':<12} {'This run':>10}  {'Paper (FT)':>10}")
print("-" * 50)
print(f"{'BLEU':<12} {bleu:>10.1f}  {'68.2':>10}")
print(f"{'NIST':<12} {nist:>10.2f}  {'8.62':>10}")
print(f"{'METEOR':<12} {meteor:>10.1f}  {'46.2':>10}")
print(f"{'ROUGE-L':<12} {rouge_l:>10.1f}  {'71.0':>10}")
print(f"{'CIDEr':<12} {cider:>10.2f}  {'2.47':>10}")
print("=" * 50)