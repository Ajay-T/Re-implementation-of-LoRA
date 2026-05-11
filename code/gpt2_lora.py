import json
import math
import os
import time
import types
from collections import defaultdict
import pandas as pd, io, urllib.request
from datasets import Dataset, DatasetDict

import evaluate
import nltk
import torch
import torch.nn as nn
from nltk.translate.nist_score import corpus_nist
from pycocoevalcap.cider.cider import Cider
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download


nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

MODEL_NAME = "gpt2-medium"
MAX_SEQ_LEN = 512
BATCH_SIZE = 8
LR = 0.0002
MAX_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 3
MIN_DELTA = 0.0
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1  # Hu et al. GPT-2 E2E (Table 11); dropout on low-rank path input (see LoRALinear)
MAX_NEW_TOKENS = 60
NUM_BEAMS = 10
NO_REPEAT_NGRAM = 4
REPETITION_PENALTY = 1.2
USE_FP16 = torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def inject_lora(model: GPT2LMHeadModel, rank: int = 4, alpha: float = 16.0):
    """Replace Q and V projection linears in every attention block with LoRALinear."""
    for block in model.transformer.h:
        attn = block.attn
        # GPT-2 fuses Q, K, V into c_attn (Conv1D, not Linear) — we wrap them manually
        # Instead, patch the individual projection via a hook approach, OR
        # rewrite using the split approach below:
        original = attn.c_attn  # Conv1D: weight shape is (in, 3*out)
        in_f  = original.weight.shape[0]          # 768

        # Inject LoRA only into Q and V slices (columns 0:768 and 1536:2304)
        attn.lora_A_q = nn.Parameter(torch.empty(rank, in_f))
        attn.lora_B_q = nn.Parameter(torch.zeros(in_f, rank))
        attn.lora_A_v = nn.Parameter(torch.empty(rank, in_f))
        attn.lora_B_v = nn.Parameter(torch.zeros(in_f, rank))
        attn.lora_rank  = rank
        attn.lora_scale = alpha / rank
        attn.lora_dropout = (
            nn.Dropout(LORA_DROPOUT) if LORA_DROPOUT > 0 else nn.Identity()
        )

        nn.init.kaiming_uniform_(attn.lora_A_q, a=math.sqrt(5))
        nn.init.kaiming_uniform_(attn.lora_A_v, a=math.sqrt(5))

    return model

def _fetch_e2e_csv(url):
    with urllib.request.urlopen(url) as r:
        df = pd.read_csv(io.StringIO(r.read().decode("utf-8")))
    return df.rename(columns={"mr": "meaning_representation", "ref": "human_reference"})

BASE = "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/"
splits = {
    "train":      _fetch_e2e_csv(BASE + "trainset.csv"),
    "validation": _fetch_e2e_csv(BASE + "devset.csv"),
    "test":       _fetch_e2e_csv(BASE + "testset_w_refs.csv"),
}
raw_dataset = DatasetDict({
    k: Dataset.from_pandas(df[["meaning_representation", "human_reference"]].reset_index(drop=True))
    for k, df in splits.items()
})
print(raw_dataset)
print(f"Train: {len(raw_dataset['train'])} | Val: {len(raw_dataset['validation'])} | Test: {len(raw_dataset['test'])}")
def linearize_mr(mr: str) -> str:
    parts = []
    for item in mr.split(","):
        key, val = item.strip().split("[")
        val = val.rstrip("]")
        parts.append(f"{key} : {val}")
    return " | ".join(parts)

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize_example(example):
    prompt_text = (
    "Input: " + linearize_mr(example["meaning_representation"]) + "\n"
    "Output:"
    )
    
    target_text = " " + example["human_reference"] + tokenizer.eos_token
    
    # Tokenize separately
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    
    # Combine
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids


    # Truncate
    input_ids = input_ids[:MAX_SEQ_LEN]
    labels = labels[:MAX_SEQ_LEN]

    # Pad
    pad_len = MAX_SEQ_LEN - len(input_ids)
    if tokenizer.padding_side == "left":
        input_ids = [tokenizer.pad_token_id] * pad_len + input_ids
        labels = [-100] * pad_len + labels
        attention_mask = [0] * pad_len + [1] * len(input_ids[pad_len:])
    else:
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
print(f"Train: {len(tokenized['train'])} | Val: {len(tokenized['validation'])} | Test: {len(tokenized['test'])}")

def make_lora_c_attn_forward(attn):
    c_attn = attn.c_attn
    if not hasattr(c_attn, "_original_forward"):
        c_attn._original_forward = c_attn.forward

    original_forward = c_attn._original_forward
    split_size = attn.split_size

    def lora_c_attn_forward(module, hidden_states):
        qkv = original_forward(hidden_states)

        h_lora = attn.lora_dropout(hidden_states)
        delta_q = (h_lora @ attn.lora_A_q.T) @ attn.lora_B_q.T * attn.lora_scale
        delta_v = (h_lora @ attn.lora_A_v.T) @ attn.lora_B_v.T * attn.lora_scale

        q, k, v = qkv.split(split_size, dim=2)
        return torch.cat([q + delta_q, k, v + delta_v], dim=2)

    return types.MethodType(lora_c_attn_forward, c_attn)


def patch_lora_forward(model):
    for block in model.transformer.h:
        block.attn.c_attn.forward = make_lora_c_attn_forward(block.attn)
    return model

def freeze_base_weights(model):
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")
    return model

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.config.pad_token_id = tokenizer.eos_token_id
model = inject_lora(model)
model = patch_lora_forward(model)
model = freeze_base_weights(model)
model.to(device)
# --- DataLoaders ---
loader_kwargs = {
    "batch_size": BATCH_SIZE,
    "num_workers": 2,
    "pin_memory": torch.cuda.is_available(),
}
train_loader = DataLoader(tokenized["train"], shuffle=True, **loader_kwargs)
val_loader   = DataLoader(tokenized["validation"], shuffle=False, **loader_kwargs)

# --- Optimizer & scheduler ---
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)
total_steps = len(train_loader) * MAX_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps,
)
print(f"Total training steps: {total_steps} | Warmup steps: {WARMUP_STEPS}")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

scaler = torch.amp.GradScaler("cuda", enabled=USE_FP16)
epoch_times = []
train_losses = []
val_losses = []
best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0
best_model_state = None
best_lora_state = None
stopped_early = False
train_start = time.time()

# --- Loop ---
for epoch in range(MAX_EPOCHS):
    model.train()
    total_loss = 0.0
    epoch_start = time.time()

    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=USE_FP16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            MAX_GRAD_NORM,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        if step % 200 == 0:
            print(
                f"Epoch {epoch+1} step {step}/{len(train_loader)} "
                f"| loss {total_loss/step:.4f}"
            )

    train_loss = total_loss / len(train_loader)
    model.eval()
    val_total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=USE_FP16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            val_total_loss += outputs.loss.item()

    val_loss = val_total_loss / len(val_loader)
    epoch_time = time.time() - epoch_start
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    epoch_times.append(epoch_time)

    if val_loss < (best_val_loss - MIN_DELTA):
        best_val_loss = val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        best_model_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        best_lora_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
            if "lora_" in name
        }
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch + 1} | train {train_loss:.4f} | val {val_loss:.4f} "
        f"| best val {best_val_loss:.4f} | time {epoch_time / 60:.1f} min"
    )

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        stopped_early = True
        print(
            f"Early stopping triggered after epoch {epoch + 1}. "
            f"Best epoch was {best_epoch} with val loss {best_val_loss:.4f}."
        )
        break

total_train_time = time.time() - train_start
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Restored best checkpoint from epoch {best_epoch}.")
peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
print(f"Total training time: {total_train_time/60:.1f} min")
print(f"Peak VRAM: {peak_vram_mb:.1f} MB")

# save_dir = "/kaggle/working/gpt2_lora_e2e"
# os.makedirs(save_dir, exist_ok=True)

# model.save_pretrained(save_dir)
# tokenizer.save_pretrained(save_dir)

lora_state_dict = best_lora_state if best_lora_state is not None else {
    name: tensor.detach().cpu()
    for name, tensor in model.state_dict().items()
    if "lora_" in name
}
full_state_dict = best_model_state if best_model_state is not None else {
    name: tensor.detach().cpu()
    for name, tensor in model.state_dict().items()
}
# torch.save(lora_state_dict, os.path.join(save_dir, "lora_adapters.pt"))
# torch.save(full_state_dict, os.path.join(save_dir, "full_state_dict.pt"))

mr_to_refs = defaultdict(list)
for ex in raw_dataset["test"]:
    mr_to_refs[ex["meaning_representation"]].append(ex["human_reference"])

model.eval()
predictions, references = [], []

with torch.no_grad():
    for i, (mr, refs) in enumerate(mr_to_refs.items(), start=1):
        prompt_text = (
            "Input: " + linearize_mr(mr) + "\n"
            "Output:"
        )
        encoded = tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )

        gen_ids = output_ids[0][input_ids.shape[1]:]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        predictions.append(pred)
        references.append(refs)

        if i % 100 == 0:
            print(f"Generated {i}/{len(mr_to_refs)} examples")

print(f"Generated {len(predictions)} predictions for {len(predictions)} unique MRs")
print("Example prediction:", predictions[0])
print("References:", references[0])

sacrebleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")

max_refs = max(len(refs) for refs in references)
padded_references = [refs + [refs[-1]] * (max_refs - len(refs)) for refs in references]

bleu = sacrebleu.compute(predictions=predictions, references=padded_references)["score"]
rouge_l = rouge.compute(
    predictions=predictions,
    references=[list(refs) for refs in references],
    use_aggregator=True,
)["rougeL"] * 100

meteor = meteor_metric.compute(
    predictions=predictions,
    references=list(references)
)["meteor"] * 100

tok_preds = [pred.split() for pred in predictions]
tok_refs = [[ref.split() for ref in refs] for refs in references]
nist = corpus_nist(tok_refs, tok_preds, n=5)

gts = {idx: refs for idx, refs in enumerate(references)}
res = {idx: [pred] for idx, pred in enumerate(predictions)}
cider, _ = Cider().compute_score(gts, res)

print("\n" + "=" * 50)
print(f"{'Metric':<12} {'LoRA run':>10}")
print("-" * 50)
print(f"{'BLEU':<12} {bleu:>10.1f}")
print(f"{'NIST':<12} {nist:>10.2f}")
print(f"{'METEOR':<12} {meteor:>10.1f}")
print(f"{'ROUGE-L':<12} {rouge_l:>10.1f}")
print(f"{'CIDEr':<12} {cider:>10.2f}")
print("=" * 50)