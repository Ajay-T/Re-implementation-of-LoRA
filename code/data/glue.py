from datasets import load_dataset
from torch.utils.data import DataLoader

# Maps each GLUE task to its input column names (single-sentence tasks use None for the second key)
TASK_TO_KEYS = {
    "mnli": ("premise", "hypothesis"),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "cola": ("sentence", None),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
}

# Number of output classes per task (stsb=1 because it's a regression task)
TASK_TO_NUM_LABELS = {
    "mnli": 3,
    "sst2": 2,
    "mrpc": 2,
    "cola": 2,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "stsb": 1,
}


def load_glue_dataset(task_name: str, tokenizer, max_length: int = 128):
    """Download a GLUE task from HuggingFace and tokenize it."""
    task_name_hf = "mnli" if task_name == "mnli" else task_name
    dataset = load_dataset("glue", task_name_hf)
    keys = TASK_TO_KEYS[task_name]

    def tokenize(examples):
        args = (examples[keys[0]],) if keys[1] is None else (examples[keys[0]], examples[keys[1]])
        return tokenizer(*args, padding="max_length", truncation=True, max_length=max_length)

    dataset = dataset.map(tokenize, batched=True)
    # HuggingFace models expect the column to be named "labels"
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return dataset


def get_dataloaders(dataset, batch_size: int = 16):
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)

    # MNLI uses "validation_matched" instead of "validation"
    val_key = "validation_matched" if "validation_matched" in dataset else "validation"
    val_loader = DataLoader(dataset[val_key], batch_size=batch_size)

    return train_loader, val_loader
