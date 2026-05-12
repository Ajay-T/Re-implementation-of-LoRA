# Data Directory

This `data/` directory contains the dataset utilities used for training and evaluation in this repository.

## What is included

- `glue.py`: helper code to download, tokenize, and prepare GLUE task datasets with the Hugging Face `datasets` library.
- `vision.py`: helper code to download and prepare vision datasets (CIFAR-10, CIFAR-100) via `torchvision.datasets`.

## Dataset sources

### NLP (GLUE)

The project uses GLUE benchmarks via `datasets.load_dataset("glue", task_name)`.

Supported tasks in this implementation include:

- `sst2`
- `mnli`
- `mrpc`
- `cola`
- `qnli`
- `qqp`
- `rte`
- `stsb`

### Vision (CIFAR)

The ViT experiments use CIFAR-10 (and optionally CIFAR-100) via `torchvision.datasets`. Supported datasets:

- `cifar10`
- `cifar100`

## How to obtain the datasets

All datasets are automatically downloaded when the training code runs. No manual download is required.

- GLUE datasets are fetched from Hugging Face when `load_glue_dataset` is called.
- CIFAR datasets are fetched from torchvision when `load_vision_dataset` is called. They are cached locally in a `data_cache/` directory.

## Colab training note

The models for this project were trained in Google Colab sessions. The datasets are obtained automatically and do not need to be checked in.
