# Data Directory

This data/ directory contains dataset utilities used for training and evaluation in this repository. It supports both GLUE benchmark tasks (classification/regression) and the E2E dataset (data-to-text generation).


## What is included

- `glue.py`: helper code to download, tokenize, and prepare GLUE task datasets with the Hugging Face `datasets` library.
- `e2e.py`: helper code to download and prepare E2E dataset for text generation from Github
- `vision.py`: helper code to download and prepare vision datasets (CIFAR-10, CIFAR-100) via `torchvision.datasets`.

## Dataset sources

## GLUE Benchmarks (Classification)

The project uses GLUE benchmarks from the HuggingFace Datasets library for classification via `datasets.load_dataset("glue", task_name)`.

Supported tasks in this implementation include:

- `sst2`
- `mnli`
- `mrpc`
- `cola`
- `qnli`
- `qqp`
- `rte`
- `stsb`

## E2E Dataset (Natural Language Generation)
We used the E2E dataset for our NLG tasks. You can find the dataset at this repository: https://github.com/tuetschek/e2e-dataset

In our implementation, dataset is automatically downloaded at runtime from raw CSV files hosted on GitHub:
- Train: https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/trainset.csv
- Validation: https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/devset.csv
- Test: https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/testset_w_refs.csv

We fetch it via `urllib.request.urlopen(url)`, parse it with Pandas, and convert it to a HuggingFace Dataset object. The dataset is combined into a DatasetDict with train/validation/test splits.

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
