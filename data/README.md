# Data Directory

This `data/` directory contains the dataset utilities used for training and evaluation in this repository.

## What is included

- `glue.py`: helper code to download, tokenize, and prepare GLUE task datasets with the Hugging Face `datasets` library.

## Dataset source

The project uses GLUE benchmarks via `datasets.load_dataset("glue", task_name)`.

Supported tasks in this implementation include:

- `mnli`
- `sst2`
- `mrpc`
- `cola`
- `qnli`
- `qqp`
- `rte`
- `stsb`

## How to obtain the dataset

The dataset is automatically downloaded from Hugging Face when the code runs and `load_glue_dataset` is called.

If you need to reproduce the environment, run the training code in a Colab session or local environment. The dataset is fetched dynamically and does not need to be checked in.

## Colab training note

The models for this project were pulled and trained during a Colab session. This README documents the data preparation process and confirms that the dataset is obtained automatically from Hugging Face when the code runs.
