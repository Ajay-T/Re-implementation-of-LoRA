import urllib.request
import io
import pandas as pd
from datasets import Dataset, DatasetDict


BASE_E2E_URL = "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/"


def _fetch_e2e_csv(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url) as r:
        df = pd.read_csv(io.StringIO(r.read().decode("utf-8")))

    return df.rename(columns={
        "mr": "meaning_representation",
        "ref": "human_reference"
    })


def load_e2e_dataset():
    """
    Loads the E2E dataset and returns a Hugging Face DatasetDict.
    """

    splits = {
        "train": _fetch_e2e_csv(BASE_E2E_URL + "trainset.csv"),
        "validation": _fetch_e2e_csv(BASE_E2E_URL + "devset.csv"),
        "test": _fetch_e2e_csv(BASE_E2E_URL + "testset_w_refs.csv"),
    }

    raw_dataset = DatasetDict({
        split_name: Dataset.from_pandas(
            df[["meaning_representation", "human_reference"]].reset_index(drop=True)
        )
        for split_name, df in splits.items()
    })

    print(raw_dataset)
    print(
        f"Train: {len(raw_dataset['train'])} | "
        f"Val: {len(raw_dataset['validation'])} | "
        f"Test: {len(raw_dataset['test'])}"
    )

    return raw_dataset