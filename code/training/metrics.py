import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score


def compute_metrics(task_name: str, preds: np.ndarray, labels: np.ndarray) -> dict:
    """Return the standard GLUE metric(s) for the given task.

    STS-B: Pearson/Spearman correlation (regression)
    CoLA: Matthews correlation coefficient
    MRPC/QQP: accuracy + F1
    Everything else: accuracy
    """
    if task_name == "stsb":
        return {
            "pearson": pearsonr(preds, labels)[0],
            "spearman": spearmanr(preds, labels)[0],
        }
    if task_name == "cola":
        return {"matthews_corr": matthews_corrcoef(labels, preds)}
    if task_name in ("mrpc", "qqp"):
        return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

    return {"accuracy": accuracy_score(labels, preds)}
