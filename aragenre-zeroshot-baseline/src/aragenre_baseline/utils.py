import json

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalise(x):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def cosine_matrix(a, b):
    return normalise(a) @ normalise(b).T


def evaluate(title, gold, pred):
    print(f"\n=== {title} ===")
    print("Accuracy:", round(accuracy_score(gold, pred), 4))
    print("Macro F1:", round(f1_score(gold, pred, average="macro", zero_division=0), 4))
    print(classification_report(gold, pred, digits=4, zero_division=0))


def hierarchical_macro_f1(gold_broad, pred_broad, gold_specific, pred_specific):
    """Official AraGenre ranking metric: mean of Broad and Specific Macro F1."""
    broad_f1 = f1_score(gold_broad, pred_broad, average="macro", zero_division=0)
    specific_f1 = f1_score(gold_specific, pred_specific, average="macro", zero_division=0)
    hier_f1 = (broad_f1 + specific_f1) / 2.0

    print("\n=== Official Ranking Metric ===")
    print(f"Broad Macro F1:        {round(broad_f1, 4)}")
    print(f"Specific Macro F1:     {round(specific_f1, 4)}")
    print(f"Hierarchical Macro F1: {round(hier_f1, 4)}  <- official score")
    return hier_f1
