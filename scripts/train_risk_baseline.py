"""Evaluate two early learning-risk baselines on the prepared EdNet splits."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = (
    "history_accuracy", "recent_5_accuracy", "recent_10_accuracy",
    "mean_elapsed_ms", "history_end",
)
SPLITS = ("train", "val", "test_id")


def budget_curve(labels, scores):
    """Expected cases reached at each budget under uniform tie-breaking."""
    order = np.argsort(-scores, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
    sizes = np.diff(np.r_[starts, len(labels)])
    positive_rates = np.add.reduceat(labels[order], starts) / sizes
    return np.r_[0., np.cumsum(np.repeat(positive_rates, sizes))]


def evaluate(path, budget_curves=None):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len({row["student_id"] for row in rows}) != len(rows):
        raise ValueError("Expected one risk example per student across all splits")
    data = {}
    for split in SPLITS:
        subset = [row for row in rows if row["split"] == split]
        features = np.array([[float(row[key]) for key in FEATURES] for row in subset])
        labels = np.array([int(row["risk_label"]) for row in subset])
        if set(labels) != {0, 1}:
            raise ValueError(f"{split} must contain both risk labels")
        data[split] = features, labels

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(*data["train"])
    scorers = {
        "one_minus_history_accuracy": lambda features: 1 - features[:, 0],
        "logistic_regression": lambda features: model.predict_proba(features)[:, 1],
    }
    results = []
    for name, score in scorers.items():
        val_features, val_labels = data["val"]
        val_scores = score(val_features)
        # Choose the alert threshold on validation data only.
        threshold = max(
            np.unique(val_scores),
            key=lambda value: (f1_score(val_labels, val_scores >= value), value),
        )
        for split in ("val", "test_id"):
            features, labels = data[split]
            scores = score(features)
            if budget_curves is not None and split == "test_id":
                budget_curves[name] = budget_curve(labels, scores)
            alerts = scores >= threshold
            results.append({
                "model": name,
                "split": split,
                "students": len(labels),
                "positives": int(labels.sum()),
                "alert_threshold": float(threshold),
                "roc_auc": roc_auc_score(labels, scores),
                "average_precision": average_precision_score(labels, scores),
                "brier_score": (
                    brier_score_loss(labels, scores) if name == "logistic_regression" else None
                ),
                "precision": precision_score(labels, alerts, zero_division=0),
                "recall": recall_score(labels, alerts, zero_division=0),
                "f1": f1_score(labels, alerts, zero_division=0),
                "macro_f1": f1_score(labels, alerts, average="macro", zero_division=0),
            })
    return {
        "scikit_learn": sklearn.__version__,
        "features": list(FEATURES),
        "train_students": len(data["train"][1]),
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("work/ednet"))
    parser.add_argument("--budget-plot", type=Path, help="Save the test-set budget comparison")
    args = parser.parse_args()
    curves = {} if args.budget_plot else None
    report = evaluate(args.input / "risk_examples.csv", curves)
    if curves is not None:
        from matplotlib import pyplot as plt

        test = next(row for row in report["results"] if row["split"] == "test_id")
        n, positives = test["students"], test["positives"]
        budgets = [k for k in (10, 25, 50, 100) if k <= n]
        fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
        ax.plot([0, n], [0, positives], "--", color="0.5", label="Random review")
        names = ("Historical error rate", "Logistic regression")
        report["test_budget"] = {}
        for (name, curve), label in zip(curves.items(), names):
            ax.plot(curve, label=label)
            report["test_budget"][name] = [
                {"slots": k, "expected_cases": float(curve[k]),
                 "precision": float(curve[k] / k), "recall": float(curve[k] / positives)}
                for k in budgets
            ]
        ax.set(xlabel="Students reviewed", ylabel="Low-accuracy cases reached",
               xlim=(0, n), ylim=(0, positives + 1))
        ax.grid(alpha=0.2)
        ax.legend()
        args.budget_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.budget_plot, dpi=160)
        plt.close(fig)
    summary = json.loads((args.input / "summary.json").read_text(encoding="utf-8"))
    report["data_config"] = summary["config"]
    print(json.dumps(report, indent=2))
