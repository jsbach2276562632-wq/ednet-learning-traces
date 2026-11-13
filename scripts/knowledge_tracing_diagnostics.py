"""Student bootstrap intervals and reliability plots for next-response scores."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def prepare_auc(labels, scores, student_index):
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    groups = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    labels = labels[order]
    return groups, student_index[order], labels


def weighted_auc(prepared, weights):
    groups, student_index, labels = prepared
    row_weights = weights[student_index]
    positives = np.add.reduceat(row_weights * labels, groups)
    negatives = np.add.reduceat(row_weights * (1 - labels), groups)
    denominator = positives.sum() * negatives.sum()
    if not denominator:
        return float("nan")
    return float(positives @ (np.cumsum(negatives) - 0.5 * negatives) / denominator)


def calibration(labels, scores):
    bins = np.minimum((scores * 10).astype(int), 9)
    return [
        {"bin": index, "targets": int(mask.sum()), "predicted": float(scores[mask].mean()),
         "observed": float(labels[mask].mean())}
        for index in range(10) if (mask := bins == index).any()
    ]


def write_diagnostics(arrays, students, predictions, output, replicates=1000, seed=2026):
    splits = ("test_id", "test_ood")
    student_ids = np.unique(students["test_id"])
    if not np.array_equal(student_ids, np.unique(students["test_ood"])):
        raise ValueError("Expected matched students across test splits")
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(student_ids), np.full(len(student_ids), 1 / len(student_ids)),
                            size=replicates)
    original = np.ones(len(student_ids), dtype=int)
    report = {"students": len(student_ids), "replicates": replicates, "seed": seed, "models": {}}
    bootstrap = {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True)
    for split, axis in zip(splits, axes):
        labels = arrays[split][:, 3]
        student_index = np.searchsorted(student_ids, students[split])
        axis.plot([0, 1], [0, 1], color="0.55", linestyle="--", linewidth=1)
        for name, by_split in predictions.items():
            scores = np.mean(by_split[split], axis=0)
            auc_inputs = prepare_auc(labels, scores, student_index)
            auc = weighted_auc(auc_inputs, original)
            samples = np.array([weighted_auc(auc_inputs, draw) for draw in draws])
            bootstrap[name, split] = auc, samples
            bins = calibration(labels, scores)
            report["models"].setdefault(name, {})[split] = {
                "auc": auc, "auc_ci": np.nanquantile(samples, [0.025, 0.975]).tolist(),
                "brier": float(np.mean((scores - labels) ** 2)), "bins": bins,
            }
            line, = axis.plot([b["predicted"] for b in bins], [b["observed"] for b in bins],
                              label=name, alpha=0.85)
            axis.scatter([b["predicted"] for b in bins], [b["observed"] for b in bins],
                         s=[16 + 100 * b["targets"] / len(labels) for b in bins],
                         color=line.get_color())
        axis.set(title="Seen bundles" if split == "test_id" else "Unseen bundles",
                 xlabel="Mean predicted probability", xlim=(0, 1), ylim=(0, 1), aspect="equal")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Observed fraction correct")
    axes[1].legend(loc="upper left", fontsize=8)
    for name, model in report["models"].items():
        seen_auc, seen = bootstrap[name, "test_id"]
        unseen_auc, unseen = bootstrap[name, "test_ood"]
        model["seen_minus_unseen"] = {
            "auc_difference": seen_auc - unseen_auc,
            "ci": np.nanquantile(seen - unseen, [0.025, 0.975]).tolist(),
        }
    output.mkdir(parents=True)
    fig.tight_layout()
    fig.savefig(output / "calibration.svg")
    plt.close(fig)
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
