"""Compare risk thresholds and label coverage using training students."""

import argparse
import csv
import json
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from statistics import mean, median


def training_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        yield from (row for row in csv.DictReader(handle) if row["split"] == "train")


def analyze(run_dir):
    config = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["config"]
    history = config["history"]
    students = {
        row["student_id"]: row for row in training_rows(run_dir / "students.csv")
    }
    risk = {
        row["student_id"]: row for row in training_rows(run_dir / "risk_examples.csv")
    }
    groups = defaultdict(list)
    prefix_accuracy = defaultdict(list)
    for student in students.values():
        groups[student["risk_status"]].append(int(student["n_interactions"]))

    for student_id, rows in groupby(
        training_rows(run_dir / "interactions.csv"), key=lambda row: row["student_id"]
    ):
        status = students[student_id]["risk_status"]
        if status == "insufficient_history":
            continue
        count = correct = 0
        for _, round_rows in groupby(rows, key=lambda row: row["solving_id"]):
            for row in round_rows:
                count += 1
                correct += int(row["is_correct"])
            if count >= history:
                accuracy = correct / count
                if status == "observed":
                    row = risk[student_id]
                    if count != int(row["history_end"]) or accuracy != float(row["history_accuracy"]):
                        raise ValueError(f"History cutoff mismatch for {student_id}")
                prefix_accuracy[status].append(accuracy)
                break

    if not risk:
        raise ValueError("No labeled risk examples in the training split")
    thresholds = []
    for threshold in (0.3, 0.4, 0.5):
        positives = sum(float(row["future_accuracy"]) < threshold for row in risk.values())
        thresholds.append({
            "threshold": threshold,
            "positive_students": positives,
            "labeled_students": len(risk),
            "positive_rate": positives / len(risk),
        })

    return {
        "config": config,
        "split": "train",
        "students": len(students),
        "thresholds": thresholds,
        "coverage": {
            status: {
                "students": len(lengths),
                "share": len(lengths) / len(students),
                "median_total_responses": median(lengths),
                "mean_history_accuracy_at_cutoff": (
                    mean(prefix_accuracy[status]) if prefix_accuracy[status] else None
                ),
            }
            for status, lengths in sorted(groups.items())
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("work/ednet"))
    args = parser.parse_args()
    print(json.dumps(analyze(args.input), indent=2))
