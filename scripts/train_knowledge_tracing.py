"""Compare recurrent and attentive models on the same EdNet response targets."""

import argparse
import copy
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn


def held_out(bundle, seed):
    digest = hashlib.sha256(f"{seed}:bundle:{bundle}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 5 == 0


def read_interactions(path, split_seed):
    vocab = {}
    interactions = []
    # Index 0 is padding, 1 is an unfamiliar tag/part pair.
    question_features = []
    examples = defaultdict(list)
    counts = Counter()
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        # Build the vocabulary before reading validation or test examples.
        for row in rows:
            if row["split"] == "train" and not held_out(row["bundle_id"], split_seed):
                key = feature_key(row)
                if key not in vocab:
                    vocab[key] = len(vocab) + 2
        handle.seek(0)
        for student, student_rows in groupby(csv.DictReader(handle), lambda row: row["student_id"]):
            start = len(interactions)
            for _, round_rows in groupby(student_rows, lambda row: row["solving_id"]):
                items = list(round_rows)
                cutoff = len(interactions)
                for row in items:
                    split = row["split"]
                    unseen = held_out(row["bundle_id"], split_seed)
                    if unseen and split != "test_id":
                        continue
                    skill = vocab.get(feature_key(row), 1)
                    if cutoff == start:
                        counts[f"{split}_no_history"] += 1
                        continue
                    if unseen and skill == 1:
                        counts["ood_unknown_feature"] += 1
                    name = "test_ood" if unseen else split
                    examples[name].append((start, cutoff, skill, int(row["is_correct"]), student))
                # Held-out responses never enter any model history.
                for row in items:
                    if not held_out(row["bundle_id"], split_seed):
                        question_features.append(vocab.get(feature_key(row), 1))
                        interactions.append(int(row["is_correct"]))
    # Compare ID and OOD on the same students.
    common = {r[4] for r in examples["test_id"]} & {r[4] for r in examples["test_ood"]}
    for split in ("test_id", "test_ood"):
        counts[f"{split}_before_matching"] = len(examples[split])
        examples[split] = [r for r in examples[split] if r[4] in common]
    counts["paired_test_students"] = len(common)
    if not all(examples[name] for name in ("train", "val", "test_id", "test_ood")):
        raise ValueError("Each split needs targets with prior history")
    students = {name: np.array([r[4] for r in rows]) for name, rows in examples.items()}
    arrays = {name: np.array([r[:4] for r in rows], dtype=np.int64) for name, rows in examples.items()}
    features = np.array(question_features, dtype=np.int64)
    responses = np.array(interactions, dtype=np.int64)
    return arrays, students, features * 2 + responses, len(vocab) + 2, dict(counts)


def feature_key(row):
    # The models share the first listed tag and TOEIC part as question inputs.
    return row["part"], row["tags"].split(";", 1)[0]


class DKT(nn.Module):
    def __init__(self, skills, width):
        super().__init__()
        self.answers = nn.Embedding(2 * skills, width, padding_idx=0)
        self.questions = nn.Embedding(skills, width)
        self.lstm = nn.LSTM(width, width, batch_first=True)
        self.output = nn.Linear(width * 2, 1)

    def forward(self, history, question, lengths):
        sequence, _ = self.lstm(self.answers(history))
        last = sequence[torch.arange(len(history), device=history.device), lengths - 1]
        return self.output(torch.cat((last, self.questions(question)), dim=-1)).squeeze(-1)


class SAKT(nn.Module):
    def __init__(self, skills, width, window):
        super().__init__()
        self.answers = nn.Embedding(2 * skills, width, padding_idx=0)
        self.questions = nn.Embedding(skills, width)
        self.positions = nn.Embedding(window, width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width))
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 1)

    def forward(self, history, question, lengths):
        positions = torch.arange(history.shape[1], device=history.device)
        # Index positions by recency so the newest answer always has position zero.
        recency = (lengths[:, None] - 1 - positions).clamp_min(0)
        memory = self.answers(history) + self.positions(recency)
        query = self.questions(question).unsqueeze(1)
        attended, _ = self.attention(query, memory, memory, key_padding_mask=history.eq(0), need_weights=False)
        hidden = self.norm(query + attended)
        return self.output(self.final_norm(hidden + self.feed_forward(hidden))).flatten()


def batches(rows, tokens, window, batch_size, device, order=None):
    if order is None:
        order = np.arange(len(rows))
    for first in range(0, len(order), batch_size):
        start, cutoff, skills, labels = rows[order[first:first + batch_size]].T
        left = np.maximum(start, cutoff - window)
        lengths = cutoff - left
        index = left[:, None] + np.arange(lengths.max())
        valid = index < cutoff[:, None]
        history = np.where(valid, tokens[np.where(valid, index, 0)], 0)
        yield tuple(torch.as_tensor(array, device=device) for array in (history, skills, lengths, labels))


def metrics(labels, scores):
    return {
        "targets": len(labels),
        "accuracy": float(np.mean((scores >= 0.5) == labels)),
        "roc_auc": roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else None,
        "average_precision": average_precision_score(labels, scores) if len(np.unique(labels)) == 2 else None,
        "log_loss": log_loss(labels, scores, labels=[0, 1]),
        "brier": brier_score_loss(labels, scores),
    }


@torch.inference_mode()
def predict(model, rows, tokens, window, args, device):
    model.eval()
    return np.concatenate([
        model(history, question, lengths).sigmoid().cpu().numpy()
        for history, question, lengths, _ in batches(rows, tokens, window, args.batch_size, device)
    ]).astype(np.float64)


def historical_scores(rows, tokens, window, batch_size):
    scores = []
    for history, _, lengths, _ in batches(rows, tokens, window, batch_size, "cpu"):
        scores.append(((history % 2).sum(dim=1) + 1) / (lengths + 2))
    return torch.cat(scores).numpy().astype(np.float64)


def train_model(model, arrays, tokens, args, device, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    best_state, best_loss, best_epoch = None, float("inf"), 0
    history_log = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(arrays["train"]))
        for history, question, lengths, labels in batches(
            arrays["train"], tokens, args.window, args.batch_size, device, order
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(history, question, lengths), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scores = predict(model, arrays["val"], tokens, args.window, args, device)
        val_loss = log_loss(arrays["val"][:, 3], scores, labels=[0, 1])
        history_log.append({"epoch": epoch, "validation_log_loss": val_loss})
        print(f"{type(model).__name__} seed={seed} epoch={epoch} val_log_loss={val_loss:.5f}", flush=True)
        if val_loss < best_loss:
            best_state, best_loss, best_epoch = copy.deepcopy(model.state_dict()), val_loss, epoch
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    return best_epoch, history_log, time.perf_counter() - started


def evaluate(scorer, arrays, students, window):
    results = {}
    for name in ("val", "test_id", "test_ood"):
        rows = arrays[name]
        results[name] = metrics(rows[:, 3], scorer(rows, window))
        results[name]["students"] = len(np.unique(students[name]))
        results[name]["positive_rate"] = float(rows[:, 3].mean())
    rows = arrays["test_id"]
    long_history = rows[:, 1] - rows[:, 0] >= window
    subset = rows[long_history]
    results["short_history"] = {
        str(size): metrics(subset[:, 3], scorer(subset, size))
        for size in sorted({5, 10, 20, window}) if size <= window and len(subset)
    }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prepared interactions.csv")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.window, args.width, args.batch_size) < 1 or args.width % 4:
        parser.error("Use positive training settings and a width divisible by four")
    if args.output.exists():
        parser.error("Output already exists")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    arrays, students, tokens, skills, audit = read_interactions(args.input, args.split_seed)
    print({split: len(rows) for split, rows in arrays.items()}, flush=True)
    report = {
        "config": {key: value for key, value in vars(args).items() if key not in ("input", "output")},
        "torch": torch.__version__,
        "audit": audit,
        "historical_accuracy": evaluate(
            lambda rows, size: historical_scores(rows, tokens, size, args.batch_size),
            arrays, students, args.window,
        ),
        "runs": [],
    }
    for seed in args.seeds:
        for name in ("DKT", "SAKT"):
            torch.manual_seed(seed)
            model = (DKT(skills, args.width) if name == "DKT" else SAKT(skills, args.width, args.window)).to(device)
            # Unseen feature IDs have no fitted embedding.
            with torch.no_grad():
                model.questions.weight[1].zero_()
                model.answers.weight[2:4].zero_()
            epoch, history_log, seconds = train_model(model, arrays, tokens, args, device, seed)
            results = evaluate(lambda rows, size: predict(model, rows, tokens, size, args, device),
                               arrays, students, args.window)
            report["runs"].append({
                "model": name, "seed": seed, "best_epoch": epoch,
                "parameters": sum(p.numel() for p in model.parameters()),
                "training_seconds": round(seconds, 2), "history": history_log, "results": results,
            })
            print(f"{name} seed={seed} ID AUC={results['test_id']['roc_auc']:.4f} "
                  f"OOD AUC={results['test_ood']['roc_auc']:.4f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
