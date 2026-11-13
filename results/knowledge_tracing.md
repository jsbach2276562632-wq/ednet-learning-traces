# Next-response prediction on seen and unseen bundles

We trained compact implementations of DKT and SAKT from scratch on the same 5,000-student EdNet sample used for the risk baseline. Each model saw up to 50 earlier responses and used the first listed skill tag and TOEIC part as its question features. The historical baseline used the same window with one correct and one incorrect pseudo-count.

The split holds out roughly one fifth of question bundles by a fixed hash. Held-out responses never enter training, validation, or model histories. Test targets come from the same 659 students in both groups. There were 119,423 seen-bundle and 29,727 unseen-bundle targets, including 148 unseen-bundle targets whose tag and part combination was absent from training. These used a zero embedding. The two groups differ in their question mix and are not matched by difficulty.

| Model | Seen AUC | Unseen AUC | Seen average precision | Unseen average precision | Seen log loss | Unseen log loss |
|---|---:|---:|---:|---:|---:|---:|
| Historical accuracy | 0.629 | 0.624 | 0.761 | 0.766 | 0.618 | 0.615 |
| DKT | 0.655 | 0.633 | 0.780 | 0.769 | 0.603 | 0.607 |
| SAKT | 0.655 | 0.630 | 0.781 | 0.764 | 0.603 | 0.610 |

The sequence-model scores are means across seeds 42, 43, and 44. Both models improve on historical accuracy for seen bundles. On unseen bundles the AUC gains shrink to about 0.009 for DKT and 0.006 for SAKT. SAKT's unseen average precision is slightly below the baseline. These are descriptive results for one student and bundle split.

On the same seen-bundle targets with at least 50 historical answers, reducing the history to 5 answers lowered AUC from 0.645 to 0.609 for DKT and from 0.644 to 0.600 for SAKT. This measures the effect of truncation on active students. The full per-seed results, Brier scores, and training logs are in [knowledge_tracing.json](knowledge_tracing.json).

After preparing EdNet as described in the README, install PyTorch for your device and run:

```powershell
python scripts/train_knowledge_tracing.py --input work/ednet/interactions.csv --output work/ednet/knowledge_tracing.json --epochs 6 --width 32
```

The reported run used PyTorch 2.11.0+cu128 and an RTX 4070 Laptop GPU. We selected each checkpoint by validation log loss. This experiment predicts answer correctness. The separate [risk baseline](risk_baseline.md) predicts low accuracy over the next 20 answers.
