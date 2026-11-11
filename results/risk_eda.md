# First look at the risk labels

We used only the training split from the first EdNet run (seed 42). To reproduce these numbers from the project folder, run `python scripts/analyze_risk.py --input work/ednet`.

| Accuracy on the next 20 questions | Positive labels | Share of 895 labeled students |
|---|---:|---:|
| Below 30% | 29 | 3.2% |
| Below 40% | 95 | 10.6% |
| Below 50% | 205 | 22.9% |

We'll keep 40% as the main definition for the first baseline and use 30% and 50% to check sensitivity. The 30% cutoff leaves only 29 positive training examples. The 50% cutoff more than doubles the number of alerts and measures a broader outcome. There is no independent educational benchmark here that makes 40% the correct cutoff.

| Training students | Count | Median total answers | Mean accuracy at the prediction cutoff |
|---|---:|---:|---:|
| Enough history and follow-up | 895 | 130 | 52.1% |
| Enough history, fewer than 20 follow-up answers | 591 | 30 | 45.9% |
| Fewer than 20 historical answers | 2,026 | 7 | — |

The two comparable groups use the same cutoff rule: the first completed round after at least 20 answers. Mean accuracy gives each student equal weight. Only 25.5% of training students have a risk label. Those with missing follow-up were already about 6.2 percentage points less accurate at the cutoff, so results on labeled students should be reported with that coverage figure.
