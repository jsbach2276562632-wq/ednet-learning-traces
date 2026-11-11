# First risk baselines

The 5,000-student EdNet sample used seed 42. The target is accuracy below 40% on the next 20 questions. The cutoff is the first completed round after at least 20 answers.

There were 895 training, 135 validation, and 247 test students with complete outcome windows. The test set contained 24 positive cases.

The historical score is one minus historical accuracy. Logistic regression also uses accuracy over the last 5 and 10 answers, mean response time, and history length. We fitted the scaler and regression on training data. For each score, we chose the alert threshold with the highest positive-class F1 on validation and applied it to the test set.

| Model | Split | ROC-AUC | Average precision | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Historical error rate | Validation | 0.661 | 0.224 | 0.250 | 0.471 | 0.327 |
| Historical error rate | Test | 0.770 | 0.379 | 0.260 | 0.542 | 0.351 |
| Logistic regression | Validation | 0.669 | 0.246 | 0.207 | 0.706 | 0.320 |
| Logistic regression | Test | 0.766 | 0.349 | 0.204 | 0.833 | 0.328 |

On the test set, the historical score had higher average precision. Logistic regression caught 20 of 24 cases and sent 98 alerts at its validation threshold. The historical score caught 13 and sent 50 alerts. With only 24 positive cases, we would repeat the comparison on a larger sample before choosing an alert policy.

This comparison covers students with enough follow-up answers. See the [label coverage analysis](risk_eda.md) for the students left out.

Run `python scripts/train_risk_baseline.py --input work/ednet` to reproduce the table. This run used Python 3.13.7 and scikit-learn 1.9.1.
