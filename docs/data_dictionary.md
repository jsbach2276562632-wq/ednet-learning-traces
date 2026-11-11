# Prepared data

The preparation script writes five files. Steps start at zero, and all timestamps and response times are in milliseconds.

## students.csv

One row per student: `student_id`, `split`, `n_interactions`, and `risk_status`.

The split is roughly 70% train, 10% validation, and 20% test. Small samples can differ quite a bit from these proportions. Risk status is `observed`, `insufficient_history`, or `future_unobserved`.

## interactions.csv

The cleaned answer history, ordered within each student:

```text
student_id, split, step, timestamp, solving_id, question_id,
bundle_id, part, tags, is_correct, elapsed_time
```

`is_correct` is 1 when the submitted answer matches the answer key. Multiple knowledge tags stay together as a semicolon-separated string.

## next_response_targets.csv

One row per question we want to predict:

```text
student_id, split, target_step, history_end, question_id, label
```

The history is that student's interactions at `[0:history_end]`; the end index is excluded. Target question metadata can be used as a query. Its answer and response time are only known afterwards.

All targets in the same solving round share a history cutoff. The first round produces no targets because there is no earlier history.

## risk_examples.csv

At most one risk example per student.

| Column | Meaning |
|---|---|
| `student_id`, `split` | Student and dataset split |
| `history_end` | Number of completed questions available at prediction time |
| `history_accuracy` | Correct fraction across that history |
| `recent_5_accuracy`, `recent_10_accuracy` | Correct fraction over the last 5 or 10 available answers |
| `mean_elapsed_ms` | Mean response time across the history |
| `future_end` | End of the outcome window, excluded |
| `future_accuracy` | Correct fraction in that future window |
| `risk_label` | 1 when future accuracy is below the threshold; otherwise 0 |

Use `history_end` and the four historical statistics as features. The three future/outcome columns are for labels and evaluation.

At the default threshold, accuracy below 0.4 gets label 1; exactly 0.4 gets label 0.

## summary.json

The run settings, input-file hashes, script hash, cleaning counts, split totals, and counts of observed or missing risk outcomes. Keep this with the corresponding input files and script version when reproducing a run.

## Cleaning decisions

Missing required columns or duplicate question metadata stop the run. Unknown questions, invalid answers, invalid numeric values, negative times, and duplicate rows are skipped and counted separately.

Each solving round must contain one bundle, and round time ranges must not overlap. The script stops if either check fails.

Zero response times are kept. We will inspect long times in the training data before deciding on clipping. Missing answers are not filled in, and answers with several tags are not duplicated into separate samples.
