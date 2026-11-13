# Experiment notes

These are the choices we need to keep consistent when comparing models.

## Student split

The preparation script hashes each student ID with seed 42 and assigns it to train, validation, or test, with approximate proportions of 70/10/20. A student belongs to one split. Sampling also uses a fixed hash, so reruns select the same files.

At test time, the model can read a student's completed history. It cannot train on that student's records or see future answers.

Fit vocabularies, scaling, difficulty estimates, and resampling on training data. Use validation data for early stopping, alert thresholds, and calibration. Keep test results for the final comparison. Any pretraining also uses training students only.

## What counts as history

Make predictions at the start of a solving round. Questions sharing a `solving_id` share the same cutoff, even if their timestamps differ slightly.

For risk prediction, select the historical feature columns explicitly. `future_accuracy`, `future_end`, and `risk_label` describe the outcome and cannot go into the model.

## Unseen question bundles

The first run uses a fixed bundle hash and a shared vocabulary of first tag and TOEIC part pairs built from training bundles. See [results and reproduction](../results/knowledge_tracing.md) for the settings and unknown-feature count.

Hold out whole `bundle_id` groups. Holding out individual questions could leave another question from the same reading passage in training.

Exclude held-out bundle responses from training, validation, and all input histories. Count test targets whose tag and part pair did not appear in training.

Compare seen and unseen bundles for the same test students. The models use tag and part pairs, with a zero embedding for unfamiliar pairs. Report sample counts and note that the bundle groups are not matched by difficulty.

Start this experiment with next-answer prediction. Keep risk labels on their original timeline: removing questions would change what “the next 20 questions” means.

## Short histories

For the same test targets with at least 50 prior answers, try the most recent 5, 10, 20, and 50 answers. All input ends before the target round. The left edge of the window can fall inside an earlier round.

Also check students who naturally have little history. Cutting a long record short does not necessarily recreate their situation.

## Reading the risk results

The platform chooses future questions, so accuracy depends on both the student and the assigned difficulty. Check results by question type and difficulty, and vary the future-window length and risk threshold.

When fewer than 20 future answers are available, mark the outcome as unobserved and report how many students this excludes.
