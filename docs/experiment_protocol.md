# Experiment notes

These are the choices we need to keep consistent when comparing models.

## Student split

The preparation script hashes each student ID with seed 42 and assigns it to train, validation, or test, with approximate proportions of 70/10/20. A student belongs to one split. Sampling also uses a fixed hash, so reruns select the same files.

At test time, the model can read a student's completed history. It cannot train on that student's records or see future answers.

Fit vocabularies, scaling, difficulty estimates, and resampling on training data. Use validation data for early stopping, alert thresholds, and calibration. Keep test results for the final comparison. Any pretraining also uses training students only.

## What counts as history

Make predictions at the start of a solving round. Questions sharing a `solving_id` share the same cutoff, even if their timestamps differ slightly.

For risk prediction, select the historical feature columns explicitly. `future_accuracy`, `future_end`, and `risk_label` describe the outcome and cannot go into the model.

## Unseen question bundles — still to implement

Hold out whole `bundle_id` groups. Holding out individual questions could leave another question from the same reading passage in training.

Prefer bundles whose tags appear in training, and count unfamiliar tags separately. Remove held-out bundle answers and times from training and validation. For the strict zero-shot test, also leave earlier answers from those bundles out of the test history.

Compare seen and unseen bundles for the same test students. Report question difficulty and sample counts alongside the scores. Unknown question IDs map to UNK; tags and question modules provide the remaining information.

Start this experiment with next-answer prediction. Keep risk labels on their original timeline: removing questions would change what “the next 20 questions” means.

## Short histories

For the same test targets, try the most recent 5, 10, 20, and 50 questions. Cut at completed-round boundaries and record the actual number available.

Also check students who naturally have little history. Cutting a long record short does not necessarily recreate their situation.

## Reading the risk results

The platform chooses future questions, so accuracy depends on both the student and the assigned difficulty. Check results by question type and difficulty, and vary the future-window length and risk threshold.

When fewer than 20 future answers are available, mark the outcome as unobserved and report how many students this excludes.
