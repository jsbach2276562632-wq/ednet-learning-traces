# Project plan

We want to find out how early a student's answer history can tell us they may need help. EdNet gives us English practice records, including which questions students attempted, their answers, and how long they spent.

The project follows the same broad structure as our satellite project: inspect the data, give every model the same split, compare results, then look closely at where the models fail.

## What we predict

The first task is to predict whether a student will answer a question correctly. We use completed answers from earlier rounds, plus the new question's ID, tags, and module.

EdNet presents some questions together. All questions in one `solving_id` therefore get the same history cutoff. Answers and times from that round only become available afterwards.

The second task is an early risk prediction. After at least 20 completed questions, we predict whether accuracy on the next 20 will fall below 40%. We make one risk prediction per student, at a completed-round boundary.

The 20-question windows and 40% threshold are starting choices. We will inspect the training data before settling on them and compare thresholds of 30%, 40%, and 50%. Students without enough future answers have an unknown outcome. We keep their usable records for the first task.

## Models to compare

Start with historical accuracy and Logistic Regression. The regression can use overall accuracy, accuracy on the last 5 and 10 questions, mean response time, and history length.

Then add DKT-LSTM and SAKT. They use the order of past answers rather than only summary statistics. We will give them the same data and comparable tuning budgets. Both initially train from scratch.

For next-answer prediction, report ROC-AUC, PR-AUC, Log Loss, and Brier score. For risk prediction, report PR-AUC, precision, recall, macro-F1, and Brier score. Also ask a practical question: if a tutor can help only k students, how many flagged students go on to have low accuracy?

Use the validation set for early stopping, alert thresholds, and calibration. Run three seeds and report training time, model size, and best epoch. Bootstrap by student for confidence intervals, and include student-weighted results so very active students do not dominate. AUC is undefined when a subgroup has only one class.

## Where we expect difficulties

We will test new students, held-out question bundles, and shorter histories. New students from the same source are our standard test set. Held-out bundles provide the OOD experiment. Truncating histories tells us how much evidence the model needs, but may differ from testing naturally low-activity students.

A low score can mean the student is struggling, or simply that the platform gave them harder questions. We need difficulty breakdowns and a few inspected examples before interpreting alerts as support needs. Answer prediction is an indirect measure of knowledge.

Students who continue for another 20 questions may also differ from those who stop earlier. We will report how many risk labels are missing and compare their earlier performance. With no demographic data, our subgroup analysis will cover learning behavior rather than demographic fairness.

Attention plots may help explore SAKT's behavior, but do not establish why a prediction was made. The logs alone also cannot tell us whether an intervention would improve learning.

## Work ahead

The first 5,000-student sample has been prepared and checked. The [risk-label analysis](../results/risk_eda.md) and [risk baseline](../results/risk_baseline.md) are available.

1. Extend the data checks to question tags and response times.
2. Build the next-response baseline, then train DKT and SAKT.
3. Evaluate held-out bundles, short histories, and calibration.
4. Collect the figures and write the report.

For now, we will stay with KT1 and these two prediction tasks. Recommendations, generated lessons, and formal dropout prediction would need separate work.

## Reading

- [EdNet data and documentation](https://github.com/riiid/ednet)
- [Deep Knowledge Tracing](https://proceedings.neurips.cc/paper/2015/hash/bac9162b47c56fc8a4d2a519803d51b3-Abstract.html)
- [Self-Attentive Knowledge Tracing](https://arxiv.org/abs/1907.06837)
