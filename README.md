# Student Knowledge Tracing

The project report brings together the next-answer and early risk experiments. [PDF](report/report.pdf) · [Markdown](report/report.md)

Can a student's past answers tell us when they are starting to struggle? We use EdNet's English practice logs to predict the next answers and flag low accuracy over the following 20 questions.

The first EdNet run used 5,000 students and produced 653,911 cleaned responses. The [risk-label analysis](results/risk_eda.md) and [baseline comparison](results/risk_baseline.md) cover early risk prediction. The [knowledge tracing results](results/knowledge_tracing.md) compare DKT and SAKT on seen and unseen question bundles and with shorter histories.

## Try it

You need Python 3.11 or later. Run these commands from this folder:

```powershell
python scripts/make_demo_data.py
python scripts/prepare_data.py --input data/demo/KT1 --questions data/demo/contents/questions.csv --output work/demo
python -m unittest discover -s tests -v
```

The demo contains made-up students. It lets us check the code before using real records. Outputs go to `work/demo/`; start with `summary.json` or the preview notebook.

For the notebook, install the extra packages:

```powershell
python -m pip install -r requirements.txt
python -m jupyterlab
```

## Add EdNet

Download KT1 and Contents from the [EdNet repository](https://github.com/riiid/ednet). Save the ZIP files as `data/raw/archives/EdNet-KT1.zip` and `data/raw/archives/contents.zip`.

```powershell
python scripts/extract_ednet.py --students 5000 --seed 42
```

This checks both archives, extracts a repeatable 5,000-student sample and the question metadata, and saves a local manifest. The resulting layout is:

```text
data/raw/EdNet-KT1/KT1/u1.csv
data/raw/EdNet-KT1/KT1/u2.csv
data/raw/contents/questions.csv
```

Start with 5,000 students:

```powershell
python scripts/prepare_data.py --input data/raw/EdNet-KT1/KT1 --questions data/raw/contents/questions.csv --output work/ednet --max-students 5000
```

The script uses seed 42 for sampling and roughly 70/10/20 student splits. Choose a new output folder when rerunning; the script stops if the folder already exists.

## Reproduce the risk results

```powershell
python scripts/analyze_risk.py --input work/ednet
python -m pip install -r requirements.txt
python scripts/train_risk_baseline.py --input work/ednet
```

GitHub-ready files consist of code, documentation, and aggregate results. EdNet archives, student-level records, and generated working files belong in the ignored local directories.

## Where things are

- `scripts/`: demo generation and data preparation.
- `notebooks/01_data_preview.ipynb`: a first look at the processed data.
- `docs/project_specification.md`: the project plan.
- `docs/data_dictionary.md`: what the output columns mean.
- `docs/experiment_protocol.md`: splitting and evaluation details.
- `docs/initial_verification.md`: what we have checked so far.
- `data/README.md`: notes on the source files.
- `tests/`: checks for the preparation code.
- `results/` and `report/`: experiment outputs and the report outline.

The risk baselines use historical accuracy and logistic regression. The next-response experiment trains DKT and SAKT from scratch. Its reproduction command and PyTorch setup notes are in the [experiment results](results/knowledge_tracing.md).
