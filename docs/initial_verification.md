# First run

Checked on 2026-11-11 with Python 3.13.7.

The demo run produced 30 synthetic students, 1,718 valid interactions, and 28 risk examples. The student split was 17 train, 4 validation, and 9 test. One student had too little history; another had too few future answers.

All six preparation tests passed:

```powershell
python -m unittest discover -s tests -v
```

The notebook's JSON and code syntax were checked. It has not been executed with the plotting dependencies yet.

At this stage, real EdNet data, model training, and the held-out-bundle experiment were still pending.

Later on 2026-11-11, the full KT1 and Contents downloads finished and passed CRC checks. We extracted 5,000 of the 784,309 student files using seed 42, together with the Contents CSVs. The question table has 13,169 rows. Model training and the held-out-bundle experiment are still pending.
