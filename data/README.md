# Getting the data

Downloaded on 2026-11-11. The full KT1 archive contains 784,309 student CSVs; 5,000 have been extracted to `raw/EdNet-KT1/KT1/` for the first experiment. `raw/contents/questions.csv` contains 13,169 questions. Both archives passed CRC checks. File sizes, SHA256 hashes, and sampled student IDs are in `raw/download_manifest.json`.

Download **EdNet-KT1** and **Contents** from the [official repository](https://github.com/riiid/ednet). KT1 is listed at about 1.2 GB compressed and 5.6 GB extracted. The dataset uses the CC BY-NC 4.0 license.

Keep the real files in `raw/`. The demo script writes made-up records to `demo/`. Both folders are ignored by Git.

Each KT1 file belongs to one student, identified by its filename. The expected columns are:

```text
timestamp,solving_id,question_id,user_answer,elapsed_time
```

The official README has a mismatched example header; the parser follows its field descriptions and checks the actual column names.

The Contents file `questions.csv` needs:

```text
question_id,bundle_id,correct_answer,part,tags
```

A question can have several tags, separated by semicolons.

A few details matter for the experiments: timestamps have been shifted, so calendar dates should not be treated as real school terms. KT1 has no mobile/web field. Knowledge tags are numeric IDs; we should check what they represent before giving them readable names.
