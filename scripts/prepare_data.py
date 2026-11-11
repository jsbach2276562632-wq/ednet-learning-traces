"""Clean EdNet KT1 and build student splits and prediction targets.

Uses the Python standard library. Pass input and output paths on the command line.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def bucket(student, seed):
    value = int(hashlib.sha256(f'{seed}:split:{student}'.encode()).hexdigest(), 16)
    fraction = value / 2**256
    return 'train' if fraction < .7 else 'val' if fraction < .8 else 'test_id'


def read_csv(path, required):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        yield from reader


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_questions(path):
    questions = {}
    fields = ['question_id', 'bundle_id', 'correct_answer', 'part', 'tags']
    for row in read_csv(path, fields):
        key = row['question_id']
        if key in questions:
            raise ValueError(f'Duplicate question metadata: {key}')
        if (not key or not row['bundle_id'] or row['part'] not in list('1234567')
                or row['correct_answer'] not in 'abcd' or not row['correct_answer']):
            raise ValueError(f'Invalid question metadata: {key}')
        questions[key] = row
    return questions


def load_student(path, questions, counts):
    required = ['timestamp', 'solving_id', 'question_id', 'user_answer', 'elapsed_time']
    seen, rounds = set(), defaultdict(list)
    for row in read_csv(path, required):
        counts['raw_rows'] += 1
        signature = tuple(row[key] for key in required)
        if signature in seen:
            counts['duplicate_rows'] += 1
            continue
        seen.add(signature)
        meta = questions.get(row['question_id'])
        if meta is None:
            counts['unknown_question_rows'] += 1
            continue
        answer = row['user_answer']
        if answer not in ('a', 'b', 'c', 'd'):
            counts['invalid_answer_rows'] += 1
            continue
        try:
            timestamp, solving, elapsed = (int(row[key]) for key in
                                           ('timestamp', 'solving_id', 'elapsed_time'))
            if min(timestamp, solving, elapsed) < 0:
                raise ValueError
        except (ValueError, TypeError):
            counts['invalid_numeric_rows'] += 1
            continue
        rounds[solving].append({
            'timestamp': timestamp, 'solving_id': solving,
            'question_id': row['question_id'], 'bundle_id': meta['bundle_id'],
            'part': meta['part'], 'tags': meta['tags'],
            'is_correct': int(answer == meta['correct_answer']), 'elapsed_time': elapsed,
        })
    ordered = sorted(rounds.values(), key=lambda rows: (min(r['timestamp'] for r in rows),
                                                       rows[0]['solving_id']))
    result, boundaries, previous_max = [], [], -1
    for rows in ordered:
        rows.sort(key=lambda row: (row['timestamp'], row['question_id']))
        if len({r['bundle_id'] for r in rows}) != 1:
            raise ValueError(f'{path}: one solving_id contains multiple bundles')
        if rows[0]['timestamp'] < previous_max:
            raise ValueError(f'{path}: overlapping solving rounds; inspect raw timestamps')
        previous_max = max(r['timestamp'] for r in rows)
        start = len(result)
        for row in rows:
            row['step'] = len(result)
            result.append(row)
        boundaries.append((start, len(result)))
    counts['valid_rows'] += len(result)
    return result, boundaries


def risk_example(rows, boundaries, history, horizon, threshold):
    cutoff = next((end for _, end in boundaries if end >= history), None)
    if cutoff is None:
        return None, 'insufficient_history'
    if len(rows) - cutoff < horizon:
        return None, 'future_unobserved'
    past, future = rows[:cutoff], rows[cutoff:cutoff + horizon]
    mean = lambda items: sum(r['is_correct'] for r in items) / len(items)
    accuracy = mean(future)
    return {
        'history_end': cutoff, 'history_accuracy': mean(past),
        'recent_5_accuracy': mean(past[-5:]), 'recent_10_accuracy': mean(past[-10:]),
        'mean_elapsed_ms': sum(r['elapsed_time'] for r in past) / len(past),
        'future_end': cutoff + horizon, 'future_accuracy': accuracy,
        'risk_label': int(accuracy < threshold),
    }, 'observed'


def prepare(input_dir, question_path, output, max_students=5000,
            seed=42, history=20, horizon=20, threshold=.4):
    if history < 1 or horizon < 1 or max_students < 0 or not 0 <= threshold <= 1:
        raise ValueError('Require positive windows, nonnegative max-students and threshold in [0,1]')
    files = sorted(Path(input_dir).glob('*.csv'))
    if not files:
        raise ValueError(f'No student CSV files in {input_dir}')
    # Hash ranking avoids bias towards the smallest numeric user IDs.
    files.sort(key=lambda p: hashlib.sha256(f'{seed}:sample:{p.stem}'.encode()).hexdigest())
    if max_students:
        files = files[:max_students]
    questions = load_questions(Path(question_path))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts, splits, risks, manifests = Counter(), defaultdict(Counter), Counter(), []
    interaction_fields = ['student_id', 'split', 'step', 'timestamp', 'solving_id',
                          'question_id', 'bundle_id', 'part', 'tags', 'is_correct', 'elapsed_time']
    target_fields = ['student_id', 'split', 'target_step', 'history_end', 'question_id', 'label']
    risk_fields = ['student_id', 'split', 'history_end', 'history_accuracy',
                   'recent_5_accuracy', 'recent_10_accuracy', 'mean_elapsed_ms',
                   'future_end', 'future_accuracy', 'risk_label']
    # Stream students rather than storing the whole dataset in memory.
    from contextlib import ExitStack
    with ExitStack() as stack:
        writers = {}
        for name, fields in [('interactions', interaction_fields), ('next_response_targets', target_fields),
                             ('risk_examples', risk_fields),
                             ('students', ['student_id', 'split', 'n_interactions', 'risk_status'])]:
            handle = stack.enter_context((output / f'{name}.csv').open('w', encoding='utf-8', newline=''))
            writers[name] = csv.DictWriter(handle, fieldnames=fields)
            writers[name].writeheader()
        for path in files:
            student, split = path.stem, bucket(path.stem, seed)
            rows, boundaries = load_student(path, questions, counts)
            common = {'student_id': student, 'split': split}
            manifests.append({'path': str(path.resolve()), 'sha256': digest(path), 'split': split})
            splits[split]['students'] += 1
            splits[split]['interactions'] += len(rows)
            splits[split]['correct'] += sum(r['is_correct'] for r in rows)
            for row in rows:
                writers['interactions'].writerow({**common, **row})
            for start, end in boundaries:
                if start == 0:
                    continue
                for step in range(start, end):
                    row = rows[step]
                    writers['next_response_targets'].writerow({**common, 'target_step': step,
                        'history_end': start, 'question_id': row['question_id'], 'label': row['is_correct']})
                    splits[split]['next_targets'] += 1
            example, status = risk_example(rows, boundaries, history, horizon, threshold)
            risks[status] += 1
            writers['students'].writerow({**common, 'n_interactions': len(rows), 'risk_status': status})
            if example:
                writers['risk_examples'].writerow({**common, **example})
                splits[split]['risk_examples'] += 1
                splits[split]['risk_positives'] += example['risk_label']
    summary = {
        'status': 'data_preparation_only_no_model_results',
        'config': {'seed': seed, 'max_students': max_students, 'history': history,
                   'horizon': horizon, 'risk_threshold': threshold},
        'question_file': {'path': str(Path(question_path).resolve()), 'sha256': digest(Path(question_path))},
        'script_sha256': digest(Path(__file__)),
        'cleaning': dict(counts), 'splits': dict(splits), 'risk_observability': dict(risks),
        'student_files': manifests,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-students', type=int, default=5000, help='0 selects all students')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--history', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=.4)
    args = parser.parse_args()
    try:
        summary = prepare(args.input, args.questions, args.output, args.max_students,
                          args.seed, args.history, args.horizon, args.threshold)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Data preparation failed: {exc}\nUse a new output directory when retrying.\n')
    print(json.dumps({key: summary[key] for key in ('config', 'cleaning', 'splits', 'risk_observability')}, indent=2))


if __name__ == '__main__':
    main()
