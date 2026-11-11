import csv
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from make_demo_data import make_demo
from prepare_data import bucket, load_questions, load_student, prepare, risk_example, write_csv


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        make_demo(self.root / 'demo')
        self.input = self.root / 'demo/KT1'
        self.questions = self.root / 'demo/contents/questions.csv'

    def read(self, path):
        with path.open(encoding='utf-8', newline='') as handle:
            return list(csv.DictReader(handle))

    def test_end_to_end_student_splits_and_bundle_boundaries(self):
        summary = prepare(self.input, self.questions, self.root / 'out')
        records = self.read(self.root / 'out/interactions.csv')
        targets = self.read(self.root / 'out/next_response_targets.csv')
        indexed = {(r['student_id'], int(r['step'])): r for r in records}
        students = {}
        for row in records:
            students.setdefault(row['student_id'], set()).add(row['split'])
        self.assertTrue(all(len(s) == 1 for s in students.values()))
        self.assertEqual(set.union(*students.values()), {'train', 'val', 'test_id'})
        for target in targets:
            cutoff = int(target['history_end'])
            first = indexed[target['student_id'], cutoff]
            actual = indexed[target['student_id'], int(target['target_step'])]
            previous = indexed[target['student_id'], cutoff - 1]
            self.assertEqual(first['solving_id'], actual['solving_id'])
            self.assertNotEqual(previous['solving_id'], actual['solving_id'])
            self.assertEqual(int(target['label']), int(actual['is_correct']))
        self.assertEqual(summary['risk_observability']['insufficient_history'], 1)
        self.assertEqual(summary['risk_observability']['future_unobserved'], 1)

    def test_risk_label_and_features_use_disjoint_windows(self):
        rows = [{'is_correct': int(i < 4), 'elapsed_time': 100} for i in range(8)]
        boundaries = [(0, 2), (2, 4), (4, 6), (6, 8)]
        sample, status = risk_example(rows, boundaries, 3, 4, .4)
        self.assertEqual(status, 'observed')
        self.assertEqual(sample['history_end'], 4)
        self.assertEqual(sample['history_accuracy'], 1)
        self.assertEqual(sample['future_accuracy'], 0)
        self.assertEqual(sample['risk_label'], 1)
        for row in rows[4:]:
            row['is_correct'] = 1
        changed, _ = risk_example(rows, boundaries, 3, 4, .4)
        self.assertEqual(changed['history_accuracy'], sample['history_accuracy'])
        self.assertEqual(changed['risk_label'], 0)

    def test_exact_threshold_is_not_positive(self):
        rows = [{'is_correct': i % 2, 'elapsed_time': 10} for i in range(4)]
        example, _ = risk_example(rows, [(0, 2), (2, 4)], 2, 2, .5)
        self.assertEqual(example['risk_label'], 0)

    def test_cleaning_counts(self):
        path = self.input / 'u0.csv'
        rows = self.read(path)
        rows += [dict(rows[0]), {**rows[0], 'question_id': 'missing'},
                 {**rows[0], 'user_answer': ''}, {**rows[0], 'elapsed_time': '-1'}]
        write_csv(path, list(rows[0]), rows)
        counts = Counter()
        valid, _ = load_student(path, load_questions(self.questions), counts)
        self.assertEqual(len(valid), 8)
        for key in ['duplicate_rows', 'unknown_question_rows', 'invalid_answer_rows', 'invalid_numeric_rows']:
            self.assertEqual(counts[key], 1)

    def test_reproducible_sampling_and_no_overwrite(self):
        first = prepare(self.input, self.questions, self.root / 'a', max_students=10)
        second = prepare(self.input, self.questions, self.root / 'b', max_students=10)
        self.assertEqual(first, second)
        self.assertEqual(bucket('u15', 42), bucket('u15', 42))
        with self.assertRaises(FileExistsError):
            prepare(self.input, self.questions, self.root / 'a')

    def test_missing_headers_fail(self):
        bad = self.root / 'bad.csv'
        write_csv(bad, ['question_id'], [{'question_id': 'q1'}])
        with self.assertRaisesRegex(ValueError, 'missing columns'):
            load_questions(bad)


if __name__ == '__main__':
    unittest.main()
