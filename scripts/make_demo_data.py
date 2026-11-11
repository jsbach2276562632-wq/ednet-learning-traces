"""Make a small synthetic dataset in EdNet's format to try the scripts locally."""
import argparse
import random
from pathlib import Path

from prepare_data import write_csv


def make_demo(output, students=30, seed=42):
    rng = random.Random(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'KT1').mkdir()
    (output / 'contents').mkdir()
    (output / 'SYNTHETIC_DATA.txt').write_text(
        'Made-up student records for testing the data preparation code.\n', encoding='utf-8')
    questions = [{'question_id': f'q{i}', 'bundle_id': f'b{i // 2}',
                  'correct_answer': 'a', 'part': str(i % 7 + 1), 'tags': str(i % 4 + 1)}
                 for i in range(20)]
    write_csv(output / 'contents/questions.csv', list(questions[0]), questions)
    for student in range(students):
        rows = []
        # Two students have too few answers to form a complete risk example.
        rounds = 4 if student == 0 else 15 if student == 1 else 30
        for solving in range(rounds):
            pair = rng.randrange(10)
            for offset in range(2):
                rows.append({'timestamp': 1500000000000 + solving * 120000 + offset,
                             'solving_id': solving + 1, 'question_id': f'q{pair * 2 + offset}',
                             'user_answer': 'a' if rng.random() < (.2 if student % 3 == 0 else .8) else 'b',
                             'elapsed_time': rng.randint(10000, 90000)})
        write_csv(output / f'KT1/u{student}.csv', list(rows[0]), rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/demo')
    args = parser.parse_args()
    make_demo(args.output)
    print(f'SYNTHETIC data created: {args.output}')
