"""Check the downloaded EdNet archives and extract a reproducible KT1 sample."""
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def extract(archives, raw, students=5000, seed=42):
    if students < 1:
        raise ValueError('students must be positive')
    kt1 = archives / 'EdNet-KT1.zip'
    contents = archives / 'contents.zip'
    output = raw / 'EdNet-KT1' / 'KT1'
    metadata = raw / 'contents'
    if output.exists() or metadata.exists():
        raise FileExistsError('Extracted data already exists; choose a new raw directory')
    with zipfile.ZipFile(kt1) as source, zipfile.ZipFile(contents) as info:
        # Reading every archive member checks CRCs before we mark the download ready.
        print('Checking KT1 archive CRCs...', flush=True)
        bad = source.testzip()
        if bad:
            raise ValueError(f'KT1 CRC failure: {bad}')
        if info.testzip():
            raise ValueError('Contents CRC failure')
        members = [m for m in source.infolist()
                   if not m.is_dir() and Path(m.filename).suffix == '.csv'
                   and Path(m.filename).stem.startswith('u')
                   and '__MACOSX' not in m.filename]
        ids = [Path(m.filename).stem for m in members]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate student filenames in archive')
        members.sort(key=lambda m: hashlib.sha256(
            f'{seed}:sample:{Path(m.filename).stem}'.encode()).hexdigest())
        selected = members[:students]
        if not selected:
            raise ValueError('No student CSVs found in KT1 archive')
        question_members = [m for m in info.infolist()
                            if Path(m.filename).name == 'questions.csv'
                            and '__MACOSX' not in m.filename]
        if len(question_members) != 1:
            raise ValueError('Expected one questions.csv in Contents')
        output.mkdir(parents=True)
        metadata.mkdir(parents=True)
        # Write only known CSV basenames; archive paths are not used as destinations.
        for member in info.infolist():
            if member.is_dir() or '__MACOSX' in member.filename:
                continue
            name = Path(member.filename).name
            if not name.endswith('.csv'):
                continue
            with info.open(member) as src, (metadata / name).open('xb') as dest:
                shutil.copyfileobj(src, dest)
        for member in selected:
            with source.open(member) as src, (output / Path(member.filename).name).open('xb') as dest:
                shutil.copyfileobj(src, dest)
        with (metadata / 'questions.csv').open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            question_columns = reader.fieldnames
            question_count = sum(1 for _ in reader)
        manifest = {
            'source': 'https://github.com/riiid/ednet',
            'download_status': 'complete_archives_crc_checked',
            'extraction': 'student_sample_full_archive_retained',
            'archive_students': len(members), 'extracted_students': len(selected),
            'seed': seed, 'question_count': question_count,
            'question_columns': question_columns,
            'student_ids': [Path(m.filename).stem for m in selected],
        }
        for path in (kt1, contents):
            with path.open('rb') as handle:
                manifest[path.name] = {'bytes': path.stat().st_size,
                                       'sha256': hashlib.file_digest(handle, 'sha256').hexdigest()}
        (raw / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in manifest.items() if k != 'student_ids'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, default=Path('data/raw/archives'))
    parser.add_argument('--raw', type=Path, default=Path('data/raw'))
    parser.add_argument('--students', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    extract(args.archives, args.raw, args.students, args.seed)
