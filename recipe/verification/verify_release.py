"""Verify bundled code and original data without starting training or using the network."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, help='Optional original project root for read-only side-by-side comparison.')
    parser.add_argument('--decode-data', action='store_true', help='Also verify rows, columns and order with pyarrow.')
    args = parser.parse_args()
    src = json.loads(Path(__file__).with_name('source_manifest.json').read_text())
    data = json.loads((ROOT/'recipe/data/data_manifest.json').read_text())['files']
    errors = []
    for name, item in src['files'].items():
        path = ROOT/name
        if not path.is_file() or digest(path) != item['sha256']:
            errors.append(name)
    runtime = {p.relative_to(ROOT).as_posix() for p in (ROOT/'verl/verl').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
    expected = {p for p in src['files'] if p.startswith('verl/verl/')}
    if runtime != expected:
        errors.extend(sorted(runtime ^ expected))
    for name, item in data.items():
        path = ROOT/'recipe/data'/name
        if not path.is_file() or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            errors.append('recipe/data/'+name)
        elif args.decode_data:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            raw = json.dumps(table.to_pylist(), ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False).encode()
            if table.num_rows != item['rows'] or hashlib.sha256(raw).hexdigest() != item['rows_sha256']:
                errors.append('decoded: '+name)
    if errors:
        raise SystemExit('Release verification failed:\n'+'\n'.join(errors))
    print(f"Verified {len(expected)} runtime files, original installation/recipe files, and all 4 original Parquet byte hashes.")
    print('Declared packaging differences:', ', '.join(src['intentional_changes']))
    if args.original:
        differences = []
        for name in src['files']:
            a, b = ROOT/name, args.original/name
            if not b.is_file() or a.read_bytes() != b.read_bytes():
                differences.append(name)
        for name in data:
            name = 'recipe/data/'+name
            a, b = ROOT/name, args.original/name
            if not b.is_file() or a.read_bytes() != b.read_bytes():
                differences.append(name)
        unexpected = sorted(set(differences)-set(src['intentional_changes']))
        if unexpected:
            raise SystemExit('Unexpected differences from original:\n'+'\n'.join(unexpected))
        print('Original comparison: only explicitly declared release and documentation changes differ.')
    print('This proves file identity, not GPU execution or paper-level accuracy.')


if __name__ == '__main__':
    main()
