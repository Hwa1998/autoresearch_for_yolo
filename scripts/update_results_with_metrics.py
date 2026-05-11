#!/usr/bin/env python3
"""Update runs/autoresearch/results.jsonl by filling missing metrics from runs/detect/* folders.

Usage:
  python scripts/update_results_with_metrics.py
"""
import json
import re
from pathlib import Path


def parse_results_csv_file(p: Path):
    import csv
    with p.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows or len(rows) < 2:
        return None
    header = rows[0]
    last = rows[-1]
    out = {}
    for i, name in enumerate(header):
        key = name.strip()
        try:
            val = float(last[i])
        except Exception:
            continue
        out[key] = val
    return out


def parse_run(run_path: Path):
    csvp = run_path / 'results.csv'
    if csvp.exists():
        return parse_results_csv_file(csvp)
    for fname in ('results.json', 'metrics.json'):
        jp = run_path / fname
        if jp.exists():
            try:
                return json.loads(jp.read_text(encoding='utf-8'))
            except Exception:
                return None
    return None


def main():
    jf = Path('runs') / 'autoresearch' / 'results.jsonl'
    if not jf.exists():
        print('no results.jsonl found')
        return
    lines = jf.read_text(encoding='utf-8').splitlines()
    out_lines = []
    changed = 0
    for line in lines:
        try:
            rec = json.loads(line)
        except Exception:
            out_lines.append(line)
            continue
        if isinstance(rec, dict) and rec.get('metrics') == {} and 'patch' in rec:
            m = re.search(r'-(\d{9,})\.patch$', rec['patch'])
            if m:
                ts = m.group(1)
                # find any runs/detect/*-<ts>
                candidates = list(Path('runs') .joinpath('detect').glob(f'*-{ts}'))
                if candidates:
                    rp = candidates[0]
                    parsed = parse_run(rp)
                    if parsed:
                        rec['metrics'] = parsed
                        changed += 1
                else:
                    rec['metrics'] = {'note': 'no matching run found'}
                    changed += 1
        out_lines.append(json.dumps(rec, ensure_ascii=False))

    if changed:
        bak = jf.with_suffix('.jsonl.bak')
        jf.replace(bak)
        jf.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
        print('Updated', changed, 'records; backup at', bak)
    else:
        print('No updates')


if __name__ == '__main__':
    main()
