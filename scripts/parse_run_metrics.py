#!/usr/bin/env python3
"""Parse ultralytics run outputs (results.csv/json) and print a JSON metrics object.

Usage:
  python scripts/parse_run_metrics.py --run runs/detect/llm-short-1778207622
"""
import argparse
import csv
import json
from pathlib import Path


def parse_results_csv(p: Path):
    with p.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows or len(rows) < 2:
        return {}
    header = rows[0]
    last = rows[-1]
    # try to map numeric columns
    out = {}
    for i, name in enumerate(header):
        key = name.strip()
        try:
            val = float(last[i])
        except Exception:
            continue
        out[key] = val
    return out


def parse_results_json(p: Path):
    try:
        j = json.loads(p.read_text(encoding='utf-8'))
        return j
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='Path to runs/detect/<run>')
    args = parser.parse_args()
    runp = Path(args.run)
    if not runp.exists():
        print(json.dumps({'error': 'run not found', 'path': str(runp)}))
        return

    # check for results.csv
    csvp = runp / 'results.csv'
    if csvp.exists():
        metrics = parse_results_csv(csvp)
        print(json.dumps(metrics, ensure_ascii=False))
        return

    # check for results.json or metrics.json
    for fname in ('results.json', 'metrics.json'):
        jp = runp / fname
        if jp.exists():
            metrics = parse_results_json(jp)
            print(json.dumps(metrics, ensure_ascii=False))
            return

    # nothing found
    print(json.dumps({}))


if __name__ == '__main__':
    main()
