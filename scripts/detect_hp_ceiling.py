#!/usr/bin/env python3
"""检测 Phase2 超参搜索是否已到天花板的简单启发式脚本。

读取 `runs/autoresearch/phase2.jsonl`（或自定义 path），输出 JSON 到 stdout，格式：{"ceiling": true/false, "max": 0.76, "median_first_half": 0.75}

启发式逻辑（保守）：如果 Phase2 的最大 mAP 与第一半 trials 的中位数差小于阈值（默认 0.005），认为到达天花板。
"""
import argparse
import json
from pathlib import Path
import statistics


def read_metrics(path: Path):
    if not path.exists():
        return []
    res = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                m = None
                if 'metrics' in r and isinstance(r['metrics'], dict):
                    m = r['metrics'].get('mAP50') or r['metrics'].get('mAP50_95')
                if m is None:
                    m = r.get('val_map') or r.get('mAP50')
                if m is not None:
                    res.append(float(m))
            except Exception:
                continue
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase2-jsonl', default='runs/autoresearch/phase2.jsonl')
    parser.add_argument('--threshold', type=float, default=0.005, help='min improvement over baseline median to consider not ceiling')
    args = parser.parse_args()

    p = Path(args.phase2_jsonl)
    metrics = read_metrics(p)
    out = {'ceiling': False, 'max': None, 'median_first_half': None, 'n': len(metrics)}
    if metrics:
        metrics_sorted = metrics[:]  # keep order
        mmax = max(metrics_sorted)
        out['max'] = mmax
        half = len(metrics_sorted) // 2 or len(metrics_sorted)
        first_half = metrics_sorted[:half]
        try:
            med = statistics.median(first_half)
        except Exception:
            med = first_half[0]
        out['median_first_half'] = med
        out['delta'] = mmax - med
        out['ceiling'] = (mmax - med) < args.threshold

    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
