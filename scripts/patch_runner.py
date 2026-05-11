#!/usr/bin/env python3
"""Apply a patch safely: create temp branch, apply patch, smoke test, short run, commit or rollback.

Usage:
  python scripts/patch_runner.py --patch PATCH_FILE --data DATA_YAML [--short-epochs 15]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd, cwd=None, timeout=None):
    print('CMD:', ' '.join(cmd))
    return subprocess.check_call(cmd, cwd=cwd, timeout=timeout)


def find_best_map_from_results(jsonl_path):
    import json
    p = Path(jsonl_path)
    if not p.exists():
        return None
    best = None
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                m = None
                if 'metrics' in r and isinstance(r['metrics'], dict):
                    m = r['metrics'].get('mAP50') or r['metrics'].get('mAP50_95')
                if m is None:
                    m = r.get('val_map') or r.get('mAP50')
                if m is not None:
                    if best is None or float(m) > float(best):
                        best = float(m)
            except Exception:
                continue
    return best


def evaluate_weights(weights_path, data_yaml):
    # use prepare.evaluate_placeholder if available
    try:
        from prepare import evaluate_placeholder
        return evaluate_placeholder(weights_path, data_yaml)
    except Exception:
        return {'mAP50': None, 'mAP50_95': None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', default=None, help='Model YAML path to use for smoke/short training runs')
    parser.add_argument('--short-epochs', type=int, default=15)
    parser.add_argument('--smoke-timeout', type=int, default=300)
    parser.add_argument('--results-jsonl', default='runs/autoresearch/results.jsonl')
    parser.add_argument('--skip-clean-check', action='store_true', help='Skip git working-tree cleanliness check')
    args = parser.parse_args()

    patch = Path(args.patch)
    if not patch.exists():
        print('Patch file not found:', patch)
        sys.exit(2)

    ts = int(time.time())
    branch = f'autoresearch/patch-{ts}'
    # ensure working tree is clean (unless --skip-clean-check)
    if not args.skip_clean_check:
        try:
            status = subprocess.check_output(['git', 'status', '--porcelain']).decode().strip()
            if status:
                print('Git working tree is not clean. Commit or stash changes, or use --skip-clean-check to bypass.')
                print('Dirty files:')
                for line in status.splitlines()[:20]:
                    print('  ', line)
                sys.exit(2)
        except Exception:
            # git not available, skip check
            print('Git not available, skipping clean check.')

    prev_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
    prev_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

    try:
        run_cmd(['git', 'checkout', '-b', branch])
    except Exception as e:
        print('Failed to create branch:', e)
        sys.exit(1)

    try:
        # apply patch
        try:
            run_cmd(['git', 'apply', '--index', str(patch)])
        except Exception:
            # try git apply without index then git add
            run_cmd(['git', 'apply', str(patch)])
            run_cmd(['git', 'add', '-A'])

        # quick smoke test: 1 epoch short run
        smoke_name = f'{branch}-smoke'
        smoke_cmd = [sys.executable, 'scripts/run_yolo_train.py', '--data', args.data, '--epochs', '1', '--name', smoke_name, '--workers', '0']
        if args.model:
            smoke_cmd += ['--model', args.model]
        try:
            run_cmd(smoke_cmd, timeout=args.smoke_timeout)
        except subprocess.CalledProcessError as e:
            print('Smoke test failed, rolling back. Error:', e)
            # rollback to previous commit/branch
            try:
                run_cmd(['git', 'reset', '--hard', prev_commit])
                run_cmd(['git', 'checkout', prev_branch])
                run_cmd(['git', 'branch', '-D', branch])
            except Exception:
                pass
            sys.exit(1)

        # short run
        short_name = f'{branch}-short'
        short_cmd = [sys.executable, 'scripts/run_yolo_train.py', '--data', args.data, '--epochs', str(args.short_epochs), '--name', short_name, '--workers', '0']
        if args.model:
            short_cmd += ['--model', args.model]
        try:
            run_cmd(short_cmd)
        except subprocess.CalledProcessError as e:
            print('Short run failed, rolling back. Error:', e)
            try:
                run_cmd(['git', 'reset', '--hard', prev_commit])
                run_cmd(['git', 'checkout', prev_branch])
                run_cmd(['git', 'branch', '-D', branch])
            except Exception:
                pass
            sys.exit(1)

        # find weights
        weights = Path('runs') / 'detect' / short_name / 'weights' / 'best.pt'
        if not weights.exists():
            weights = Path('runs') / 'detect' / short_name / 'weights' / 'last.pt'

        metrics = {}
        if weights.exists():
            metrics = evaluate_weights(str(weights), args.data)

        # compare with best known
        best_known = find_best_map_from_results(args.results_jsonl) or 0.0
        new_map = metrics.get('mAP50') or metrics.get('mAP50_95') or 0.0

        print('best_known:', best_known, 'new_map:', new_map)
        if new_map > float(best_known):
            # commit
            run_cmd(['git', 'add', '-A'])
            run_cmd(['git', 'commit', '-m', f'Autoresearch patch: {patch.name} result mAP50={new_map}'])
            print('Patch improved metric — commit kept on branch', branch)
            # checkout prev branch and keep branch for inspection
            run_cmd(['git', 'checkout', prev_branch])
        else:
            print('No improvement — rolling back changes')
            try:
                run_cmd(['git', 'reset', '--hard', prev_commit])
                run_cmd(['git', 'checkout', prev_branch])
                run_cmd(['git', 'branch', '-D', branch])
            except Exception:
                pass

    except Exception as e:
        print('Unexpected error:', e)
        try:
            run_cmd(['git', 'reset', '--hard'])
            run_cmd(['git', 'checkout', prev_branch])
            run_cmd(['git', 'branch', '-D', branch])
        except Exception:
            pass
        raise


if __name__ == '__main__':
    main()
