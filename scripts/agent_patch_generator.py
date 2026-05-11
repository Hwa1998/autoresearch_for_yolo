#!/usr/bin/env python3
"""LLM-agent scaffold: generate git-style .patch files from simple rules.

This script produces a unified diff patch that modifies target files
in-place (without committing). It's intentionally conservative: it only
updates default values of a few hyperparameter CLI args in
`scripts/run_yolo_train.py` so smoke tests are unlikely to fail.

Usage:
  python scripts/agent_patch_generator.py --target scripts/run_yolo_train.py --out-dir suggestions/patches --apply --data <data.yaml>

Options:
  --auto  : auto choose suggested values from heuristics (default True)
  --apply : after creating patch, call scripts/patch_runner.py to test and commit if improves
"""
import argparse
import difflib
import os
import subprocess
import sys
import time
from pathlib import Path
import sys


DEFAULT_UPDATES = {
    '--topk': '20',
    '--tal-beta': '2.5',
    '--crazing-boost': '2.0'
}


def read_text(p: Path):
    return p.read_text(encoding='utf-8')


def generate_modified_content(orig: str, updates: dict):
    lines = orig.splitlines()
    out = []
    for ln in lines:
        modified = ln
        for flag, val in updates.items():
            # find add_argument lines that contain the flag name
            if f"add_argument('{flag.lstrip('-')}" in ln or f'add_argument("{flag.lstrip("-")}' in ln:
                # naive replace default=None -> default=<val>
                if 'default=None' in ln or 'default=None,' in ln:
                    modified = ln.replace('default=None', f'default={val}')
        out.append(modified)
    return '\n'.join(out) + '\n'


def write_patch(orig_path: Path, new_content: str, out_dir: Path) -> Path:
    orig = read_text(orig_path).splitlines(keepends=True)
    new = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(orig, new, fromfile=str(orig_path), tofile=str(orig_path), lineterm='')
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    patch_path = out_dir / f'patch-{orig_path.stem}-{ts}.patch'
    with patch_path.open('w', encoding='utf-8') as f:
        for line in diff:
            f.write(line + '\n')
    return patch_path


def call_patch_runner(patch_path: Path, data: str, short_epochs: int = 15):
    cmd = [sys.executable, 'scripts/patch_runner.py', '--patch', str(patch_path), '--data', data, '--short-epochs', str(short_epochs)]
    print('Calling patch_runner:', ' '.join(cmd))
    subprocess.check_call(cmd)


def run_direct_apply(target: Path, new_text: str, data: str, short_epochs: int = 15):
    """Apply change directly (backup original), run smoke + short run, then restore original.

    This avoids relying on git being available in the environment. It is conservative:
    - backup created at <target>.bak.<ts>
    - write new content to target
    - run 1-epoch smoke, then short_epochs run
    - evaluate best weights if present
    - restore original file
    """
    ts = int(time.time())
    bak = target.with_suffix(target.suffix + f'.bak.{ts}')
    print('Backing up', target, 'to', bak)
    target.replace(bak)
    try:
        # write new content
        target.write_text(new_text, encoding='utf-8')

        # smoke test (1 epoch)
        smoke_name = f'direct-smoke-{ts}'
        smoke_cmd = [sys.executable, 'scripts/run_yolo_train.py', '--data', data, '--epochs', '1', '--name', smoke_name, '--workers', '0']
        print('Running smoke test:', ' '.join(smoke_cmd))
        # ensure local yolov13 ultralytics is importable by child process
        repo_root = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        yolov13_path = str(repo_root / 'yolov13')
        old_py = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = yolov13_path + (os.pathsep + old_py if old_py else '')
        subprocess.check_call(smoke_cmd, env=env)

        # short run
        short_name = f'direct-short-{ts}'
        short_cmd = [sys.executable, 'scripts/run_yolo_train.py', '--data', data, '--epochs', str(short_epochs), '--name', short_name, '--workers', '0']
        print('Running short run:', ' '.join(short_cmd))
        subprocess.check_call(short_cmd, env=env)

        # find best weights
        w_best = Path('runs') / 'detect' / short_name / 'weights' / 'best.pt'
        if not w_best.exists():
            w_best = Path('runs') / 'detect' / short_name / 'weights' / 'last.pt'

        metrics = {}
        if w_best.exists():
            # ensure project root is importable for prepare.py
            repo_root = Path(__file__).resolve().parent.parent
            added = False
            try:
                if str(repo_root) not in sys.path:
                    sys.path.insert(0, str(repo_root))
                    added = True
                # also set PYTHONPATH in env for any subprocesses
                old_py = os.environ.get('PYTHONPATH')
                os.environ['PYTHONPATH'] = str(repo_root)
                try:
                    from prepare import evaluate_placeholder
                    metrics = evaluate_placeholder(str(w_best), data)
                except Exception as e:
                    print('Evaluation failed:', e)
                finally:
                    # restore PYTHONPATH
                    if old_py is None:
                        os.environ.pop('PYTHONPATH', None)
                    else:
                        os.environ['PYTHONPATH'] = old_py
            finally:
                if added:
                    try:
                        sys.path.remove(str(repo_root))
                    except Exception:
                        pass

        print('Direct-apply run metrics:', metrics)

    finally:
        # restore original
        if bak.exists():
            print('Restoring original file from', bak)
            bak.replace(target)
        else:
            print('Backup not found; original not restored')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=False, default='scripts/run_yolo_train.py')
    parser.add_argument('--out-dir', required=False, default='suggestions/patches')
    parser.add_argument('--auto', action='store_true', default=True)
    parser.add_argument('--apply', action='store_true', help='Run patch_runner on generated patch')
    parser.add_argument('--direct-apply', action='store_true', help='Directly overwrite target file, run smoke/short run, then restore original (no git required)')
    parser.add_argument('--data', required=False, help='Dataset yaml for patch_runner (if --apply)')
    parser.add_argument('--short-epochs', type=int, default=15)
    args = parser.parse_args()

    target = Path(args.target)
    if not target.exists():
        print('Target file not found:', target)
        sys.exit(2)

    orig_text = read_text(target)

    updates = DEFAULT_UPDATES.copy()
    # placeholder for heuristics: if previous runs suggest different values, adjust here
    # for now we keep conservative defaults; this is where an LLM would be invoked.

    new_text = generate_modified_content(orig_text, updates)
    patch_path = write_patch(target, new_text, Path(args.out_dir))

    print('Generated patch:', patch_path)

    if args.apply:
        if not args.data:
            print('Error: --data required when using --apply')
            sys.exit(2)
        try:
            call_patch_runner(patch_path, args.data, short_epochs=args.short_epochs)
        except subprocess.CalledProcessError as e:
            print('patch_runner failed:', e)
            sys.exit(1)

    if args.direct_apply:
        if not args.data:
            print('Error: --data required when using --direct-apply')
            sys.exit(2)
        try:
            run_direct_apply(Path(args.target), new_text, args.data, short_epochs=args.short_epochs)
        except subprocess.CalledProcessError as e:
            print('Direct apply failed:', e)
            sys.exit(1)


if __name__ == '__main__':
    main()
