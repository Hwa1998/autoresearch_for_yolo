#!/usr/bin/env python3
"""Helper to run a full Phase1->Phase4 experiment using Deepseek as the LLM.

Sequence:
 - Phase1 (baselines)
 - Phase2 (hp search)
 - Phase3 (long validation)
 - Call `scripts/llm_patch_agent.py` with Deepseek API to generate patches into `suggestions/patches/llm_phase4`
 - Phase4 (apply patches)

Supports `--dry-run` which forwards `--dry-run` to `scripts/orchestrator.py` for quick validation.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, env=None):
    print('Running:', ' '.join(cmd))
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        raise SystemExit(f'Command failed with exit code {rc}: {cmd}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--llm-api-key', required=True)
    parser.add_argument('--llm-model', default='deepseek-v4-flash')
    parser.add_argument('--llm-endpoint', default='https://api.deepseek.com/v1/chat/completions')
    parser.add_argument('--short-epochs', type=int, default=15)
    parser.add_argument('--phase2-trials', type=int, default=12)
    parser.add_argument('--phase4-a-trials', type=int, default=10)
    parser.add_argument('--phase4-b-trials', type=int, default=30)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    env = None

    # Phase1
    cmd1 = [sys.executable, str(repo_root / 'scripts' / 'orchestrator.py'), '--data', args.data, '--phase', '1']
    if args.dry_run:
        cmd1 += ['--dry-run']
    run(cmd1, env=env)

    # Phase2
    cmd2 = [sys.executable, str(repo_root / 'scripts' / 'orchestrator.py'), '--data', args.data, '--phase', '2', '--phase2-trials', str(args.phase2_trials)]
    if args.dry_run:
        cmd2 += ['--dry-run']
    run(cmd2, env=env)

    # Phase3
    cmd3 = [sys.executable, str(repo_root / 'scripts' / 'orchestrator.py'), '--data', args.data, '--phase', '3']
    if args.dry_run:
        cmd3 += ['--dry-run']
    run(cmd3, env=env)

    # Generate Phase4 patches via llm_patch_agent (Deepseek)
    llm_out = repo_root / 'suggestions' / 'patches' / 'llm_phase4'
    llm_out.mkdir(parents=True, exist_ok=True)
    llm_cmd = [
        sys.executable,
        str(repo_root / 'scripts' / 'llm_patch_agent.py'),
        '--api-key', args.llm_api_key,
        '--api-endpoint', args.llm_endpoint,
        '--model', args.llm_model,
        '--target', 'scripts/run_yolo_train.py',
        '--data', args.data,
        '--short-epochs', str(args.short_epochs),
        '--out-dir', str(llm_out),
        '--task', 'Conservatively try small structural or loss changes to improve mAP on this dataset; be conservative and keep changes minimal.'
    ]
    print('\nInvoking llm_patch_agent to generate Phase4 patches (Deepseek)')
    if args.dry_run:
        print('[dry-run] would call:', ' '.join(llm_cmd))
    else:
        run(llm_cmd, env=env)

    # Phase4: run patch directory
    phase4_cmd = [sys.executable, str(repo_root / 'scripts' / 'orchestrator.py'), '--data', args.data, '--phase', '4', '--patch-dir', str(repo_root / 'suggestions' / 'patches' / 'llm_phase4'), '--phase4-a-trials', str(args.phase4_a_trials), '--phase4-b-trials', str(args.phase4_b_trials)]
    if args.dry_run:
        phase4_cmd += ['--dry-run']
    print('\nRunning Phase4 (apply generated patches)')
    run(phase4_cmd, env=env)

    print('\nExperiment complete. Results are stored under runs/autoresearch and suggestions/patches/llm_phase4')


if __name__ == '__main__':
    main()
