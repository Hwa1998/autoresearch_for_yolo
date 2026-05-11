#!/usr/bin/env python3
"""长期自治 Agent 主循环：周期性调用补丁生成器并尝试应用。

使用示例：
python scripts/autonomous_agent.py --data path/to/data.yaml --api-key $KEY --sleep 30 --max-iterations 0

该脚本会：
 - 周期性调用 `scripts/llm_patch_agent.py`（或 `scripts/agent_patch_generator.py`，由 --use-conservative 控制）
 - 若生成补丁，会调用 `scripts/patch_runner.py` 来安全应用
 - 记录简易日志到 `runs/autoresearch/autonomous.log`
"""
import argparse
import subprocess
import time
import sys
import shutil
from pathlib import Path


def run_cmd(cmd, timeout=None):
    print('RUN:', ' '.join(cmd))
    return subprocess.check_call(cmd, timeout=timeout)


def run_proc(cmd, timeout=None):
    """Run a command and return CompletedProcess with captured output."""
    print('RUN:', ' '.join(cmd))
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)


def find_latest_patch(out_dir: Path):
    if not out_dir.exists():
        return None
    patches = sorted(out_dir.glob('*.patch'), key=lambda p: p.stat().st_mtime, reverse=True)
    return patches[0] if patches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--api-endpoint', default=None)
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--short-epochs', type=int, default=15)
    parser.add_argument('--sleep', type=int, default=30, help='seconds between iterations')
    parser.add_argument('--max-iterations', type=int, default=0, help='0 means infinite')
    parser.add_argument('--out-dir', default='suggestions/patches')
    parser.add_argument('--use-conservative', action='store_true', help='Use agent_patch_generator (conservative diffs)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = Path('runs') / 'autoresearch' / 'autonomous.log'
    logf.parent.mkdir(parents=True, exist_ok=True)

    iteration = 0
    while True:
        iteration += 1
        with logf.open('a', encoding='utf-8') as f:
            f.write(f'Iteration {iteration} start\n')

        # call conservative patch generator that emits unified diffs
        try:
            if args.use_conservative:
                cmd = [sys.executable, 'scripts/agent_patch_generator.py', '--target', 'scripts/run_yolo_train.py', '--out-dir', str(out_dir), '--apply', '--data', args.data, '--short-epochs', str(args.short_epochs)]
                cp = run_proc(cmd)
            else:
                # normalize model id: if user passed owner/model, use model part only
                model_arg = args.model.split('/')[-1] if args.model else ''
                cmd = [sys.executable, 'scripts/llm_patch_agent.py', '--api-key', args.api_key or '', '--api-endpoint', args.api_endpoint or '', '--model', model_arg, '--data', args.data, '--short-epochs', str(args.short_epochs), '--out-dir', str(out_dir)]
                cp = run_proc(cmd)

            if cp.returncode != 0:
                stderr = (cp.stderr or '') + (cp.stdout or '')
                print('Patch generator exited non-zero. stderr:', stderr)
                # try fallback: if api endpoint likely missing path, append common OpenAI chat path
                if args.api_endpoint and ('404' in stderr or 'Not Found' in stderr or '404 Client Error' in stderr):
                    fallback = args.api_endpoint.rstrip('/') + '/v1/chat/completions'
                    print('Attempting LLM fallback endpoint:', fallback)
                    cmd = [sys.executable, 'scripts/llm_patch_agent.py', '--api-key', args.api_key or '', '--api-endpoint', fallback, '--model', args.model, '--data', args.data, '--short-epochs', str(args.short_epochs), '--out-dir', str(out_dir)]
                    cp2 = run_proc(cmd)
                    if cp2.returncode == 0:
                        cp = cp2
                    else:
                        print('Fallback also failed. stderr:', (cp2.stderr or '') + (cp2.stdout or ''))
                        with logf.open('a', encoding='utf-8') as f:
                            f.write(f'Iteration {iteration} patch generation failed: {stderr}\n')
                else:
                    with logf.open('a', encoding='utf-8') as f:
                        f.write(f'Iteration {iteration} patch generation failed: {stderr}\n')
        except Exception as e:
            print('Unexpected error during patch generation:', e)
            with logf.open('a', encoding='utf-8') as f:
                f.write(f'Iteration {iteration} unexpected error: {e}\n')

        # look for latest patch and try to apply via patch_runner (if unified diff exists)
        latest = find_latest_patch(out_dir)
        if latest:
            try:
                # check git availability
                git_path = shutil.which('git')
                if git_path:
                    run_cmd([sys.executable, 'scripts/patch_runner.py', '--patch', str(latest), '--data', args.data, '--short-epochs', str(args.short_epochs)])
                else:
                    # fallback to direct-apply via agent_patch_generator
                    print('git not found; using direct-apply via agent_patch_generator')
                    run_cmd([sys.executable, 'scripts/agent_patch_generator.py', '--target', 'scripts/run_yolo_train.py', '--out-dir', str(out_dir), '--direct-apply', '--data', args.data, '--short-epochs', str(args.short_epochs)])

                with logf.open('a', encoding='utf-8') as f:
                    f.write(f'Iteration {iteration} applied patch: {latest.name}\n')
            except subprocess.CalledProcessError as e:
                print('Applying patch failed:', e)
                with logf.open('a', encoding='utf-8') as f:
                    f.write(f'Iteration {iteration} apply failed: {e}\n')

        # iteration control
        if args.max_iterations and iteration >= args.max_iterations:
            print('Reached max iterations, exiting')
            break

        time.sleep(args.sleep)


if __name__ == '__main__':
    main()
