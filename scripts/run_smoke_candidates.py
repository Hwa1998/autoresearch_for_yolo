import yaml
import subprocess
import os
import re
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(REPO_ROOT / '.venv' / 'Scripts' / 'python.exe')
RUN_YOLO = str(REPO_ROOT / 'scripts' / 'run_yolo_train.py')
CONFIG = REPO_ROOT / 'configs' / 'autoresearch_config.yaml'
OUT_DIR = REPO_ROOT / 'runs' / 'autoresearch'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / 'smoke_params.jsonl'

with open(CONFIG, 'r', encoding='utf-8') as f:
    conf = yaml.safe_load(f)

candidates = conf.get('candidates', [])
if not candidates:
    print('No candidates found in config')
    raise SystemExit(1)

# data path: prefer CLI env var or config; fallback to user-provided path
DATA = os.environ.get('AUTORESEARCH_DATA') or os.environ.get('DATA') or None
if DATA is None:
    # default from conversation
    DATA = r'D:\DingReihwa\6_AI\datasets\yolo_datasets\detect\dingweixiao\dingweixiao.yaml'

results = []

for cand in candidates:
    p = (REPO_ROOT / cand).resolve()
    if not p.exists():
        print('Candidate not found:', p)
        continue
    # load YAML
    try:
        with p.open('r', encoding='utf-8') as f:
            d = yaml.safe_load(f) or {}
    except Exception as e:
        print('Failed to load', p, e)
        continue
    scales = d.get('scales')
    if not isinstance(scales, dict):
        # try sibling family
        family = p.stem.split('_')[0]
        sibling = p.parent / f"{family}.yaml"
        if sibling.exists():
            try:
                with sibling.open('r', encoding='utf-8') as sf:
                    fam = yaml.safe_load(sf) or {}
                    if isinstance(fam.get('scales'), dict):
                        scales = fam['scales']
            except Exception:
                pass
    if not isinstance(scales, dict):
        # treat as single-scale (no expansion)
        scale_keys = [d.get('scale') or '']
    else:
        scale_keys = list(scales.keys())

    for s in scale_keys:
        name = f"smoke-{p.stem}-{s}" if s else f"smoke-{p.stem}"
        cmd = [PYTHON, RUN_YOLO, '--model', str(p), '--data', DATA, '--scale', s, '--epochs', '1', '--name', name, '--workers', '0']
        env = os.environ.copy()
        env['PYTHONPATH'] = str(REPO_ROOT / 'yolov13') + os.pathsep + str(REPO_ROOT)
        print('Running:', ' '.join(cmd))
        try:
            cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=600)
            out = cp.stdout.decode('utf-8', errors='replace')
        except subprocess.TimeoutExpired:
            out = 'TIMEOUT'
        # parse parameters and GFLOPs
        m = re.search(r"summary: .*?, ([\d,]+) parameters.*?, ([\d\.]+) GFLOPs", out)
        if not m:
            # try alternative pattern
            m = re.search(r"summary: .* ([\d,]+) parameters.* ([\d\.]+) GFLOPs", out)
        params = None
        gflops = None
        if m:
            params = int(m.group(1).replace(',', ''))
            gflops = float(m.group(2))
        # also capture first lines that indicate tmp yaml
        tmp_info = None
        tm = re.search(r"WROTE tmp model yaml: (.+)", out)
        if tm:
            tmp_info = tm.group(1).strip()
        entry = {'candidate': cand, 'model_path': str(p), 'scale': s, 'name': name, 'params': params, 'gflops': gflops, 'tmp_yaml': tmp_info, 'stdout_snippet': out[-4000:]} 
        results.append(entry)
        with OUT_FILE.open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print('Result:', entry['candidate'], s, 'params=', params, 'gflops=', gflops)

print('\nAll runs finished. Results written to', OUT_FILE)
print('Summary:')
for r in results:
    print(r['candidate'], r['scale'], r['params'], r['gflops'])
