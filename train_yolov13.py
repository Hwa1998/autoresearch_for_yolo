"""
Wrapper to run a short or full YOLOv13 training job under the project TIME_BUDGET
with CLI overrides for dataset, class names and time budget. This wrapper uses a
temporary Ultralytics runner (so it does not modify `yolov13/train.py`).

Usage examples:
  python train_yolov13.py --data ./mydataset --names-file classes.txt --time-budget 180
  python train_yolov13.py --data yolov13/cfg/datasets/my.yaml --time-budget 300
"""

import os
import sys
import time
import signal
import subprocess
import tempfile
import argparse
from pathlib import Path

# Import TIME_BUDGET from prepare.py (fallback allowed)
try:
    from prepare import TIME_BUDGET as DEFAULT_TIME_BUDGET
except Exception:
    DEFAULT_TIME_BUDGET = 300


ROOT = Path(__file__).parent
LOG_PATH = ROOT / "run_yolov13.log"


def call_prepare_detection(python_exe, dataset_dir, names, names_file, out_yaml=None):
    cmd = [python_exe, str(ROOT / 'prepare_detection.py'), '--dataset-dir', str(dataset_dir)]
    if out_yaml:
        cmd += ['--output', str(out_yaml)]
    if names_file:
        cmd += ['--names-file', str(names_file)]
    if names:
        cmd += ['--names', names]
    subprocess.check_call(cmd)
    # return created yaml path
    return out_yaml or (ROOT / 'yolov13' / 'cfg' / 'datasets' / (Path(dataset_dir).name + '.yaml'))


def write_temp_runner(model_spec, data_yaml, epochs, imgsz, batch, name):
    code = f"""
from ultralytics import YOLO
model = YOLO('{model_spec}')
model.train(data=r'{data_yaml}', epochs={epochs}, imgsz={imgsz}, batch={batch}, name=r'{name}')
"""
    tf = tempfile.NamedTemporaryFile(delete=False, suffix='.py', mode='w', encoding='utf-8')
    tf.write(code)
    tf.flush()
    tf.close()
    return tf.name


def graceful_terminate(proc):
    try:
        proc.send_signal(signal.SIGINT)
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def find_latest_weights():
    candidates = [ROOT / 'yolov13' / 'runs' / 'detect', ROOT / 'runs' / 'detect', ROOT / 'yolov13' / 'runs', ROOT / 'runs']
    best = None
    best_mtime = 0
    for base in candidates:
        if not base.exists():
            continue
        for d in base.iterdir():
            if not d.is_dir():
                continue
            w_best = d / 'weights' / 'best.pt'
            w_last = d / 'weights' / 'last.pt'
            if w_best.exists() or w_last.exists():
                m = d.stat().st_mtime
                if m > best_mtime:
                    best_mtime = m
                    best = d
    if best is None:
        return None
    w_best = best / 'weights' / 'best.pt'
    if w_best.exists():
        return str(w_best)
    w_last = best / 'weights' / 'last.pt'
    if w_last.exists():
        return str(w_last)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='Path to dataset dir or dataset yaml')
    parser.add_argument('--names-file', type=str, default=None, help='Classes file (one per line)')
    parser.add_argument('--names', type=str, default=None, help='Comma-separated class names')
    parser.add_argument('--time-budget', type=int, default=int(DEFAULT_TIME_BUDGET), help='Seconds to allow training')
    parser.add_argument('--model', type=str, default='cfg/models/v8/yolov8n.yaml', help='Model yaml or weights to use with YOLO()')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs for the temporary run')
    parser.add_argument('--imgsz', type=int, default=640, help='imgsz passed to model.train')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--name', type=str, default='autoresearch_run', help='Run name')
    args = parser.parse_args()

    python_exe = sys.executable
    time_budget = args.time_budget
    print(f"TIME_BUDGET = {time_budget}s")

    # prepare data yaml if needed
    data_yaml = None
    if args.data:
        p = Path(args.data)
        # If user passed a directory, generate a dataset YAML into yolov13/cfg/datasets
        if p.is_dir():
            out_yaml = ROOT / 'yolov13' / 'cfg' / 'datasets' / (p.name + '.yaml')
            call_prepare_detection(python_exe, p, args.names, args.names_file, out_yaml=out_yaml)
            data_yaml = str(out_yaml)
        else:
            # If user passed a YAML file (possibly outside the repo), copy it into the
            # repo's yolov13/cfg/datasets folder for consistent relative paths.
            if p.suffix.lower() in ('.yml', '.yaml') and p.exists():
                dest_dir = ROOT / 'yolov13' / 'cfg' / 'datasets'
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / (p.stem + p.suffix)
                # avoid copying if already present and same file
                try:
                    import shutil
                    if not dest.exists() or p.resolve() != dest.resolve():
                        shutil.copy(str(p), str(dest))
                except Exception:
                    pass
                data_yaml = str(dest)
            else:
                data_yaml = str(p)

    # create temporary runner script that uses ultralytics API directly
    runner = write_temp_runner(args.model, data_yaml or '', args.epochs, args.imgsz, args.batch, args.name)

    env = os.environ.copy()
    with open(LOG_PATH, 'wb') as fout:
        start = time.time()
        proc = subprocess.Popen([python_exe, runner], stdout=fout, stderr=subprocess.STDOUT, env=env)
        try:
            while True:
                if proc.poll() is not None:
                    break
                elapsed = time.time() - start
                if elapsed >= time_budget:
                    print('TIME_BUDGET reached, terminating training...')
                    graceful_terminate(proc)
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            graceful_terminate(proc)

    total_seconds = time.time() - start

    weights = find_latest_weights()
    val_map = 0.0
    peak_vram_mb = 0.0
    if weights is None:
        print('No weights found after training; reporting crash-like summary.')
        print('---')
        print(f"val_map:          {0.0:.6f}")
        print(f"training_seconds: {min(total_seconds, time_budget):.1f}")
        print(f"total_seconds:    {total_seconds:.1f}")
        print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
        print(f"status:           crash")
        return

    # evaluate using ultralytics API
    try:
        from ultralytics import YOLO
        model = YOLO(str(weights))
        kwargs = {}
        if data_yaml:
            kwargs['data'] = data_yaml
        metrics = model.val(**kwargs)
        try:
            val_map = float(metrics.box.map) if hasattr(metrics, 'box') and hasattr(metrics.box, 'map') else 0.0
        except Exception:
            val_map = 0.0
    except Exception as e:
        print(f'Evaluation failed: {e}')
        val_map = 0.0

    try:
        import torch
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        peak_vram_mb = 0.0

    print('---')
    print(f"val_map:          {val_map:.6f}")
    print(f"training_seconds: {min(total_seconds, time_budget):.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")


if __name__ == '__main__':
    main()
