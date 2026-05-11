"""
训练运行器 - 调用 run_yolo_train.py 并收集结果。
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def run_training(name: str, data: str, epochs: int, hp: dict | None = None) -> float:
    """
    启动 YOLO 训练子进程，返回 elapsed 秒数。
    hp dict: key 映射为 --key value CLI 参数，bool 值仅传 flag。
    """
    cmd = [
        sys.executable, "scripts/run_yolo_train.py",
        "--data", data, "--epochs", str(epochs),
        "--name", name, "--workers", "0",
    ]
    if hp:
        for k, v in hp.items():
            arg = "--" + str(k).replace("_", "-")
            if v is None:
                continue
            if isinstance(v, bool):
                if v:
                    cmd.append(arg)
                continue
            cmd += [arg, str(v)]

    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    print("Running:", " ".join(cmd))
    t0 = time.time()
    subprocess.check_call(cmd, env=env)
    return time.time() - t0


def find_best_weights(run_name: str) -> str | None:
    """在 runs/ 下递归查找 run_name/weights/best.pt 或 last.pt。"""
    runs_root = Path("runs")
    if not runs_root.exists():
        return None
    for weights_dir in runs_root.rglob(f"{run_name}/weights"):
        for f in ("best.pt", "last.pt"):
            p = weights_dir / f
            if p.exists():
                return str(p)
    return None


def evaluate(weights_path: str, data: str, imgsz: int = 640, batch: int = 16, name: str | None = None) -> dict:
    """使用 prepare.evaluate_placeholder 评估权重。"""
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from prepare import evaluate_placeholder

    return evaluate_placeholder(weights_path, data, imgsz=imgsz, batch=batch, name=name)

