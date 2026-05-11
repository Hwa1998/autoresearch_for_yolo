"""
补丁工具 - 生成、校验、安全应用补丁。
"""
import difflib
import os
import subprocess
import sys
import time
from pathlib import Path


# ── 补丁生成 ──────────────────────────────────────────────────

def generate_diff(orig_text: str, new_text: str, filepath: str) -> str:
    """生成 unified diff 文本。"""
    orig = orig_text.splitlines(keepends=True)
    new = new_text.splitlines(keepends=True)
    lines = list(difflib.unified_diff(orig, new, fromfile=filepath, tofile=filepath, lineterm=""))
    return "\n".join(lines)


def write_patch(text: str, out_dir: Path, prefix: str) -> Path:
    """将 diff 文本写入 .patch 文件，返回 path。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = out_dir / f"{prefix}-{ts}.patch"
    path.write_text(text, encoding="utf-8")
    return path


# ── 保守补丁 (修改 run_yolo_train.py 默认值) ────────────────

CONSERVATIVE_UPDATES = {
    "--topk": "20",
    "--tal-beta": "2.5",
    "--crazing-boost": "2.0",
}


def generate_conservative_patch(target: str | Path = "scripts/run_yolo_train.py", updates: dict | None = None) -> str:
    """生成保守补丁：只改 add_argument 的 default 值。"""
    if updates is None:
        updates = CONSERVATIVE_UPDATES
    p = Path(target)
    orig = p.read_text(encoding="utf-8")
    lines = orig.splitlines()
    out_lines = []
    for ln in lines:
        modified = ln
        for flag, val in updates.items():
            flag_stripped = flag.lstrip("-")
            if f"add_argument('{flag_stripped}" in ln or f'add_argument("{flag_stripped}' in ln:
                modified = ln.replace("default=None", f"default={val}")
        out_lines.append(modified)
    return generate_diff(orig, "\n".join(out_lines) + "\n", str(p))


# ── 校验 ──────────────────────────────────────────────────────

def validate_python(filepath: Path) -> bool:
    """AST compile + py_compile 双重校验。"""
    import py_compile
    try:
        compile(filepath.read_text(encoding="utf-8"), str(filepath), "exec")
        py_compile.compile(str(filepath), doraise=True)
        return True
    except Exception as e:
        print(f"Validation failed: {e}")
        return False


def safe_to_apply(new_text: str, orig_text: str | None = None) -> bool:
    """检查 LLM 生成的代码是否安全。"""
    lowered = new_text.lower()
    # 必须保留训练调用
    if not ("model.train(" in lowered or ".train(" in lowered or "yolo(" in lowered):
        return False
    # 变更不能超过 300 行
    if orig_text:
        diff = list(difflib.unified_diff(orig_text.splitlines(), new_text.splitlines()))
        changed = sum(1 for l in diff if l.startswith("+") or l.startswith("-"))
        if changed > 300:
            print(f"Patch too large: {changed} lines changed")
            return False
    return True


# ── 安全应用 ──────────────────────────────────────────────────

def apply_and_test(
    patch_path: Path,
    data: str,
    model: str | None = None,
    short_epochs: int = 15,
    results_jsonl: str = "runs/autoresearch/results.jsonl",
    skip_clean_check: bool = False,
) -> dict:
    """
    对 patch 进行 git 分支 → smoke → short run → keep/discard 流程。
    返回 {"mAP50": ..., "result": "kept"|"discarded"|"error"}
    """
    import json
    from lib.training import find_best_weights, evaluate

    patch_name = patch_path.stem
    ts = int(time.time())
    branch = f"autoresearch/patch-{ts}"

    # --- git 准备 ---
    if not skip_clean_check:
        try:
            status = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
            if status:
                print("Working tree dirty; skip or clean first.")
                return {"mAP50": 0.0, "result": "error"}
        except Exception:
            pass

    try:
        prev_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        prev_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return {"mAP50": 0.0, "result": "error"}

    # --- 创建分支并应用补丁 ---
    try:
        subprocess.check_call(["git", "checkout", "-b", branch])
    except Exception as e:
        print(f"Failed to create branch: {e}")
        return {"mAP50": 0.0, "result": "error"}

    try:
        # apply patch
        try:
            subprocess.check_call(["git", "apply", "--index", str(patch_path)])
        except Exception:
            subprocess.check_call(["git", "apply", str(patch_path)])
            subprocess.check_call(["git", "add", "-A"])

        # smoke test (1 epoch)
        _run_training(f"{branch}-smoke", data, 1, model, timeout=300)

        # short run
        short_name = f"{branch}-short"
        _run_training(short_name, data, short_epochs, model)

        # 评估
        weights = find_best_weights(short_name)
        metrics = evaluate(weights, data) if weights else {}
        new_map = float(metrics.get("mAP50") or metrics.get("mAP50_95") or 0.0)

        # 比较历史最佳
        best_known = _read_best_map(results_jsonl)
        print(f"best_known={best_known:.4f} new_map={new_map:.4f}")

        if new_map > best_known:
            subprocess.check_call(["git", "add", "-A"])
            subprocess.check_call(
                ["git", "commit", "-m", f"Autoresearch patch: {patch_name} mAP50={new_map:.4f}"]
            )
            print(f"Patch improved → kept on branch {branch}")
            subprocess.check_call(["git", "checkout", prev_branch])
            return {"mAP50": new_map, "result": "kept"}
        else:
            print("No improvement → rollback")
            _rollback(prev_commit, prev_branch, branch)
            return {"mAP50": new_map, "result": "discarded"}

    except Exception as e:
        print(f"Error during patch apply: {e}")
        _rollback(prev_commit, prev_branch, branch)
        return {"mAP50": 0.0, "result": "error"}


# ── 内部 ──────────────────────────────────────────────────────

def _run_training(name: str, data: str, epochs: int, model: str | None, timeout: int | None = None):
    cmd = [
        sys.executable, "scripts/run_yolo_train.py",
        "--data", data, "--epochs", str(epochs),
        "--name", name, "--workers", "0",
    ]
    if model:
        cmd += ["--model", model]
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    print("CMD:", " ".join(cmd))
    subprocess.check_call(cmd, env=env, timeout=timeout)


def _rollback(prev_commit: str, prev_branch: str, branch: str):
    try:
        subprocess.check_call(["git", "reset", "--hard", prev_commit])
        subprocess.check_call(["git", "checkout", prev_branch])
        subprocess.check_call(["git", "branch", "-D", branch])
    except Exception:
        pass


def _read_best_map(jsonl_path: str) -> float:
    import json
    p = Path(jsonl_path)
    if not p.exists():
        return 0.0
    best = 0.0
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            m = r.get("metrics", {}).get("mAP50") or r.get("mAP50") or 0.0
            best = max(best, float(m))
        except Exception:
            pass
    return best
