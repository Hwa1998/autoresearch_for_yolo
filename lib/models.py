"""
模型候选发现 - 自动扫描 YAML 并展开尺度。
"""
import re
import yaml
from pathlib import Path


SCALES = {"n", "s", "m", "l", "x"}
FALLBACK_SCALES = ["n", "s"]


def discover_models(task_type: str = "detect") -> list[dict]:
    """
    扫描 yolov13/ultralytics/cfg/models/ 下所有 .yaml，筛选匹配 task_type 的模型。
    返回 [{"model": path, "display": stem, "scale": scale_or_None}, ...]
    """
    repo_root = Path(__file__).resolve().parent.parent
    models_root = repo_root / "yolov13" / "ultralytics" / "cfg" / "models"
    candidates = _scan_candidates(models_root, task_type)
    if not candidates:
        candidates = [
            "yolov13/ultralytics/cfg/models/v13/yolov13_nmsfree.yaml",
            "yolov13/ultralytics/cfg/models/v13/yolov13.yaml",
            "yolov13/ultralytics/cfg/models/v13/yolov13_v10detect.yaml",
        ]
    return _expand_to_scale_entries(candidates, models_root, task_type)


def expand_from_config(candidate_paths: list[str], task_type: str = "detect") -> list[dict]:
    """从配置指定的候选路径展开尺度，返回 [{"model": ..., "display": ..., "scale": ...}, ...]"""
    repo_root = Path(__file__).resolve().parent.parent
    models_root = repo_root / "yolov13" / "ultralytics" / "cfg" / "models"
    return _expand_to_scale_entries(candidate_paths, models_root, task_type)


# ── internal helpers ──────────────────────────────────────────

def _scan_candidates(models_root: Path, task: str) -> list[str]:
    out = []
    if not models_root.exists():
        return out
    for p in models_root.rglob("*.yaml"):
        try:
            if _matches_task(p, task):
                out.append(str(p))
        except Exception:
            out.append(str(p))
    return list(dict.fromkeys(out))


def _matches_task(p: Path, task: str) -> bool:
    """检查模型 YAML 是否匹配目标任务类型。"""
    try:
        with p.open("r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
    except Exception:
        return True
    declared = (y.get("task") or y.get("type") or "").lower()
    if declared:
        return task in declared
    # 回退：文件名启发
    name = p.name.lower()
    exclude = {"_cls", "/cls", "obb", "pose", "seg", "rtdetr", "rt-detr", "rt_detr"}
    if task == "detect":
        return not any(tok in name for tok in exclude)
    return task in name


def _expand_to_scale_entries(candidate_paths: list[str], models_root: Path, task_type: str) -> list[dict]:
    expanded = []
    all_yaml = list(models_root.rglob("*.yaml")) if models_root.exists() else []

    for cand in candidate_paths:
        p = Path(cand)
        # 处理相对路径
        if not p.exists():
            repo_root = Path(__file__).resolve().parent.parent
            p_alt = repo_root / cand
            if p_alt.exists():
                p = p_alt

        if not p.exists():
            continue

        if not _matches_task(p, task_type):
            continue

        stem = p.stem
        base_token = stem.split("_")[0]

        # 判断是否已包含尺度标记（如 yolov13n）
        token_scale = None
        if len(base_token) > 1 and base_token[-1] in SCALES and any(ch.isdigit() for ch in base_token):
            token_scale = base_token[-1]
            family = base_token[:-1]
        else:
            family = base_token

        if token_scale:
            # 已指定尺度，只添加一个条目
            display = _insert_scale(stem, token_scale)
            entry = {"model": str(p), "display": display, "scale": token_scale}
            if entry not in expanded:
                expanded.append(entry)
        else:
            # family 级别，从 family YAML 读取 scales
            scales_list = _read_family_scales(p, family, models_root)
            if not scales_list:
                scales_list = FALLBACK_SCALES
            for sc in scales_list:
                display = _insert_scale(stem, sc)
                entry = {"model": str(p), "display": display, "scale": sc}
                if entry not in expanded:
                    expanded.append(entry)

    return expanded


def _read_family_scales(model_path: Path, family: str, models_root: Path) -> list[str]:
    """读取 family YAML 中声明的 scales 列表。"""
    # 先尝试当前文件自身
    try:
        with model_path.open("r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        if isinstance(y.get("scales"), dict):
            return [k for k, v in y["scales"].items() if isinstance(v, (list, tuple))]
    except Exception:
        pass

    # 再尝试同级目录下的 family.yaml
    for sibling in [
        model_path.parent / f"{family}.yaml",
        *models_root.rglob(f"{family}.yaml"),
    ]:
        if sibling.exists():
            try:
                with sibling.open("r", encoding="utf-8") as f:
                    y = yaml.safe_load(f) or {}
                if isinstance(y.get("scales"), dict):
                    return [k for k, v in y["scales"].items() if isinstance(v, (list, tuple))]
            except Exception:
                pass
    return []


def _insert_scale(stem: str, scale: str) -> str:
    """将 scale 插入到 stem 中的 yolov<digits> 之后。"""
    m = re.search(r"(yolov\d+)", stem, re.IGNORECASE)
    if m:
        start, end = m.span(1)
        return stem[:end] + scale + stem[end:]
    return f"{stem}{scale}"


def sort_models(models: list[dict]) -> list[dict]:
    """按 family → scale 顺序排序。"""
    default_order = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}

    def key(e):
        stem = Path(e["model"]).stem
        family = stem.split("_")[0]
        scale_idx = default_order.get(e.get("scale", ""), 999)
        return (family, scale_idx, e["display"])

    return sorted(models, key=key)
