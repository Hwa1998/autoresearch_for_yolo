"""
统一配置加载 - 从 YAML 配置文件或 CLI args 合并参数。
"""
import yaml
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict:
    """加载 YAML 配置文件，返回 dict。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_cli_overrides(cfg: dict, cli_args: Any, known_cli_flags: list[str]):
    """
    将 YAML 配置中的 key 映射到 CLI args 的属性上。
    CLI 参数优先：若用户在命令行显式传入，则不用 config 覆盖。
    匹配规则：config key `xxx_yyy` 对应 CLI flag `--xxx-yyy`。
    """
    import sys

    for key in known_cli_flags:
        if key not in cfg:
            continue
        cli_flag = "--" + key.replace("_", "-")
        if cli_flag in sys.argv:
            continue
        setattr(cli_args, key, cfg[key])
