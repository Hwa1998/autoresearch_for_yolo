#!/usr/bin/env python3
"""Lightweight validator for LLM-generated Python file patches.

Checks:
- Syntax compile
- Byte-compile with py_compile
- (optional) smoke import / model build when given --smoke (adds yolov13 to PYTHONPATH)

Usage:
  python scripts/llm_patch_validator.py --file scripts/run_yolo_train.py [--smoke]
"""
import argparse
import compileall
import importlib.util
import os
import py_compile
import sys
from pathlib import Path


def compile_check(path: Path) -> bool:
    try:
        py_compile.compile(str(path), doraise=True)
        return True
    except py_compile.PyCompileError as e:
        print('py_compile failed:', e)
        return False


def ast_compile_check(path: Path) -> bool:
    try:
        src = path.read_text(encoding='utf-8')
        compile(src, str(path), 'exec')
        return True
    except Exception as e:
        print('compile() failed:', e)
        return False


def smoke_import(path: Path) -> bool:
    # Import module by path without adding parent package to sys.modules
    # Assumes safe top-level code; caller uses this only if they accept risk.
    try:
        spec = importlib.util.spec_from_file_location('llm_patch_validator.tmp', str(path))
        mod = importlib.util.module_from_spec(spec)
        # Insert temporary path to ensure local yolov13 is importable
        repo_root = Path(__file__).resolve().parent.parent
        yolov13_path = str(repo_root / 'yolov13')
        old_sys_path0 = None
        if yolov13_path not in sys.path:
            old_sys_path0 = list(sys.path)
            sys.path.insert(0, yolov13_path)
        spec.loader.exec_module(mod)
        # restore sys.path
        if old_sys_path0 is not None:
            sys.path[:] = old_sys_path0
        return True
    except Exception as e:
        print('smoke import failed:', e)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Python file to validate')
    parser.add_argument('--smoke', action='store_true', help='Attempt to import the module (risky)')
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print('File not found:', path)
        sys.exit(2)

    ok = True
    print('Running AST compile check...')
    if not ast_compile_check(path):
        ok = False

    print('Running py_compile...')
    if not compile_check(path):
        ok = False

    if args.smoke:
        print('Running smoke import (may execute top-level code)...')
        if not smoke_import(path):
            ok = False

    if ok:
        print('Validator: OK')
        sys.exit(0)
    else:
        print('Validator: FAILED')
        sys.exit(1)


if __name__ == '__main__':
    main()
