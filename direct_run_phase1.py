#!/usr/bin/env python3
"""
直接运行 Phase1 - 基线建立
"""
import os
import sys
import time
from pathlib import Path

# 设置路径
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'yolov13'))

os.chdir(repo_root)

print(f"[INFO] 工作目录: {os.getcwd()}")
print(f"[INFO] PYTHONPATH: {sys.path[:3]}")

# 现在导入 prepare 和 autoresearch_loop 模块
from prepare import evaluate_placeholder, append_record

# 现在导入 autoresearch_loop 中的函数
import argparse
import json
import random
import subprocess
from pathlib import Path

# 让我们查看一下 autoresearch_loop 的内容
with open('scripts/autoresearch_loop.py', 'r', encoding='utf-8') as f:
    content = f.read()
print(f"[INFO] autoresearch_loop 长度: {len(content)}")

# 首先测试一下是否能导入 ultralytics
try:
    import ultralytics
    print(f"[SUCCESS] 导入 ultralytics 成功: {ultralytics.__version__ if hasattr(ultralytics, '__version__') else 'unknown'}")
    print(f"[INFO] ultralytics 路径: {ultralytics.__file__}")
except Exception as e:
    print(f"[ERROR] 导入 ultralytics 失败: {e}")

print("\n" + "="*80)
print("环境检查完成")
print("="*80)
