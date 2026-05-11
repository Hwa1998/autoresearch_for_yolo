#!/usr/bin/env python3
"""精简版 LLM 补丁生成器。

流程：读取目标文件，调用 `lib.llm.LLMClient` 获取修改后的文件内容（要求 LLM 在
==BEGIN_FILE== / ==END_FILE== 标记间返回完整文件），生成统一 diff 并写入 `--out-dir`。
可选：使用 `--apply` 调用 `lib.patches.apply_and_test` 在临时分支上测试并决定是否保留。
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from lib.llm import LLMClient, LLMConfig
from lib import patches as patches_lib


def extract_between_markers(text: str) -> str | None:
    m = re.search(r"==BEGIN_FILE==\s*(.*?)\s*==END_FILE==", text, flags=re.S)
    if m:
        return m.group(1)
    m2 = re.search(r"==BEGIN_FILE==\s*(.*)", text, flags=re.S)
    if m2:
        return m2.group(1).rstrip()
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY'))
    p.add_argument('--api-endpoint', default=os.environ.get('LLM_API_ENDPOINT', 'https://api.openai.com/v1/chat/completions'))
    p.add_argument('--model', default='gpt-4o-mini')
    p.add_argument('--target', default='scripts/run_yolo_train.py')
    p.add_argument('--data', required=False)
    p.add_argument('--short-epochs', type=int, default=15)
    p.add_argument('--out-dir', default='suggestions/patches')
    p.add_argument('--task', default='Conservatively adjust default hyperparameters and expose TAL params as env vars; keep changes minimal.')
    p.add_argument('--apply', action='store_true', help='If set, attempt to apply patch via lib.patches.apply_and_test')
    args = p.parse_args()

    if not args.api_key:
        print('API key required via --api-key or OPENAI_API_KEY')
        sys.exit(2)

    target = Path(args.target)
    if not target.exists():
        print('Target not found:', target)
        sys.exit(2)

    orig = target.read_text(encoding='utf-8')
    prompt = f"Original file:\n---START---\n{orig}\n---END---\n\nTask:\n{args.task}\n\nReturn the full modified file content only between the markers ==BEGIN_FILE== and ==END_FILE==."

    client = LLMClient(LLMConfig(endpoint=args.api_endpoint, api_key=args.api_key, model=args.model))
    print('Calling LLM...')
    resp = client.chat('You are a code assistant. Return only the modified file between markers.', prompt)
    if not resp:
        print('LLM returned empty response')
        sys.exit(1)

    new_body = extract_between_markers(resp)
    if not new_body:
        print('LLM did not return content between markers. Full response:')
        print(resp[:2000])
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # basic safety check
    orig_text = orig
    if not patches_lib.safe_to_apply(new_body, orig_text):
        print('Generated content failed safety heuristics; aborting')
        sys.exit(1)

    # generate diff and write patch
    diff = patches_lib.generate_diff(orig_text, new_body, str(target))
    ts = int(__import__('time').time())
    patch_path = out_dir / f'llm-patch-{target.stem}-{ts}.patch'
    patches_lib.write_patch(patch_path, diff, out_dir)
    print('Wrote patch to', patch_path)

    result = {'patch': str(patch_path), 'applied': False, 'metrics': {}}

    if args.apply:
        if not args.data:
            print('--data is required when --apply')
        else:
            res = patches_lib.apply_and_test(patch_path, args.data, model=None, short_epochs=args.short_epochs, results_jsonl='runs/autoresearch/results.jsonl')
            result['applied'] = (res.get('result') == 'kept')
            result['metrics'] = {'mAP50': res.get('mAP50')}

    # write result record
    out_json = Path('runs') / 'autoresearch' / 'results.jsonl'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open('a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print('Done. Patch saved and record appended to', out_json)


if __name__ == '__main__':
    main()
