#!/usr/bin/env python3
"""生成 model YAML 变体并输出统一 diff 补丁供 patch_runner 使用。

此脚本极其保守：仅文本级替换给定关键字（如替换某一类 block 名称），并输出差异文件到 out-dir。
用法示例：
python scripts/model_variant_generator.py --base cfg/models/v8/yolov8n.yaml --replace C2f:C2PSA --out-dir suggestions/patches/model_variants
"""
import argparse
from pathlib import Path
import difflib
import time


def read_text(p: Path):
    return p.read_text(encoding='utf-8')


def write_patch(orig_path: Path, new_text: str, out_dir: Path) -> Path:
    orig = read_text(orig_path).splitlines(keepends=True)
    new = new_text.splitlines(keepends=True)
    diff = difflib.unified_diff(orig, new, fromfile=str(orig_path), tofile=str(orig_path), lineterm='')
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    patch_path = out_dir / f'variant-{orig_path.stem}-{ts}.patch'
    with patch_path.open('w', encoding='utf-8') as f:
        for line in diff:
            f.write(line + '\n')
    return patch_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, help='base model yaml path')
    parser.add_argument('--replace', required=True, help='REPLACE pattern like Old:New, supports multiple separated by comma')
    parser.add_argument('--data', help='optional dataset yaml to infer task (detect, cls, obb, seg)')
    parser.add_argument('--out-dir', default='suggestions/patches/model_variants')
    args = parser.parse_args()

    base = Path(args.base)

    # If dataset provided, try to infer task and filter candidate files accordingly
    dataset_task = None
    if args.data:
        try:
            dpath = Path(args.data)
            dn = dpath.stem.lower()
            if 'seg' in dn or 'mask' in dn:
                dataset_task = 'seg'
            elif 'obb' in dn or 'rbox' in dn:
                dataset_task = 'obb'
            elif 'cls' in dn or 'class' in dn:
                dataset_task = 'cls'
            else:
                dataset_task = 'detect'
        except Exception:
            dataset_task = None

    # If the provided base path doesn't exist, try common fallback locations
    if not base.exists():
        # Candidate files and directories to search for matching YAMLs
        # include both 'ultralytics' and possible 'ultralytice' dirs
        candidates = [
            Path('yolov13') / 'ultralytics' / 'cfg' / 'models' / 'v8',
            Path('yolov13') / 'ultralytice' / 'cfg' / 'models' / 'v8',
            Path('yolov13') / 'ultralytics' / 'cfg' / 'models',
            Path('yolov13') / 'ultralytice' / 'cfg' / 'models',
            Path('yolov13') / 'ultralytics' / 'cfg',
            Path('yolov13') / 'ultralytice' / 'cfg',
            Path('yolov13'),
        ]
        found = None
        # First check direct filenames under candidates
        for c in candidates:
            direct = c / base.name
            if direct.exists():
                if args.data and dataset_task:
                    if model_matches_task(direct, dataset_task):
                        found = direct
                        break
                    else:
                        continue
                else:
                    found = direct
                    break
        # If not found, try fuzzy match by stem
        if not found:
            # collect fuzzy matches and score them to prefer base-name prefix matches like 'yolov8' <- 'yolov8n'
            scored = []
            for c in candidates:
                if not c.exists():
                    continue
                for f in c.rglob('*.yaml'):
                    try:
                        name = f.stem.lower()
                        base_st = base.stem.lower()
                        score = 0
                        if name == base_st:
                            score += 200
                        # prefer exact basename as prefix (yolov8 <- yolov8n)
                        if base_st.startswith(name):
                            score += 150
                        # substring matches
                        if name in base_st or base_st in name:
                            score += 100
                        # common 6-char prefix
                        if name.startswith(base_st[:6]) or base_st.startswith(name[:6]):
                            score += 50
                        # task match bonus
                        if args.data and dataset_task and model_matches_task(f, dataset_task):
                            score += 20
                        if score > 0:
                            scored.append((score, f))
                    except Exception:
                        continue
            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                found = scored[0][1]
            # else found remains None
        if found:
            print('Base model yaml not found at', base, '-- falling back to', found)
            base = found
        else:
            print('Base model yaml not found:', base)
            return
    def model_matches_task(model_path: Path, task: str) -> bool:
        name = model_path.stem.lower()
        # filename hints
        if task == 'cls':
            if 'cls' in name or 'class' in name or 'resnet' in name:
                return True
            # avoid detection/seg models
            if any(k in name for k in ('yolov8', 'detect', 'obb', 'seg')):
                return False
        if task == 'seg':
            if 'seg' in name or 'mask' in name:
                return True
        if task == 'obb':
            if 'obb' in name or 'rot' in name or 'rbox' in name:
                return True
        if task == 'detect':
            # avoid models clearly for other tasks
            if any(k in name for k in ('cls', 'seg', 'pose', 'obb')):
                return False
            # prefer yolov8/detect-like names
            if 'yolo' in name or 'yolov8' in name or 'detect' in name or 'rtde' in name:
                return True
        # fallback: inspect file header for 'tasks/' comment
        try:
            header = model_path.read_text(encoding='utf-8').splitlines()[:20]
            header_text = '\n'.join(header).lower()
            if task == 'detect' and 'tasks/detect' in header_text:
                return True
            if task == 'cls' and ('tasks/class' in header_text or 'task docs' in header_text and 'class' in header_text):
                return True
            if task == 'seg' and ('tasks/seg' in header_text or 'segmentation' in header_text):
                return True
        except Exception:
            pass
        return False

    orig_text = read_text(base)
    out_dir = Path(args.out_dir)

    for pair in args.replace.split(','):
        if ':' not in pair:
            continue
        old, new = pair.split(':', 1)
        new_text = orig_text.replace(old, new)
        patch = write_patch(base, new_text, out_dir)
        print('Wrote patch:', patch)


if __name__ == '__main__':
    import time
    main()
