from pathlib import Path
base_stem='yolov8n'
candidates=[Path('yolov13')/'ultralytics'/'cfg'/'models'/'v8']

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

for c in candidates:
    print('Scanning', c)
    for f in sorted(c.glob('*.yaml')):
        fstem=f.stem
        fuzzy = (base_stem in fstem) or (fstem in base_stem) or fstem.startswith(base_stem[:6]) or base_stem.startswith(fstem[:6])
        match = model_matches_task(f, 'detect')
        print(fstem, 'fuzzy=',fuzzy,'match=',match)
