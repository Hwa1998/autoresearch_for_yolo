from pathlib import Path
base = Path('cfg/models/v8/yolov8n.yaml')
args_data = 'D:/DingReihwa/6_AI/datasets/yolo_datasets/detect/dingweixiao/dingweixiao.yaml'

candidates = [
    Path('yolov13') / 'ultralytics' / 'cfg' / 'models' / 'v8',
    Path('yolov13') / 'ultralytics' / 'cfg' / 'models',
    Path('yolov13') / 'ultralytics' / 'cfg',
    Path('yolov13'),
]

# infer dataset task
dataset_task=None
if args_data:
    dn = Path(args_data).stem.lower()
    if 'seg' in dn or 'mask' in dn:
        dataset_task='seg'
    elif 'obb' in dn or 'rbox' in dn:
        dataset_task='obb'
    elif 'cls' in dn or 'class' in dn:
        dataset_task='cls'
    else:
        dataset_task='detect'

print('dataset_task=',dataset_task)


def model_matches_task(model_path: Path, task: str) -> bool:
    name = model_path.stem.lower()
    if task == 'cls':
        if 'cls' in name or 'class' in name or 'resnet' in name:
            return True
        if any(k in name for k in ('yolov8', 'detect', 'obb', 'seg')):
            return False
    if task == 'seg':
        if 'seg' in name or 'mask' in name:
            return True
    if task == 'obb':
        if 'obb' in name or 'rot' in name or 'rbox' in name:
            return True
    if task == 'detect':
        if any(k in name for k in ('cls', 'seg', 'pose', 'obb')):
            return False
        if 'yolo' in name or 'yolov8' in name or 'detect' in name or 'rtde' in name:
            return True
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

found=None
for c in candidates:
    print('checking dir',c)
    direct = c / base.name
    print(' direct candidate',direct, 'exists?', direct.exists())
    if direct.exists():
        if dataset_task and model_matches_task(direct, dataset_task):
            found=direct
            break
        else:
            continue
    if c.exists():
        for f in c.rglob('*.yaml'):
            try:
                if (base.stem in f.stem) or (f.stem in base.stem) or f.stem.startswith(base.stem[:6]) or base.stem.startswith(f.stem[:6]):
                    print(' candidate',f)
                    if dataset_task and not model_matches_task(f, dataset_task):
                        print('  rejected by task filter')
                        continue
                    found=f
                    break
            except Exception:
                continue
        if found:
            break
print('found=',found)
