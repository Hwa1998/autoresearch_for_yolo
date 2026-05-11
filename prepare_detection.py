"""
Prepare detection dataset for `yolov13` experiments.

Creates a dataset YAML compatible with Ultralytics/YOLO from a directory
containing images and labels in the common YOLO TXT format:

 expected layout:
   dataset_root/
     images/
       train/
       val/
     labels/
       train/
       val/

Usage examples:
  python prepare_detection.py --dataset-dir ./data/mydataset
  python prepare_detection.py --dataset-dir yolov13/datasets/mydata --output yolov13/cfg/datasets/mydata.yaml

The script will try to detect number of classes from label files or an optional classes file.
"""
from pathlib import Path
import argparse
import yaml
import sys


def detect_num_classes_and_names(labels_dir):
    # labels_dir: Path pointing to labels/train or labels root
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        return 0, []
    max_id = -1
    for txt in labels_dir.rglob('*.txt'):
        try:
            for line in txt.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                cid = int(parts[0])
                if cid > max_id:
                    max_id = cid
        except Exception:
            continue
    if max_id < 0:
        return 0, []
    nc = max_id + 1
    names = [f'class{i}' for i in range(nc)]
    return nc, names


def read_names_file(path):
    p = Path(path)
    if not p.exists():
        return None
    lines = [l.strip() for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
    return lines if lines else None


def main():
    parser = argparse.ArgumentParser(description='Prepare Ultralytics dataset YAML for YOLO training')
    parser.add_argument('--dataset-dir', type=Path, required=True, help='Root of dataset (images/, labels/ subfolders)')
    parser.add_argument('--output', type=Path, default=None, help='Output YAML path')
    parser.add_argument('--names-file', type=Path, default=None, help='Optional file with class names (one per line)')
    parser.add_argument('--names', type=str, default=None, help='Optional comma-separated class names')
    args = parser.parse_args()

    ds = args.dataset_dir.resolve()
    if not ds.exists():
        print(f"Dataset dir not found: {ds}")
        sys.exit(1)

    images_train = ds / 'images' / 'train'
    images_val = ds / 'images' / 'val'
    labels_train = ds / 'labels' / 'train'
    labels_val = ds / 'labels' / 'val'

    if not images_train.exists() or not images_val.exists():
        print('Warning: expected images/train and images/val. Falling back to images/ if present.')
        if (ds / 'images').exists():
            images_train = ds / 'images'
            images_val = ds / 'images'
        else:
            print('No images found. Aborting.')
            sys.exit(1)

    if not labels_train.exists() or not labels_val.exists():
        print('Warning: labels/train or labels/val not found; attempting to detect labels in dataset root.')
        if (ds / 'labels').exists():
            labels_train = ds / 'labels'
            labels_val = ds / 'labels'

    # determine classes
    names = None
    if args.names_file:
        names = read_names_file(args.names_file)
    if names is None and args.names:
        names = [n.strip() for n in args.names.split(',') if n.strip()]
    if names is None:
        nc, names = detect_num_classes_and_names(labels_train)
    else:
        nc = len(names)

    if nc == 0:
        print('Could not detect any classes. Create a classes file or check label files.')
        sys.exit(1)

    # build yaml
    out_yaml = args.output or (Path('yolov13') / 'cfg' / 'datasets' / (ds.name + '.yaml'))
    out_yaml.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'train': str(images_train.as_posix()),
        'val': str(images_val.as_posix()),
        'nc': int(nc),
        'names': names,
    }

    with open(out_yaml, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f'Wrote dataset yaml to: {out_yaml}')
    print(f"  train: {data['train']}")
    print(f"  val:   {data['val']}")
    print(f"  nc:    {data['nc']}")


if __name__ == '__main__':
    main()
