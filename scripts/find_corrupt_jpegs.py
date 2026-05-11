#!/usr/bin/env python
"""扫描数据集（YAML 或图片目录），列出无法打开或验证失败的图片路径。"""
import argparse
import os
from pathlib import Path
import yaml
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def gather_images_from_yaml(yaml_path: str):
    p = Path(yaml_path)
    with open(p, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    imgs = []
    # 常见字段： train, val, path
    for key in ('train', 'val', 'path'):
        if key in cfg:
            val = cfg[key]
            # 如果是相对路径，则以 yaml 文件同级为基准
            base = p.parent
            if isinstance(val, str):
                candidate = (base / val).resolve()
                if candidate.is_dir():
                    for ext in ('*.jpg', '*.jpeg', '*.png'):
                        imgs += list(map(str, candidate.rglob(ext)))
                elif candidate.is_file():
                    imgs.append(str(candidate))
            elif isinstance(val, list):
                for v in val:
                    candidate = (base / v).resolve()
                    if candidate.is_dir():
                        for ext in ('*.jpg', '*.jpeg', '*.png'):
                            imgs += list(map(str, candidate.rglob(ext)))
    return list(dict.fromkeys(imgs))


def gather_images_from_dir(dir_path: str):
    imgs = []
    p = Path(dir_path)
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        imgs += list(map(str, p.rglob(ext)))
    return imgs


def check_images(img_paths):
    bad = []
    for fp in img_paths:
        try:
            with Image.open(fp) as im:
                im.verify()
        except Exception:
            bad.append(fp)
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='dataset yaml or image directory')
    parser.add_argument('--out', default='runs/detect/corrupt_images.txt')
    args = parser.parse_args()
    p = Path(args.data)
    imgs = []
    if p.is_file() and p.suffix in ('.yaml', '.yml'):
        imgs = gather_images_from_yaml(str(p))
    elif p.is_dir():
        imgs = gather_images_from_dir(str(p))
    else:
        print('data argument not a yaml or directory')
        return

    print(f'Found {len(imgs)} images, checking...')
    bad = check_images(imgs)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w', encoding='utf-8') as f:
        for b in bad:
            f.write(b + '\n')
    print(f'Found {len(bad)} corrupt images, saved to {outp}')


if __name__ == '__main__':
    main()
