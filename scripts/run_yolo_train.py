#!/usr/bin/env python3
import argparse
import multiprocessing
import os
import time
import sys
from pathlib import Path
import yaml

# Ensure the local yolov13/ultralytics package is importable as `ultralytics`.
# This makes `from ultralytics import YOLO` prefer the repository copy.
repo_root = Path(__file__).resolve().parent.parent
yolov13_path = str(repo_root / 'yolov13')
if yolov13_path not in sys.path:
	sys.path.insert(0, yolov13_path)

from ultralytics import YOLO


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='cfg/models/v8/yolov8n.yaml')
	parser.add_argument('--data', type=str, required=True)
	parser.add_argument('--write-tmp-only', action='store_true', help='Write temporary model YAML and exit')
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--imgsz', type=int, default=640)
	parser.add_argument('--batch', type=int, default=8)
	parser.add_argument('--name', type=str, default='autoresearch_run')
	parser.add_argument('--workers', type=int, default=0)
	parser.add_argument('--scale', type=str, default=None, help='Model scale (n, s, m, l, x) for compound-scaled YAMLs')
	parser.add_argument('--time-budget', type=int, default=864000)
	parser.add_argument('--patience', type=int, default=None,
					help='Early stopping patience (epochs)')
	# Additional hyperparameters commonly used in suggestions
	parser.add_argument('--lr', type=float, default=None, help='learning rate')
	parser.add_argument('--optimizer', type=str, default=None, help='optimizer name')
	parser.add_argument('--mosaic', action='store_true', help='enable mosaic augmentation')
	parser.add_argument('--augment', action='store_true', help='enable augmentation')
	# additional hyperparameters exposed for autoresearch suggestions
	parser.add_argument('--degrees', type=float, default=None, help='rotation degrees for augmentation')
	parser.add_argument('--topk', type=int, default=None, help='TAL topk to consider')
	parser.add_argument('--tal-alpha', type=float, default=None, help='TAL alpha parameter')
	parser.add_argument('--tal-beta', type=float, default=None, help='TAL beta parameter')
	parser.add_argument('--crazing-boost', type=float, default=None, help='focused TAL boost for crazing class')
	parser.add_argument('--crazing-class-index', type=int, default=None, help='class index for crazing (default 0)')
	parser.add_argument('--save-period', type=int, default=None, help='save period (epochs)')
	args = parser.parse_args()

	# If a compound-model YAML supports 'scales' (family-level), create a temporary
	# model YAML with the requested scale key so the loader picks the correct depth/width.
	model_path = args.model
	if args.scale is not None:
		try:
			import yaml as _yaml
			p_model = Path(args.model)
			orig = None
			with p_model.open('r', encoding='utf-8') as f:
				orig = _yaml.safe_load(f) or {}
			# If the model YAML doesn't declare 'scales', try to inherit from a family YAML
			if not isinstance(orig, dict):
				orig = {}
			if 'scales' not in orig:
				# family token before first '_' in stem, e.g. 'yolov13_nmsfree' -> 'yolov13'
				family = p_model.stem.split('_')[0]
				sibling = p_model.parent / f"{family}.yaml"
				if sibling.exists():
					try:
						with sibling.open('r', encoding='utf-8') as sf:
							fam = _yaml.safe_load(sf) or {}
							if isinstance(fam, dict) and 'scales' in fam and isinstance(fam['scales'], dict):
								orig['scales'] = fam['scales']
					except Exception:
						pass
			# set scale key (string like 'n','m','s','l' expected by model YAML)
			orig['scale'] = args.scale
			# prefer to base tmp filename on provided run name so tmp YAML and
			# training run directory share an identifying token. If no name is
			# provided, fall back to inserting scale into the model stem.
			import re as _re
			if getattr(args, 'name', None):
				# sanitize name for filesystem (allow letters, digits, dash, underscore)
				label = _re.sub(r'[^A-Za-z0-9_-]', '', args.name)
				# ensure scale token is present in label; if not, try to insert
				if args.scale and not _re.search(r'yolov\d+[nslmx]', label, flags=_re.IGNORECASE):
					# attempt to insert scale after yolov<digits> if present in label
					if _re.search(r'(yolov\d+)', label, flags=_re.IGNORECASE):
						label = _re.sub(r'(yolov\d+)', r"\1" + args.scale, label, flags=_re.IGNORECASE)
				# final tmp stem uses label
				tmp_stem = label
			else:
				stem = p_model.stem
				if _re.search(r'yolov\d+[nslmx]', stem, flags=_re.IGNORECASE):
					new_stem = stem
				else:
					new_stem = _re.sub(r'(yolov\d+)', r"\1" + args.scale, stem, flags=_re.IGNORECASE)
				if '_' in new_stem:
					new_stem = new_stem.replace('_', '-', 1)
				tmp_stem = new_stem
			# create tmp yaml: if tmp_stem already ends with a timestamp
			# (e.g., provided by the caller like 'phase1-...-1778287520'),
			# do not append a second timestamp. This keeps names like
			# 'phase1-yolov13n-1778287520.yaml' instead of
			# 'phase1-yolov13n-1778287520_1778287525.yaml'.
			import re as _re_internal
			if _re_internal.search(r'[_-]\d{6,}$', tmp_stem):
				tmp = Path('runs') / f'{tmp_stem}.yaml'
			else:
				tmp = Path('runs') / f'{tmp_stem}_{int(time.time())}.yaml'
			tmp.parent.mkdir(parents=True, exist_ok=True)
			with tmp.open('w', encoding='utf-8') as f:
				_yaml.safe_dump(orig, f)
			model_path = str(tmp)
			# Ensure training run name maps to the tmp YAML basename so directories
			# created by YOLO.train are easy to trace back to the tmp YAML.
			# We set args.name (if present) to the tmp stem for consistency.
			try:
				tmp_basename = Path(model_path).stem
				# only override when a name was provided or to ensure strict mapping
				args.name = tmp_basename
			except Exception:
				pass
			# Print a short verification of the temporary YAML so callers can confirm
			try:
				_print_scales = {k: v for k, v in (orig.get('scales') or {}).items()}
			except Exception:
				_print_scales = orig.get('scales')
			print(f"WROTE tmp model yaml: {model_path}")
			print(f"tmp yaml scale={orig.get('scale')}, scales keys={list((_print_scales or {}).keys())}")
		except Exception:
			# fallback to original model if any error
			model_path = args.model

	# If requested, only write the temporary YAML and exit before heavy imports/instantiation.
	if getattr(args, 'write_tmp_only', False):
		print('DRY_RUN: tmp model yaml at', model_path)
		# print a short confirm of intended training name
		try:
			print('DRY_RUN: intended run name:', args.name or Path(model_path).stem)
		except Exception:
			pass
		return

	model = YOLO(model_path)
	# Use provided args.name (overridden earlier to match tmp yaml stem) or
	# fallback to the model YAML stem so training folder matches the YAML file.
	try:
		effective_name = args.name or Path(model_path).stem
	except Exception:
		effective_name = args.name or Path(model_path).stem
	print(f"Starting training: model={model_path}, data={args.data}, epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, workers={args.workers}, name={effective_name}, patience={args.patience}")
	train_kwargs = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, name=effective_name, workers=args.workers)
	# optional params: add only when provided to avoid passing None
	if args.patience is not None:
		train_kwargs['patience'] = args.patience
	if args.lr is not None:
		# Ultralytics trainer expects initial learning rate key 'lr0'
		train_kwargs['lr0'] = args.lr
	if args.optimizer is not None:
		train_kwargs['optimizer'] = args.optimizer
	# augmentation flags
	if args.augment:
		train_kwargs['augment'] = True
	if args.mosaic:
		train_kwargs['mosaic'] = True
	# save period
	if args.save_period is not None:
		train_kwargs['save_period'] = args.save_period

	# export selected hp to environment so internal modules (e.g., TAL) can read them
	# only set when provided to avoid overwriting defaults
	if args.topk is not None:
		os.environ['TOPK'] = str(args.topk)
	if args.tal_alpha is not None:
		os.environ['TAL_ALPHA'] = str(args.tal_alpha)
	if args.tal_beta is not None:
		os.environ['TAL_BETA'] = str(args.tal_beta)
	if args.crazing_boost is not None:
		os.environ['CRAZING_BOOST'] = str(args.crazing_boost)
	if args.crazing_class_index is not None:
		os.environ['CRAZING_CLASS_INDEX'] = str(args.crazing_class_index)
	if args.degrees is not None:
		os.environ['AUG_DEGREES'] = str(args.degrees)

	# validate and sanitize train kwargs before calling YOLO.train
	def sanitize_train_kwargs(kwargs: dict) -> dict:
		# Known keys supported by ultralytics trainer (common subset)
		allowed = {
			'data', 'epochs', 'imgsz', 'batch', 'name', 'workers', 'patience',
			'lr0', 'lrf', 'optimizer', 'augment', 'mosaic', 'save_period',
			'project', 'exist_ok', 'device'
		}
		clean = {}
		for k, v in kwargs.items():
			if k in allowed:
				clean[k] = v
			else:
				print(f"Warning: dropping unsupported train arg: {k}")
		return clean

	train_kwargs = sanitize_train_kwargs(train_kwargs)

	# call training
	model.train(**train_kwargs)
	print('Training finished')


if __name__ == '__main__':
	multiprocessing.freeze_support()
	main()