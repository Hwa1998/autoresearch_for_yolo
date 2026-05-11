"""
Safe smoke-run for yolov13 integration.
Tries to import key modules and reports import errors without raising.
Does NOT execute training or heavy operations.
"""
import importlib
import traceback
import sys

ROOT = '.'
sys.path.insert(0, ROOT)

modules = [
    'yolov13',
    'yolov13.train',
    'yolov13.predict',
    'yolov13.ultralytics',
    'train_yolov13',
    'prepare_detection',
]

results = {}
for mod in modules:
    try:
        importlib.invalidate_caches()
        m = importlib.import_module(mod)
        results[mod] = ('ok', None)
    except Exception as e:
        tb = traceback.format_exc()
        results[mod] = ('error', tb)

print('\nSMOKE-RUN RESULTS:\n')
for mod, (status, info) in results.items():
    if status == 'ok':
        print(f'[OK]     {mod}')
    else:
        first_line = info.splitlines()[0]
        print(f'[ERROR]  {mod} -> {first_line}')

print('\nDetailed errors written to smoke_yolov13_errors.log')
with open('smoke_yolov13_errors.log', 'w', encoding='utf-8') as f:
    for mod, (status, info) in results.items():
        f.write(f'MODULE: {mod}\nSTATUS: {status}\n')
        if info:
            f.write(info + '\n')

if any(status == 'error' for status, _ in results.values()):
    print('\nSmoke-run detected import errors; check smoke_yolov13_errors.log for details.')
else:
    print('\nSmoke-run succeeded: key modules importable.')
