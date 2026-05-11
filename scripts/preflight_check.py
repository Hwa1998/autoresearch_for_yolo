import sys, os, importlib, yaml
print('python', sys.executable)
print('cwd', os.getcwd())
ds = r'D:\DingReihwa\6_AI\datasets\yolo_datasets\detect\dingweixiao\dingweixiao.yaml'
print('dataset_exists', os.path.exists(ds))
try:
    m = importlib.import_module('ultralytics')
    print('ultralytics_file', getattr(m, '__file__', None))
except Exception as e:
    print('ultralytics_import_error', e)
cfg = 'configs/autoresearch_config.yaml'
print('config_exists', os.path.exists(cfg))
if os.path.exists(cfg):
    data = yaml.safe_load(open(cfg))
    print('candidates=', data.get('candidates'))
else:
    print('no config')
