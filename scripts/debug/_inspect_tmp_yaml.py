import yaml
p = r"runs\tmp_model_yolov13_n_1778243057.yaml"
with open(p, 'r', encoding='utf-8') as f:
    d = yaml.safe_load(f)
print('scale type:', type(d.get('scale')))
print('scale repr:', repr(d.get('scale')))
print('scales keys:', list(d.get('scales', {}).keys()))
print('scales for scale:', d.get('scales', {}).get(d.get('scale')))
print('top-level keys:', list(d.keys())[:20])
print('sample backbone first entry args:', d.get('backbone', [])[0])
