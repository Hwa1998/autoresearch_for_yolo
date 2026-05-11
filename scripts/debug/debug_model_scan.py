from pathlib import Path
p=Path('yolov13')/ 'ultralytics' / 'cfg' / 'models' / 'v8'
print('dir',p.exists())
for f in sorted(p.glob('*.yaml')):
    s=f.stem
    h='\n'.join(f.read_text(encoding='utf-8').splitlines()[:20]).lower()
    print(s, '-> header contains detect?', 'tasks/detect' in h)
