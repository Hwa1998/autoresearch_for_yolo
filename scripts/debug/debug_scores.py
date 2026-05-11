from pathlib import Path
base=Path('cfg/models/v8/yolov8n.yaml')
base_st=base.stem
candidates=[Path('yolov13')/'ultralytics'/'cfg'/'models'/'v8']
scored=[]
for c in candidates:
    if not c.exists():
        continue
    for f in c.rglob('*.yaml'):
        name=f.stem.lower()
        bs=base_st.lower()
        score=0
        if name==bs:
            score+=200
        if bs.startswith(name):
            score+=150
        if name in bs or bs in name:
            score+=100
        if name.startswith(bs[:6]) or bs.startswith(name[:6]):
            score+=50
        # inspect header
        try:
            header='\n'.join(f.read_text(encoding='utf-8').splitlines()[:20]).lower()
            if 'tasks/detect' in header:
                score+=20
        except Exception:
            pass
        if score>0:
            scored.append((score,f))
scored.sort(key=lambda x:x[0],reverse=True)
for s,f in scored[:10]:
    print(s,f)
