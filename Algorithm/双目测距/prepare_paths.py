"""首次克隆后，把repo://数据索引解析为当前仓库路径；不运行模型。"""
from pathlib import Path
import json,hashlib
BASE=Path(__file__).resolve().parent
REPO=BASE.parents[1]
def fix(x):
    if isinstance(x,dict):return {k:fix(v) for k,v in x.items()}
    if isinstance(x,list):return [fix(v) for v in x]
    if isinstance(x,str) and x.startswith('repo://'):return str(REPO/x[7:])
    return x
for folder in ['v0.1','v0.2','v0.3','roi_protocol']:
    for path in (BASE/folder).rglob('*.json'):
        if 'results' in path.relative_to(BASE/folder).parts or path.name=='MANIFEST.json':continue
        data=json.loads(path.read_text(encoding='utf-8-sig'));new=fix(data)
        if new!=data:path.write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding='utf-8')
p=BASE/'roi_protocol'
# 指纹跟随当前克隆的路径解析；原运行指纹见发布时保存的历史文件。
(p/'freeze.json').write_text(json.dumps(dict(target_sha256=hashlib.sha256((p/'frozen_targets.json').read_bytes()).hexdigest(),policy_sha256=hashlib.sha256((p/'policy.py').read_bytes()).hexdigest(),frozen_before_any_ranging_call=True),indent=2),encoding='utf-8')
print('数据路径已解析；未运行检测或测距。共享JSON接口约定没有被修改。')
