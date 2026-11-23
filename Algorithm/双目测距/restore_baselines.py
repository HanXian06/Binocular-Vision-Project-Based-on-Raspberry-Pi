"""按需恢复未发布的大批SGBM数组。显式运行才会调用匹配，不改旧统计。"""
from pathlib import Path
import subprocess,sys,json,shutil
BASE=Path(__file__).resolve().parent
configs=json.loads((BASE/'v0.2/configs/dataset_pairs.json').read_text(encoding='utf-8'))
sample=json.loads((BASE/'v0.1/configs/sample_pair0006.json').read_text(encoding='utf-8-sig'))
work=BASE/'v0.1/results/generated_baselines';work.mkdir(exist_ok=True)
for a in configs:
    name=a['part']+'_'+a['pair_id'];target=BASE/'v0.1/results/dataset_evaluation/runs'/name
    if (target/'arrays.npz').exists():continue
    cfg={**sample,'left_image':a['left'],'right_image':a['right'],'camera_parameters':str(BASE/'v0.1/parameters/background1105.npz'),'roi_xywh':a['roi_xywh'],'pair_id':a['part']+'/'+a['pair_id'],'reference_distance_m':a['reference_distance_m']}
    path=work/(name+'.json');path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    result=work/name
    if not (result/'arrays.npz').exists():subprocess.run([sys.executable,str(BASE/'v0.1/run_ranging.py'),'--config',str(path),'--out',str(result)],check=True,stdout=subprocess.DEVNULL)
    target.mkdir(parents=True,exist_ok=True);shutil.copy2(result/'arrays.npz',target/'arrays.npz')
print('基础数组已恢复。原发布统计未改写。')
