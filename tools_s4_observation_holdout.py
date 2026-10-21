"""Frozen train-only observation lookup; exploratory holdout errors, not biology."""
import json
import hashlib
import statistics
import math
from pathlib import Path
from tools_s4_descriptive_baseline import position_bin

INPUT=Path('/home/sensen/Desktop/engine修复验收/S4-calcium-input')
OUT=Path('/home/sensen/Desktop/engine修复验收/S4-observation-holdout')


def checked(rows):
    for r in rows:
        if r['analyze']!=1 or any(type(r[k]) not in (float,int) or not math.isfinite(r[k]) for k in ('animal_id','calcium','L1C_flex')):
            raise ValueError('invalid observation')
        if not 0<r['animal_id']<=2**31-1 or int(r['animal_id'])!=r['animal_id']:raise ValueError('invalid animal')


def fit_train(rows,held):
    checked(rows);groups={};bins={}
    for r in rows:
        a=int(r['animal_id'])
        if a in held:continue
        groups.setdefault(r['roi'],{}).setdefault(a,[]).append(r['calcium'])
        b=position_bin(r['L1C_flex'])
        if b is not None:bins.setdefault((r['roi'],b),{}).setdefault(a,[]).append(r['calcium'])
    if not groups:raise ValueError('no training observations')
    return {roi:dict(constant=statistics.mean(statistics.mean(v) for v in animals.values()),
                      lookup={str(b):statistics.mean(statistics.mean(v) for v in values.values()) for (name,b),values in bins.items() if name==roi},
                      train_animals=sorted(animals)) for roi,animals in sorted(groups.items())}


def evaluate(rows,held,model):
    checked(rows);groups={}
    for r in rows:
        a=int(r['animal_id'])
        if a not in held:continue
        key=(r['roi'],a);g=groups.setdefault(key,dict(total=0,truth=[],lookup=[],constant=[]));g['total']+=1
        b=position_bin(r['L1C_flex']);m=model.get(r['roi'])
        if m is None or b is None or str(b) not in m['lookup']:continue
        g['truth'].append(r['calcium']);g['lookup'].append(m['lookup'][str(b)]);g['constant'].append(m['constant'])
    result=[]
    for (roi,animal),g in sorted(groups.items()):
        row=dict(roi=roi,animal=animal,total=g['total'],predicted=len(g['truth']),uncovered=g['total']-len(g['truth']))
        for name in ('lookup','constant'):
            err=[p-y for p,y in zip(g[name],g['truth'])]
            row[name+'_mae']=statistics.mean(map(abs,err)) if err else None
            row[name+'_rmse']=math.sqrt(statistics.mean(e*e for e in err)) if err else None
        result.append(row)
    return result


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import pyarrow.parquet as pq
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(exist_ok=False);specs=json.loads((INPUT/'predeclared-split.json').read_text());tables={};model={};before={}
    for s in specs:
        p=INPUT/s['file'];before[str(p)]=digest(p);assert before[str(p)]==s['sha256']
        rows=pq.read_table(p).to_pylist();tables[s['file']]=rows;model[s['file']]=fit_train(rows,s['holdout_animals'])
    path=OUT/'frozen-model.json'
    with path.open('x') as f:json.dump(model,f,indent=2,allow_nan=False);f.write('\n')
    identity=digest(path);result={s['file']:evaluate(tables[s['file']],s['holdout_animals'],model[s['file']]) for s in specs}
    assert digest(path)==identity and all(digest(p)==h for p,h in before.items())
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,(name,rows) in zip(axes,result.items()):
        labels=[r['roi']+' / A'+str(r['animal']) for r in rows];x=list(range(len(rows)))
        ax.bar([i-.18 for i in x],[r['lookup_mae'] for r in rows],width=.36,label='position lookup')
        ax.bar([i+.18 for i in x],[r['constant_mae'] for r in rows],width=.36,label='train constant')
        ax.set_xticks(x,labels,rotation=30,ha='right');ax.set_ylabel('Original calcium MAE');ax.set_title(name.split('_')[0]);ax.legend()
    fig.suptitle('Exploratory held-animal errors — NOT blind validation or biological calibration')
    fig.tight_layout();fig.savefig(OUT/'held-animal-errors.png',dpi=150);plt.close(fig)
    with (OUT/'evaluation.json').open('x') as f:json.dump(dict(files=result,input_sha256=before,frozen_model_sha256=identity,model_unchanged=True,biological_validation=False,prior_holdout_exploration=True),f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
