"""Read-only audit of the two received Mamiya2018 calcium tables, no fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

INPUT=Path('/home/sensen/Desktop/engine修复验收/S4-calcium-input')
OUTPUT=Path('/home/sensen/Desktop/engine修复验收/S4-actual-calcium-audit')
NAMES=('claw_magnet_Mamiya2018.parquet','hook_flexion_01_magnet_Mamiya2018.parquet')


def audit(name):
    p=INPUT/name;table=pq.read_table(p);rows=table.to_pylist();groups={}
    for row in rows:
        key=(row['animal_id'],row['trial'],row['roi'])
        groups.setdefault(key,[]).append(row)
    details=[]
    for key,group in groups.items():
        t=np.array([r['time'] for r in group]);dt=np.diff(t)
        details.append(dict(animal_id=key[0],trial=key[1],roi=key[2],samples=len(group),
                            time_start_s=float(t[0]),time_end_s=float(t[-1]),strictly_increasing=bool(np.all(dt>0)),
                            dt_min_s=float(dt.min()) if len(dt) else None,dt_max_s=float(dt.max()) if len(dt) else None,
                            stimulus_types=sorted(set(r['stimulus_type'] for r in group))))
    numeric={}
    for column in ('animal_id','frame','time','analyze','calcium','L1C_flex','L1C_flex_vel'):
        v=np.array([r[column] if r[column] is not None else np.nan for r in rows],float)
        numeric[column]=dict(finite_count=int(np.isfinite(v).sum()),null_count=table[column].null_count,
                             min=float(np.nanmin(v)),max=float(np.nanmax(v)))
    return dict(file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=len(rows),
                schema=str(table.schema),drivers=sorted(set(r['driver'] for r in rows)),
                rois=sorted(set(r['roi'] for r in rows)),animals=sorted(set(r['animal_id'] for r in rows)),
                trial_names=len(set(r['trial'] for r in rows)),group_count=len(groups),groups=details,
                numeric=numeric,analyze_counts={str(v):sum(r['analyze']==v for r in rows) for v in sorted(set(r['analyze'] for r in rows))},
                predicted_columns=[c for c in table.column_names if 'predict' in c],
                raw_identity_is_banc=False,calibration_executed=False),groups


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUTPUT.mkdir(exist_ok=False)
    receipt=json.loads((INPUT/'receipt.json').read_text());expected={Path(v['path']).name:v['sha256'] for v in receipt['files']}
    result={};fig,axs=plt.subplots(2,2,figsize=(12,7));examples={}
    for index,name in enumerate(NAMES):
        info,groups=audit(name);assert info['sha256']==expected[name];result[name]=info
        key=next(iter(groups));group=groups[key];t=[r['time'] for r in group]
        axs[0,index].plot(t,[r['L1C_flex'] for r in group]);axs[0,index].set_ylabel('Recorded joint angle (deg)')
        axs[1,index].plot(t,[r['calcium'] for r in group],color='tab:orange');axs[1,index].set_ylabel('Recorded calcium (original scale)')
        axs[0,index].set_title(('Claw' if index==0 else 'Hook flexion')+' / '+str(key))
        for ax in axs[:,index]:ax.set_xlabel('Recorded time (s)');ax.grid(alpha=.2)
        examples[name]=dict(group=list(key),selection='first source-order animal/trial/ROI; not performance selected')
    fig.suptitle('Received biological calcium observations — NOT spikes or model predictions')
    fig.tight_layout();fig.savefig(OUTPUT/'actual-observations.png',dpi=150);plt.close(fig)
    with (OUTPUT/'audit.json').open('x') as f:json.dump(dict(files=result,plot_examples=examples,biological_validation=False,scope='asset/schema/time/group audit only'),f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({n:{k:r[k] for k in ('rows','group_count','animals','rois','predicted_columns')} for n,r in result.items()},indent=2))


if __name__=='__main__':main()
