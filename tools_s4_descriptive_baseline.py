"""Predeclared descriptive animal-equal calcium summaries; no fitting."""
import hashlib
import json
from pathlib import Path
import math
import statistics

INPUT=Path('/home/sensen/Desktop/engine修复验收/S4-calcium-input')
OUTPUT=Path('/home/sensen/Desktop/engine修复验收/S4-descriptive-baseline')


def position_bin(angle):
    if not math.isfinite(angle) or not 0<=angle<=200:return None
    return min(int(angle//10),19)


def summarize(rows, held):
    cells={};outside=0
    for r in rows:
        if r['analyze']!=1 or any(not math.isfinite(r[k]) for k in ('animal_id','calcium','L1C_flex','L1C_flex_vel')):raise ValueError('invalid row')
        animal=int(r['animal_id'])
        if animal!=r['animal_id'] or animal<=0:raise ValueError('invalid animal')
        split='holdout' if animal in held else 'train'
        angle=position_bin(r['L1C_flex'])
        if angle is None:outside+=1
        direction='negative' if r['L1C_flex_vel']<0 else 'positive' if r['L1C_flex_vel']>0 else 'zero'
        for kind,bin in (('position',angle),('direction',direction)):
            if bin is None:continue
            cells.setdefault((split,r['roi'],kind,str(bin)),{}).setdefault(animal,[]).append(r['calcium'])
    result=[]
    for (split,roi,kind,bin),animals in sorted(cells.items()):
        means={str(a):statistics.mean(v) for a,v in sorted(animals.items())}
        result.append(dict(split=split,roi=roi,kind=kind,bin=bin,animal_count=len(means),sample_count=sum(map(len,animals.values())),
                           animal_means=means,mean=statistics.mean(means.values()),animal_sd=statistics.stdev(means.values()) if len(means)>1 else None))
    return dict(out_of_angle_range_samples=outside,cells=result,rows=len(rows))


def main():
    import pyarrow.parquet as pq
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUTPUT.mkdir(exist_ok=False);splits=json.loads((INPUT/'predeclared-split.json').read_text());result={}
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for fileidx,spec in enumerate(splits):
        p=INPUT/spec['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==spec['sha256']
        rows=pq.read_table(p).to_pylist();r=summarize(rows,spec['holdout_animals']);result[spec['file']]=r
        rois=sorted(set(v['roi'] for v in rows))
        for j,roi in enumerate(rois):
            ax=axs[fileidx,j]
            for split in ('train','holdout'):
                cells=sorted([c for c in r['cells'] if c['roi']==roi and c['split']==split and c['kind']=='position'],key=lambda c:int(c['bin']))
                ax.plot([int(c['bin'])*10+5 for c in cells],[c['mean'] for c in cells],'o-',label=split)
            ax.set_title(spec['file'].split('_')[0]+' / '+roi);ax.set_xlabel('Source angle bin center (deg)');ax.set_ylabel('Animal-equal mean original calcium');ax.legend();ax.grid(alpha=.2)
        for j in range(len(rois),3):axs[fileidx,j].axis('off')
    fig.suptitle('Descriptive observations only — no fitted model or holdout prediction')
    fig.tight_layout();fig.savefig(OUTPUT/'position-descriptive.png',dpi=150);plt.close(fig)
    output=dict(files=result,biological_validation=False,calibration_executed=False,policy='fixed10deg bins, exact velocity sign, per-animal equal weighting')
    with (OUTPUT/'summary.json').open('x') as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print('Descriptive summaries published for',sum(v['rows'] for v in result.values()),'unique source rows')


if __name__=='__main__':main()
