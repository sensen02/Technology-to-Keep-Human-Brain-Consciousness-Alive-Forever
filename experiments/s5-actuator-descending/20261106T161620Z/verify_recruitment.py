"""Verify the C1-C5 recruitment/routing runs: recompute every reported count from stored spikes."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z')
audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());flex=audit['targets']['1'];ext=audit['targets']['-1']
dirs=[]
for stage in ['recruit','recruit2','reflex','premotor','routing']:
    dirs+=sorted((ROOT/stage/'20261106T225055Z').glob('*'))
dirs=[d for d in dirs if (d/'SUMMARY.json').exists()]
checks=[]
for d in dirs:
    s=json.loads((d/'SUMMARY.json').read_text())
    for r in s['records']:
        f=d/'trail.npz'
        if not f.exists(): f=d/'normal_off0.0.npz'
        packed=np.load(f)['packed'];sp=np.unpackbits(packed,axis=1)[:,:12000];tail=sp[round(100/.0078125):]
        def g(*names):
            for n in names:
                if n in r: return r[n]
            raise KeyError(names)
        assert int(sp[:,flex].sum())==g('flex_spikes'),d.name
        assert int(sp[:,ext].sum())==g('ext_spikes'),d.name
        assert int(tail[:,flex].sum())==g('flex_tail','flex_tail_spikes'),d.name
        assert int(tail[:,ext].sum())==g('ext_tail','ext_tail_spikes'),d.name
        assert int(sp.sum())==g('network_spikes'),d.name
        assert np.array_equal(sp[:,flex].sum(0),np.array(r['flex_counts']))
        assert np.array_equal(sp[:,ext].sum(0),np.array(r['ext_counts']))
        checks.append(dict(run=d.name,flex=g('flex_spikes'),ext=g('ext_spikes'),network=g('network_spikes')))
sel=[]
for pool in ['flex','ext']:
    row={}
    for a in ['0.05','0.1','0.3']:
        s=json.loads((ROOT/f'premotor/20261106T225055Z/{pool}_a{a}'/'SUMMARY.json').read_text())['records'][0]
        row[a]=dict(flex=s['flex_spikes'],ext=s['ext_spikes'])
    mono=bool(row['0.05'][pool]<=row['0.1'][pool] and row['0.1'][pool]<=row['0.3'][pool] and row['0.3'][pool]>row['0.05'][pool])
    other=[k for k in ['flex','ext'] if k!=pool][0]
    selective=all(row[a][other]==0 for a in row)
    sel.append(dict(pool=pool,by_current=row,monotone_graded=bool(mono),other_pool_silent=bool(selective),usable=bool(mono and selective and row['0.3'][pool]>0)))
report=dict(importance='核心必需',verified=True,runs_checked=len(checks),checks=checks,premotor_selection=sel,all_verified=True)
(ROOT/'premotor/20261106T225055Z/VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(runs_checked=len(checks),premotor=sel)))
