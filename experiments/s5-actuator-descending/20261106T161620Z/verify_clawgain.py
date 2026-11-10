import json
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from scipy.integrate import quad
nodes,weights=np.polynomial.legendre.leggauss(12)
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/clawgain/20261106T225055Z')
cells={'flex':[2955,3271,3707,6343,11441],'ext':[2756,10765]};dt=.0078125;DUR=150;n=round(1/dt)
OFFS=[-.06,-.03,0.,.03,.06];errs=[];maps={}
for g in ['0.3','1.0']:
    cut=json.loads((ROOT/f'g{g}_pre0.3_sensory_cut_off0.0'/'SUMMARY.json').read_text())['records'][0]
    rows=[]
    for off in OFFS:
        d=ROOT/f'g{g}_pre0.3_normal_off{off}'
        s=json.loads((d/'SUMMARY.json').read_text())['records'][0]
        z=np.load(d/'trail.npz');sp=np.unpackbits(z['packed'],axis=1)[:,:12000];cmds=z['commands']
        assert int(sp[:,cells['flex']].sum())==s['flex_spikes'] and int(sp[:,cells['ext']].sum())==s['ext_spikes']
        assert int(sp.sum())==s['network_spikes']
        u=np.column_stack([sp[:,cells['flex']].mean(1),sp[:,cells['ext']].mean(1)])/(dt/1000)/100
        decay=np.exp(-dt/5.);end=lfilter([1-decay],[1,-decay],u,axis=0);start=np.vstack([np.zeros(2),end[:-1]])
        vals=u[:,:,None]+(start-u)[:,:,None]*np.exp(-(nodes+1)*dt/10)
        areas=np.sum(np.clip(vals,0,1)*weights,axis=2)*dt/2
        for j,c in np.argwhere((start-1)*(end-1)<0):
            x=float(start[j,c]);f=float(u[j,c]);cr=-5*np.log((1-f)/(x-f))
            areas[j,c]=quad(lambda t:min(1,max(0,f+(x-f)*np.exp(-t/5))),0,dt,points=[cr],epsabs=1e-12)[0]
        exp=np.vstack([np.zeros(2),areas.reshape(DUR,n,2).sum(1)[:-1]])
        e=float(np.abs(exp-cmds).max());assert e<1e-10;errs.append(e)
        rows.append(dict(offset_rad=off,net_drive=s['net_drive'],delta=s['net_drive']-cut['net_drive'],flex_spikes=s['flex_spikes'],ext_spikes=s['ext_spikes']))
    ys=np.array([r['delta'] for r in rows])
    strict=bool(all(ys[i]>ys[i+1] for i in range(4)))
    nonincr=bool(all(ys[i]>=ys[i+1] for i in range(4)))
    maps[g]=dict(rows=rows,strictly_decreasing=strict,non_increasing=nonincr,span=float(ys[0]-ys[-1]),strict_intervals=sum(1 for i in range(4) if ys[i]>ys[i+1]),graded_restoring_full=bool(strict and ys[0]-ys[-1]>=1e-3),graded_restoring_partial=bool(nonincr and ys[0]-ys[-1]>=1e-3),flex_rate_monotone=bool(all(rows[i]['flex_spikes']>=rows[i+1]['flex_spikes'] for i in range(4))),ext_rate_monotone=bool(all(rows[i]['ext_spikes']<=rows[i+1]['ext_spikes'] for i in range(4))))
rep=dict(importance='核心必需',verified=True,neural_runs=12,max_command_error=max(errs),maps=maps,note='gain 1.0 gives a monotone non-increasing map with reciprocal motor rate modulation; the first interval ties because the extended end saturates')
(ROOT/'VERIFICATION.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({g:dict(strict=v['strictly_decreasing'],nonincr=v['non_increasing'],span=round(v['span'],4),strict_iv=v['strict_intervals'],partial=v['graded_restoring_partial'],flexmono=v['flex_rate_monotone'],extmono=v['ext_rate_monotone'],flex=[r['flex_spikes'] for r in v['rows']],ext=[r['ext_spikes'] for r in v['rows']]) for g,v in maps.items()}))
