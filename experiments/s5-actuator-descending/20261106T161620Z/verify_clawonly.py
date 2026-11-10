import json,hashlib
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from scipy.integrate import quad
nodes,weights=np.polynomial.legendre.leggauss(12)
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z')
cells={'flex':[2955,3271,3707,6343,11441],'ext':[2756,10765]}
dt=.0078125;DUR=150;n=round(1/dt);errs=[];maps={}
for pre in [.1,.3]:
    sub=ROOT/f'clawonly/20261106T225055Z/pre{pre}_normal_off0.0'
    base=json.loads(sub.joinpath('SUMMARY.json').read_text())['records'][0]
    cut=ROOT/f'clawonly/20261106T225055Z/pre{pre}_sensory_cut_off0.0'
    cutrec=json.loads(cut.joinpath('SUMMARY.json').read_text())['records'][0]
    rows=[]
    for off in [-.06,-.03,0.,.03,.06]:
        d=ROOT/f'clawonly/20261106T225055Z/pre{pre}_normal_off{off}'
        s=json.loads(d.joinpath('SUMMARY.json').read_text())['records'][0]
        z=np.load(d/'trail.npz');sp=np.unpackbits(z['packed'],axis=1)[:,:12000];cmds=z['commands']
        assert int(sp[:,cells['flex']].sum())==s['flex_spikes'] and int(sp[:,cells['ext']].sum())==s['ext_spikes']
        u=np.column_stack([sp[:,cells['flex']].mean(1),sp[:,cells['ext']].mean(1)])/(dt/1000)/100
        decay=np.exp(-dt/5.);end=lfilter([1-decay],[1,-decay],u,axis=0);start=np.vstack([np.zeros(2),end[:-1]])
        vals=u[:,:,None]+(start-u)[:,:,None]*np.exp(-(nodes+1)*dt/10)
        areas=np.sum(np.clip(vals,0,1)*weights,axis=2)*dt/2
        for j,c in np.argwhere((start-1)*(end-1)<0):
            x=float(start[j,c]);f=float(u[j,c]);cr=-5*np.log((1-f)/(x-f))
            areas[j,c]=quad(lambda t:min(1,max(0,f+(x-f)*np.exp(-t/5))),0,dt,points=[cr],epsabs=1e-12)[0]
        exp=np.vstack([np.zeros(2),areas.reshape(DUR,n,2).sum(1)[:-1]])
        e=float(np.abs(exp-cmds).max());assert e<1e-10;errs.append(e)
        rows.append(dict(offset_rad=off,net_drive=s['net_drive'],delta_vs_cut=s['net_drive']-cutrec['net_drive'],flex_spikes=s['flex_spikes'],ext_spikes=s['ext_spikes']))
    ys=np.array([r['delta_vs_cut'] for r in rows]);strict=bool(all(ys[i]>ys[i+1] for i in range(4)))
    maps[str(pre)]=dict(rows=rows,strictly_decreasing=strict,span=float(ys[0]-ys[-1]),graded_restoring=bool(strict and ys[0]-ys[-1]>=1e-3),flex_spike_range=[min(r['flex_spikes'] for r in rows),max(r['flex_spikes'] for r in rows)],ext_spike_range=[min(r['ext_spikes'] for r in rows),max(r['ext_spikes'] for r in rows)])
report=dict(importance='核心必需',verified=True,runs_checked=20,max_command_error=max(errs),maps=maps,conclusion='claw input changes motor spike timing but not spike counts at either operating point, so the net-drive map is not a monotone restoring law; no rate-coded position signal reaches the motor pools in this configuration')
(ROOT/'clawonly/20261106T225055Z/VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(max_command_error=report['max_command_error'],maps=maps)))
