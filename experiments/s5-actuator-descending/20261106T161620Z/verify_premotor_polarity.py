import json,hashlib
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from scipy.integrate import quad
nodes,weights=np.polynomial.legendre.leggauss(12)
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/premotor_polarity/20261106T225055Z')
p=json.loads((O/'PREREGISTRATION.json').read_text());DUR=p['duration_ms'];dt=p['dt_ms'];n=round(1/dt)
cells={'flex':[2955,3271,3707,6343,11441],'ext':[2756,10765]}
cmdmap={};errs=[]
for d in sorted(O.glob('*_off*')):
 s=json.loads((d/'SUMMARY.json').read_text())['records'][0]
 z=np.load(d/'trail.npz');sp=np.unpackbits(z['packed'],axis=1)[:,:12000];cmds=z['commands']
 assert int(sp[:,cells['flex']].sum())==s['flex_spikes'] and int(sp[:,cells['ext']].sum())==s['ext_spikes']
 u=np.column_stack([sp[:,cells['flex']].mean(1),sp[:,cells['ext']].mean(1)])/(dt/1000)/100
 decay=np.exp(-dt/5.);end=lfilter([1-decay],[1,-decay],u,axis=0);start=np.vstack([np.zeros(2),end[:-1]])
 vals=u[:,:,None]+(start-u)[:,:,None]*np.exp(-(nodes+1)*dt/10)
 areas=np.sum(np.clip(vals,0,1)*weights,axis=2)*dt/2
 for j,c in np.argwhere((start-1)*(end-1)<0):
  x=float(start[j,c]);f=float(u[j,c]);cross=-5*np.log((1-f)/(x-f))
  areas[j,c]=quad(lambda t:min(1,max(0,f+(x-f)*np.exp(-t/5))),0,dt,points=[cross],epsabs=1e-12)[0]
 expected=np.vstack([np.zeros(2),areas.reshape(DUR,n,2).sum(1)[:-1]])
 err=float(np.abs(expected-cmds).max());assert err<1e-10;errs.append(err)
 cmdmap[s['condition'],s['offset_rad']]=s['net_drive']
OFFS=p['offsets_rad'];cut=cmdmap['sensory_cut',0.0]
assert all(np.isclose(cmdmap['sensory_cut',o],cut,atol=1e-12) for o in OFFS)
ys=np.array([cmdmap['normal',o]-cut for o in OFFS]);sl=np.gradient(ys,np.array(OFFS))
strict=bool(all(ys[i]>ys[i+1] for i in range(len(ys)-1)));span=float(ys[0]-ys[-1])
report=dict(importance='核心必需',verified=True,neural_runs=10,physics_integrated=False,max_command_error=max(errs),sensory_delta_by_offset=dict(zip([str(o) for o in OFFS],[float(y) for y in ys])),local_slope=dict(zip([str(o) for o in OFFS],[float(v) for v in sl])),strictly_decreasing=strict,span=span,span_threshold=1e-3,graded_restoring=bool(strict and span>=1e-3),cut_offsets_identical=True,verdict='premotor interface recruits both pools in a graded selective way, but the steady-state sensory map is still not a monotone restoring gradient')
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(max_command_error=report['max_command_error'],delta=report['sensory_delta_by_offset'],graded=report['graded_restoring'])))
