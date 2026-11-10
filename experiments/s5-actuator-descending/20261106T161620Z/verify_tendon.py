import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=ROOT/'tendon_muscle';p=json.loads((O/'PREREGISTRATION.json').read_text());s=json.loads((O/'SUMMARY.json').read_text())
l0=np.array(p['l0']);fmax=np.array(p['fmax']);errors=[];dirs={}
for name,target in p['conditions'].items():
 a=np.load(O/(name+'.npz'))['rows'];activation=np.zeros(2);acts=[]
 for k in range(len(a)):
  command=np.array(target) if k*.0000125<.03 else np.zeros(2);tau=np.where(command>activation,.01,.03);activation=command+(activation-command)*np.exp(-.0000125/tau);acts.append(activation.copy())
 assert np.array_equal(acts,a[:,3:5])
 lam=a[:,5:7]/l0;v=a[:,7:9]/(10*l0);fl=np.exp(-((lam-1)/.25)**2);fv=np.ones_like(v);short=v<0
 fv[short]=np.maximum(0,1+v[short])/(1-4*v[short]);fv[~short]=1+.5*v[~short]/(v[~short]+.25)
 passive=.05*(np.maximum(lam-1,0)/.1)**2;viscous=.02*np.maximum(v,0);force=np.clip(fmax*(a[:,3:5]*fl*fv+passive+viscous),0,fmax)
 err=float(abs(force-a[:,9:11]).max());assert err<1e-12;errors.append(err);assert np.isfinite(a).all() and (a[:,1]>=.4789).all() and (a[:,1]<=2.502).all()
 dirs[name]=float(a[2399,1]-1.2246235578147273)
assert dirs['flex']>dirs['zero']>dirs['ext'];assert dirs['zero']==0
src=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-actuator-descending/20261106T161620Z/tendon_muscle.py')
assert hashlib.sha256(src.read_bytes()).hexdigest()==p['source_sha256']
report=dict(importance='核心必需',verified=True,physical_runs=4,max_force_reconstruction_error=max(errors),activation_bitwise=True,pulse_end_shifts=dirs,scope='declared engineering muscle law and preserved tendon geometry, no biological calibration or neural support')
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
