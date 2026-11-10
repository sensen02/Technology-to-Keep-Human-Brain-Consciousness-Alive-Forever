import json,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from scipy.integrate import quad
nodes,weights=np.polynomial.legendre.leggauss(12)
O=Path(sys.argv[1]);p=json.loads((O/'PREREGISTRATION.json').read_text());s=json.loads((O/'SUMMARY.json').read_text());DUR=p['duration_ms'];checks=[]
audit=json.loads(Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/descending/PATH_AUDIT.json').read_text());cells=audit['targets']['1']+audit['targets']['-1']
for r in s['records']:
 z=np.load(O/(r['tag']+'.npz'));sp=np.unpackbits(z['packed'],axis=1)[:,:12000];cmds=z['commands'];dt=r['bias_nA']*0+p['dt_ms'];n=round(1/dt)
 assert int(sp.sum())==r['spikes']
 u=np.column_stack([sp[:,cells[:5]].mean(1),sp[:,cells[5:]].mean(1)])/(dt/1000)/100
 decay=np.exp(-dt/5.);end=lfilter([1-decay],[1,-decay],u,axis=0);start=np.vstack([np.zeros(2),end[:-1]])
 vals=u[:,:,None]+(start-u)[:,:,None]*np.exp(-(nodes+1)*dt/10)
 areas=np.sum(np.clip(vals,0,1)*weights,axis=2)*dt/2
 for j,c in np.argwhere((start-1)*(end-1)<0):
  x=float(start[j,c]);f=float(u[j,c]);cross=-5*np.log((1-f)/(x-f))
  areas[j,c]=quad(lambda t:min(1,max(0,f+(x-f)*np.exp(-t/5))),0,dt,points=[cross],epsabs=1e-12)[0]
 expected=np.vstack([np.zeros(2),areas.reshape(DUR,n,2).sum(1)[:-1]])
 err=float(np.abs(expected-cmds).max());assert err<1e-10
 tail=cmds[-25:];assert abs(float((tail[:,0]-tail[:,1]).mean())-r['net_drive'])<1e-12
 checks.append(dict(tag=r['tag'],max_command_error=err))
for k,v in p['hashes'].items():assert hashlib.sha256(Path(k).read_bytes()).hexdigest()==v
report=dict(importance='核心必需',verified=True,max_command_error=max(c['max_command_error'] for c in checks),checks=len(checks),physics_integrated=False)
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(verified=True,max_command_error=report['max_command_error'],checks=len(checks))))
