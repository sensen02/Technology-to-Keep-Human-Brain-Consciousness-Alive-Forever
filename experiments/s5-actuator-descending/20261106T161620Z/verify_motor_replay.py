import json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/motor_replay/20261106T192549Z');p=json.loads((O/'PREREGISTRATION.json').read_text());s=json.loads((O/'SUMMARY.json').read_text());cells=p['cells'];A=Path('/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z/ARMS/A');we=sparse.load_npz(A/'ACTUAL_We.npz').tocsr()[cells];wi=sparse.load_npz(A/'ACTUAL_Wi.npz').tocsr()[cells]
# LF target indices are outside the RH scaled rows; check identity explicitly.
m=json.loads(Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/RUN/MAPPING.json').read_text());rh={i for r in m['rows'] if r['leg']=='RH' for i in r['local_indices']};assert not set(cells)&rh
results=[];curves={};events={};same=[]
for r in s['records']:
 z=np.load(O/(r['tag']+'.npz'));sp=z['spikes'];st=z['states'];exc=z['exc_increments'];inh=z['inh_increments'];dt=r['dt_ms'];method=r['method'];delta=r['delta_nA'];source=np.unpackbits(np.load(Path(p['source_dirs'][method])/f'all_claw_dt0.015625_delta{delta}.npz')['packed'],axis=1)[:,:12000]
 arrival=np.ceil(((np.arange(len(source))+1)*p['source_dt_ms']+2)/dt).astype(int);keep=arrival<len(sp)
 for w,key in [(we,'exc_increments'),(wi,'inh_increments')]:
  inc=np.asarray((w@source.T).T);expected=np.zeros_like(z[key]);np.add.at(expected,arrival[keep],inc[keep]);assert np.allclose(expected,z[key],rtol=1e-12,atol=1e-12)
 v=np.full(7,-65.);ge=np.zeros(7);gi=np.zeros(7);ad=np.zeros(7);until=np.zeros(7);maxerr=0.;rates=np.zeros(7);commands=[]
 for k in range(len(sp)):
  if k%round(1/dt)==0:commands.append(np.clip([rates[:5].mean()/100,rates[5:].mean()/100],0,1))
  ge+=exc[k];gi+=inh[k];G=.001+ge+gi;drive=-.065-70*gi-ad;active=k*dt+1e-10>=until
  if method=='backward_euler':nv=(.02/dt*v+drive)/(.02/dt+G)
  else:nv=v+(drive/G-v)*(-np.expm1(-dt*G/.02))
  v=np.where(active,nv,-65.);fired=active&(v>=-50);assert np.array_equal(fired,sp[k]);v[fired]=-65;until[fired]=(k+1)*dt+2;ad=ad*np.exp(-dt/100)+fired*.002;ge*=np.exp(-dt/3);gi*=np.exp(-dt/8);maxerr=max(maxerr,float(abs(np.r_[v,ge,gi,ad]-st[k]).max()));rates=np.exp(-dt/5)*rates+(1-np.exp(-dt/5))*fired/(dt/1000)
 assert maxerr<1e-10;assert np.array_equal(sp.sum(0),r['counts']);curves[method,delta,dt]=np.array(commands);events[method,delta,dt]=[(np.flatnonzero(sp[:,j])+1)*dt for j in range(7)]
 if dt==p['source_dt_ms']:
  match=bool(np.array_equal(sp,source[:,cells]));assert match==r['same_dt_bitwise'];same.append(match)
 results.append(dict(tag=r['tag'],state_reconstruction_max_error=maxerr))
comparisons=[]
for method in p['methods']:
 for delta in p['deltas_nA']:
  for dt in [.03125,.0625]:
   ea=events[method,delta,p['source_dt_ms']];eb=events[method,delta,dt];equal=all(len(a)==len(b) for a,b in zip(ea,eb));timing=max([float(abs(a-b).max()) for a,b in zip(ea,eb) if len(a)] or [0]) if equal else None;mae=float(abs(curves[method,delta,p['source_dt_ms']]-curves[method,delta,dt]).mean());comparisons.append(dict(method=method,delta_nA=delta,dt_ms=dt,counts_equal=equal,max_event_error_ms=timing,drive_mae=mae,pass_local=bool(equal and timing<=1 and mae<=.01)))
assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in p['hashes'].items())
report=dict(importance='核心必需',verified=True,replay_runs=18,all_same_dt_bitwise=all(same),checks=results,comparisons=comparisons,scope='fixed recorded presynaptic arrivals; diagnostic only, event quantization at coarser steps included')
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(verified=True,all_same_dt_bitwise=all(same),comparisons=comparisons)))
