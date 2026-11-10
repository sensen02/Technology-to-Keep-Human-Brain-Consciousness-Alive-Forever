import json,hashlib
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from interval_readout import step
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');S=ROOT/'synaptic_scale/20261106T193236Z/0.25';O=ROOT/'interval_readout/20261106T194505Z';O.mkdir(parents=True,exist_ok=True);assert not (O/'REPLAY.json').exists();p=json.loads((S/'PREREGISTRATION.json').read_text());cells=p['flex']+p['ext'];files=[Path(__file__),Path(__file__).with_name('interval_readout.py'),S/'PREREGISTRATION.json']+list(S.glob('all_claw*.npz'));hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
(O/'PREREGISTRATION.json').write_text(json.dumps(dict(importance='核心必需',source=str(S),phase_shifts_ms=[j*.125 for j in range(8)],duration_ms=180,rule='same source spikes; compare 1ms endpoint sample vs exact within-bin clipped-rate integral using unchanged exponential recurrence and Hz normalization; previous interval drives next interval; no body inputs',criteria='quadrature error<1e-10, end-rate recurrence error<1e-12; report absolute pulse-area spread across phase shifts; no physics claim',hashes=hashes),ensure_ascii=False,indent=2)+'\n')
quad_errors=[]
for x,u in [(0,0),(0,100),(2,0),(.8,4),(1.2,.1),(3,5),(.3,.8)]:
 for dt in [.015625,.03125,.5]:
  end,area=step([x],[u],dt);truth=quad(lambda t:min(1,u+(x-u)*np.exp(-t/5)),0,dt,epsabs=1e-12,points=([-5*np.log((1-u)/(x-u))] if (x-1)*(end[0]-1)<0 else None))[0];quad_errors.append(abs(area[0]-truth))
assert max(quad_errors)<1e-10;rows=[];maxerr=0
for dt in p['dt_ms']:
 for delta in p['deltas_nA']:
  src=np.unpackbits(np.load(S/f'all_claw_dt{dt}_delta{delta}.npz')['packed'],axis=1)[:,cells];phase_rows=[]
  for shift in [j*.125 for j in range(8)]:
   sp=np.zeros((round(180/dt),7));offset=round(shift/dt);sp[offset:offset+len(src)]=src;rate=np.zeros(2);indiv=np.zeros(7);avg=np.zeros(2);acc=np.zeros(2);sample=[];mean=[];n=round(1/dt)
   for k,x in enumerate(sp):
    if k%n==0:sample.append(np.clip(rate,0,1));mean.append(avg.copy());acc[:]=0
    u=np.array([x[:5].mean(),x[5:].mean()])/(dt/1000)/100;rate,area=step(rate,u,dt);acc+=area
    indiv=np.exp(-dt/5)*indiv+(1-np.exp(-dt/5))*x/(dt/1000);maxerr=max(maxerr,float(abs(rate-[indiv[:5].mean()/100,indiv[5:].mean()/100]).max()))
    if (k+1)%n==0:avg=acc.copy() # divide by 1ms interval
   a=np.array(sample);b=np.array(mean);phase_rows.append(dict(shift_ms=shift,sampled_area_s=(a.sum(0)*.001).tolist(),averaged_area_s=(b.sum(0)*.001).tolist()))
   if shift==0:np.savez_compressed(O/f'dt{dt}_delta{delta}.npz',sampled_commands=a,averaged_commands=b)
  rows.append(dict(dt_ms=dt,delta_nA=delta,sampled_area_spread_s=np.ptp([r['sampled_area_s'] for r in phase_rows],axis=0).tolist(),averaged_area_spread_s=np.ptp([r['averaged_area_s'] for r in phase_rows],axis=0).tolist(),phases=phase_rows))
assert maxerr<1e-12;assert hashes=={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
report=dict(importance='核心必需',verified=True,quadrature_max_error=max(quad_errors),recurrence_max_error=maxerr,source_trains=6,replays=48,neural_runs=0,physical_runs=0,rows=rows)
(O/'REPLAY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(verified=True,quadrature_max_error=max(quad_errors),recurrence_max_error=maxerr,spreads=[{k:r[k] for k in ['dt_ms','delta_nA','sampled_area_spread_s','averaged_area_spread_s']} for r in rows])))
