"""S142 supplemental independent 164-tick neural provenance and dynamics check."""
from pathlib import Path
import json,hashlib,ast
import numpy as np
from verify_com_contact_probe import reconstruct_fir
from verify_neural_activation_initialized_probe import reconstruct_activation
ROOT=Path('/home/sensen/Desktop/engine修复验收/s142-free-physical-prewarm/20261024T201419378677Z')
BASE=Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z/init80ms')
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=ROOT/'init80ms';a=load(p/'TRAJECTORY.npz');b=load(BASE/'TRAJECTORY.npz');i=load(p/'INITIALIZATION.npz');h=json.loads((ROOT/'INPUT_HASHES.json').read_text());n=len(a['time']);ticks=len(a['spikes']);assert n==78 and ticks==4
 for k,v in b.items():
  if k not in a:
   assert k in ('com_position_fd_velocity_world','com_momentum_fd_force_world','com_impulse_residual_world'),k
   continue
  if k in ('weights','actuator_pool'):assert np.array_equal(a[k],v),k
  elif v.shape and v.shape[0]==121:assert np.array_equal(a[k],v[:n]),k
  elif v.shape and v.shape[0]==6:assert np.array_equal(a[k],v[:ticks]),k
 paths={str(Path(x).resolve()) for x in h if Path(x).name=='MANIFEST.json'};assert len(paths)==1
 mf=sorted(json.loads(Path(next(iter(paths))).read_text())['rows'],key=lambda r:r['actuator_index']);pools=sorted({tuple(sorted(r['local_indices'])) for r in mf if r['local_indices']});lookup={x:j for j,x in enumerate(pools)};mapping=np.array([lookup.get(tuple(sorted(r['local_indices'])),-1) for r in mf]);assert np.array_equal(mapping,a['actuator_pool'])
 fs,us,hist=reconstruct_fir(np.concatenate((i['spikes'],a['spikes'])),pools,mapping,a['weights'])
 for x,y in [(fs[:160],i['features']),(us[:160],i['decoded']),(fs[160:],a['features']),(us[160:],a['applied']),(hist,load(p/'NEURAL_FINAL.npz')['FIR_history'])]:assert np.max(abs(x-y))<=1e-14
 for k in range(n):
  u=np.zeros(90) if k==0 else us[160+(k-1)//20];assert np.array_equal(a['ctrl'][k],u)
 # Full20-substep reconstruction then truncate partial final physical interval.
 import xml.etree.ElementTree as ET
 xml=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml');nodes=list(ET.parse(xml).getroot().find('actuator'));prm=np.array([list(map(float,x.get('dynprm').split()))[:3] for x in nodes]);assert prm.shape==(90,3)
 # Existing scalar helper accepts six commands; pad irrelevant future commands, compare only n.
 pad=np.concatenate((a['applied'],np.repeat(a['applied'][-1:],6-ticks,axis=0)));acts=reconstruct_activation(np.zeros(90),pad,prm)[:n];err=float(abs(acts-a['act']).max());assert err<=1e-12
 force_res=a['qfrc_actuator']+a['qfrc_passive']+a['qfrc_constraint']-a['qfrc_bias']-a['Ma'];assert abs(force_res).max()<=1e-10
 stop=json.loads((p/'STOP.json').read_text());assert stop['stop_reasons']==['LOAD_RATIO_LT_0.8'];assert a['all_six_gates'][:-1].all() and not a['all_six_gates'][-1]
 report=dict(passed=True,ticks=164,samples=n,all_original_baseline_fields_prefix_exact=True,FIR_max_tolerance=1e-14,activation_maxabs=err,native_all48_balance_maxabs=float(abs(force_res).max()),prewarm_success=False,physical_units='NOT_ESTABLISHED',hashes={str(x):sha(x) for x in (Path(__file__),xml,p/'TRAJECTORY.npz',p/'INITIALIZATION.npz',Path(next(iter(paths))))})
 with (p/'COMMAND_DYNAMICS_INDEPENDENT_CHECK.json').open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report))
if __name__=='__main__':main()
