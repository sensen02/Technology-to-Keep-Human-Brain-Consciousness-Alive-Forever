"""Analytic two-cell numerical controls, separate from 12k biological claims."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from engine.neural_cond import ConductanceNetwork,ConductanceParams
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/claw_direction/fine/20261106T191739Z');records=[]
for method in ['backward_euler','frozen_exact']:
 for dt in [.0625,.03125,.015625]:
  for current in [.04,.1]:
   p=ConductanceParams();net=ConductanceNetwork(2,p,dt_ms=dt,voltage_method=method);net.set_connectivity(sparse.csr_matrix(([.001],([1],[0])),shape=(2,2)),sparse.csr_matrix((2,2)))
   vinf=p.rest_mV+current/p.leak_uS;fraction=(vinf-p.threshold_mV)/(vinf-p.rest_mV);continuous=-p.capacitance_nF/p.leak_uS*np.log(fraction)
   if method=='backward_euler':expected_steps=int(np.ceil(np.log(fraction)/np.log(p.capacitance_nF/(p.capacitance_nF+p.leak_uS*dt))))
   else:expected_steps=int(np.ceil(continuous/dt))
   first=None;arrival=None;refractory_ok=True
   for k in range(round(20/dt)):
    before=net.t_ms;sp=net.step(current_nA=np.array([current if first is None else 0,0]))
    if first is None and sp[0]:first=net.t_ms
    if first is not None and first<=before<first+p.refractory_ms:refractory_ok=refractory_ok and net.v[0]==p.reset_mV and not sp[0]
    if arrival is None and net.ge[1]>0:arrival=before
   assert first is not None and abs(first-expected_steps*dt)<1e-10 and abs(arrival-first-p.delay_ms)<1e-10 and refractory_ok
   records.append(dict(method=method,dt_ms=dt,current_nA=current,first_spike_ms=first,continuous_first_spike_ms=float(continuous),time_error_ms=float(first-continuous),arrival_start_ms=arrival,delay_ms=arrival-first,refractory_ok=bool(refractory_ok)))
report=dict(importance='核心必需',passed=True,component_checks=len(records),records=records,scope='two-cell analytic RC threshold, explicit event delay and refractory; does not establish full-network convergence',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),engine_sha256=hashlib.sha256((R/'engine/neural_cond.py').read_bytes()).hexdigest())
(O/'COMPONENT_TIMING.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
