"""Independent S123 frozen-array arithmetic verifier; no model/physics imports."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
B=Path('/home/sensen/Desktop/engine修复验收')
P=B/'s121-constraint-row-recording/20261024T072500Z/init80ms'
Q=B/'s122-secant-constraint-pair/20261024T073500Z/w36b4_secant'
OUT=B/'s123-muscle-compensation/20261024T074500Z'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 with np.load(P/'TRAJECTORY.npz') as z:a={k:z[k] for k in z.files}
 with np.load(Q/'TRAJECTORY.npz') as z:b={k:z[k] for k in z.files}
 assert np.array_equal(a['time'],b['time']);mask=a['time']<=.002825+1e-12
 f0=a['actuator_force'][mask,:,None];f1=b['actuator_force'][mask,:,None];m0=a['actuator_moment'][mask,:,6:9];m1=b['actuator_moment'][mask,:,6:9]
 dtau=f1*m1-f0*m0;force=(f1-f0)*m0;moment=f0*(m1-m0);cross=(f1-f0)*(m1-m0)
 err=float(abs(dtau-force-moment-cross).max());assert err<1e-12
 for x in (a,b):assert abs((x['actuator_force'][:,:,None]*x['actuator_moment'][:,:,6:9]).sum(1)-x['qfrc_actuator'][:,6:9]).max()<1e-10
 keep=np.arange(90)!=2;assert np.array_equal(a['ctrl'][:,keep],b['ctrl'][:,keep]);assert np.array_equal(a['act'][:,keep],b['act'][:,keep])
 i=int(np.argmin(abs(a['time']-.0025)));full=(b['actuator_force'][:,:,None]*b['actuator_moment'][:,:,6:9]-a['actuator_force'][:,:,None]*a['actuator_moment'][:,:,6:9]);other=float(full[i,keep,2].sum());assert abs(other-.03589112560424998)<1e-14
 summary=dict(passed=True,zero_model_loads=True,zero_forward=True,zero_steps=True,all89_ctrl_exact=True,all89_activation_exact=True,states=int(mask.sum()),three_term_max_abs=err,other89_tau8_2p5=other,per_actuator_tau8_2p5=[float(x) for x in full[i,:,2]],inputs={str(p):sha(p) for p in (P/'TRAJECTORY.npz',Q/'TRAJECTORY.npz')},verifier_sha256=sha(__file__))
 for manifest in ('INPUT_HASHES.json','ARTIFACT_HASHES.json'):
  p=OUT/manifest
  if p.exists():assert all(sha(name)==h for name,h in json.loads(p.read_text()).items())
 (OUT/'PARENT_INDEPENDENT_CHECK.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
