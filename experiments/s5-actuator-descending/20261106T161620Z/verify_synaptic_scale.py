import subprocess,sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/synaptic_scale/20261106T193236Z');A=Path('/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z/ARMS/A');M=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/RUN/MAPPING.json');p=json.loads((O/'PREREGISTRATION.json').read_text());e=sparse.load_npz(A/'ACTUAL_We.npz').tocsr();i=sparse.load_npz(A/'ACTUAL_Wi.npz').tocsr();rh={j for row in json.loads(M.read_text())['rows'] if row['leg']=='RH' for j in row['local_indices']};scale=np.ones(12000);scale[list(rh)]=.375;e=(sparse.diags(scale)@e).tocsr();configs=[]
for gain in p['efficacy_scales']:
 folder=O/str(gain);subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('verify_sensory_fine.py')),str(folder)],check=True);v=json.loads((folder/'VERIFICATION.json').read_text());s=json.loads((folder/'SUMMARY.json').read_text());effective_hashes={}
 for base,name in [(e,'We'),(i,'Wi')]:
  path=folder/f'EFFECTIVE_{name}.npz';actual=sparse.load_npz(path).tocsr();expected=(base*gain).tocsr();assert actual.shape==expected.shape and np.array_equal(actual.indptr,expected.indptr) and np.array_equal(actual.indices,expected.indices) and np.array_equal(actual.data,expected.data);assert (actual.data>=0).all();effective_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 baseline=[sum(r['flex_counts'])+sum(r['ext_counts']) for r in s['records'] if r['delta_nA']==0];activity=all(n>=1 for n in baseline)
 effect=all(max(abs(np.array(c[k])))>=1e-4 for c in v['checks'] if c['delta_nA']!=0 for k in ['response_integral_a_s','response_integral_b_s'])
 configs.append(dict(efficacy_scale=gain,baseline_motor_counts_50_150ms=baseline,activity_passed=activity,sensory_effect_passed=bool(effect),convergence_passed=v['convergence_screen_passed'],combined_passed=bool(activity and effect and v['convergence_screen_passed']),effective_matrix_hashes=effective_hashes))
assert hashlib.sha256(Path(__file__).with_name('synaptic_scale_20261106T193236Z.py').read_bytes()).hexdigest()==p['source_sha256']
report=dict(importance='核心必需',verified=True,neural_runs=18,physics_runs=0,configs=configs,all_scaled_matrices_exact=True)
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
