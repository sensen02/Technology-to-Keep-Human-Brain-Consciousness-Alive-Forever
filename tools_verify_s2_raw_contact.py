"""Independent saved-data arithmetic only; no runner import or MuJoCo dynamics."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def verify(out):
    out=Path(out); manifest=json.loads((out/'manifest.json').read_text()); checks=[]; maxima=[]
    for r in manifest['records']:
        saved=json.loads((out/r['pair_json']).read_text())
        with np.load(out/r['npz'],allow_pickle=False) as a:
            checks.append(all(a[k].dtype.kind != 'O' for k in a.files))
            checks.append(len(a['time_s'])==400 and np.allclose(a['time_s'],np.arange(1,401)*25e-6,rtol=0,atol=1e-14))
            checks.append(np.array_equal(np.diff(a['contact_offsets']),a['ncon']) and np.array_equal(np.diff(a['efc_offsets']),a['nefc']))
            n=round(.01/r['dt_s']); control=np.zeros_like(a['step_ctrl']);control[:round(.001/r['dt_s']),0]=.1
            checks.append(len(control)==n and np.array_equal(control,a['step_ctrl']) and np.allclose(a['step_time_s'],np.arange(n)*r['dt_s'],rtol=0,atol=1e-14))
            checks.append(np.count_nonzero(a['initial_qvel'])==0 and np.count_nonzero(a['initial_act'])==0 and np.count_nonzero(a['initial_ctrl'])==0)
            residual=0.
            for i in range(400):
                lo,hi=a['contact_offsets'][i:i+2]; reconstructed={}; total=np.zeros(3)
                for j in range(lo,hi):
                    g1,g2=map(int,a['contact_geom'][j]);key=f'{min(g1,g2)},{max(g1,g2)}';f=a['contact_local6'][j,:3];frame=a['contact_frame'][j].reshape(3,3)
                    # Explicit basis linear combination, not runner's matrix expression.
                    world=f[0]*frame[0]+f[1]*frame[1]+f[2]*frame[2]
                    if g1>g2:world=-world
                    reconstructed[key]=reconstructed.get(key,np.zeros(3))+world;total+=world
                checks.append(set(reconstructed)==set(saved[i]))
                for key,v in reconstructed.items():
                    expected=np.asarray(saved[i][key]);checks.append(np.allclose(v,expected,rtol=1e-12,atol=1e-14));residual=max(residual,float(np.max(np.abs(v-expected))))
                expected_total=np.sum(list(saved[i].values()),axis=0) if saved[i] else np.zeros(3)
                checks.append(np.allclose(total,expected_total,rtol=1e-12,atol=1e-14))
            maxima.append(residual)
    report=json.loads((out/'report.json').read_text()); pair_data=[json.loads((out/r['pair_json']).read_text()) for r in manifest['records']]
    for key,series in report['force_difference_series'].items():
        independent=[float(np.sqrt(sum((pair_data[0][i].get(key,[0,0,0])[j]-pair_data[1][i].get(key,[0,0,0])[j])**2 for j in range(3)))) for i in range(400)]
        checks.append(np.allclose(independent,series,rtol=1e-12,atol=1e-14))
    checks.append(manifest['source_hashes_before']==manifest['source_hashes_after'])
    for p,h in manifest['source_hashes_before'].items():checks.append(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h)
    result=dict(passed=all(checks),checks=len(checks),failed=sum(not c for c in checks),pair_residual_max=maxima,scope='Independent raw contact arithmetic replay only, no new dynamics')
    (out/'independent-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert result['passed']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);verify(parser.parse_args().out)
