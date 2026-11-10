"""Independent saved-result audit: no model construction or neural/physics steps.
Run only after the experiment writes its full delivery manifest.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

O=Path('/home/sensen/Desktop/engine修复验收/s5-strength-fit/20261106T051337Z')
W=24.446354755878005

def read(p):
    return json.loads(Path(p).read_text())

def main():
    manifest=read(O/'MANIFEST_SHA256.json')
    bad=[]
    for name,expected in manifest.items():
        p=Path(name)
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected: bad.append(name)
    assert not bad,bad
    results=[Path(p) for p in manifest if Path(p).name=='RESULT.json' and 'RUNS' in Path(p).parts]
    assert results,'manifest must include per-release results'
    checked=[]
    for p in results:
        r=read(p)
        with np.load(p.parent/'NATIVE.npz') as z: a=z['rows']
        with np.load(p.parent/'NEURAL.npz') as z:
            action=z['action'];residual=z['residual'];applied=z['applied']
        assert np.isfinite(a).all() and np.isfinite(applied).all()
        assert not action.any() and not residual.any()
        assert a.shape[0]==round(r['duration_s']/.0000125)+1
        assert np.max(abs(np.diff(a[:,0])-.0000125))<1e-11
        foot=float(a[:,1:7].sum(axis=1).mean()/W)
        com=float(a[:,10].min())
        assert abs(foot-r['parts']['foot_support_mean'])<1e-9
        assert abs(com-r['parts']['com_z_min'])<1e-9
        pos=np.flatnonzero(a[:,7]>0)
        first=None if not len(pos) else float(a[pos[0],0]-r['release_time_s'])
        assert bool(len(pos)==0)==r['validation']['checks']['D1_no_nonfoot_ground_force']
        assert all(r['validation']['checks'].values())==r['validation']['passed']
        assert r['warnings']==0 and r['release_aux_max']==0 and r['release_external_max']==0
        checked.append(dict(result=str(p),J=r['J'],foot_support_mean=foot,com_z_min=com,first_positive_nonfoot_s=first,passed=r['validation']['passed']))
    out=dict(importance='重要辅助',passed=True,manifest_entries=len(manifest),hash_mismatches=bad,releases_checked=len(checked),D_pass_count=sum(x['passed'] for x in checked),method='parent independent raw native support/height/positive contact/causal residual/count checks plus manifest hashes; no runtime scorer',results=checked)
    dest=O/'PARENT_VERIFICATION.json'
    assert not dest.exists(),'retain original verification; do not overwrite'
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='results'},ensure_ascii=False))

if __name__=='__main__':main()
