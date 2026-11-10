"""Independent final audit of algorithm comparison evidence, no simulation."""
from pathlib import Path
import hashlib,json
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-optimizer-comparison/20261106T053754Z')
W=24.446354755878005

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main():
    manifest=json.loads((O/'MANIFEST_SHA256.json').read_text())
    bad=[p for p,h in manifest.items() if not Path(p).exists() or sha(p)!=h]
    assert not bad,bad
    paths=[Path(p) for p in manifest if Path(p).name=='RESULT.json' and 'RUNS' in Path(p).parts]
    assert paths
    rows=[]
    for p in paths:
        r=json.loads(p.read_text())
        with np.load(p.parent/'NATIVE.npz') as z:a=z['rows']
        with np.load(p.parent/'NEURAL.npz') as z:
            action=z['action'];residual=z['residual'];applied=z['applied']
        assert np.isfinite(a).all() and np.isfinite(applied).all()
        assert not action.any() and not residual.any()
        assert a.shape[0]==round(r['duration_s']/.0000125)+1
        assert np.max(abs(np.diff(a[:,0])-.0000125))<1e-11
        foot=float(a[:,1:7].sum(1).mean()/W);com=float(a[:,10].min())
        assert abs(foot-r['parts']['foot_support_mean'])<1e-9
        assert abs(com-r['parts']['com_z_min'])<1e-9
        assert bool(a[:,7].max()==0)==r['validation']['checks']['D1_no_nonfoot_ground_force']
        assert all(r['validation']['checks'].values())==r['validation']['passed']
        positive=np.flatnonzero(a[:,7]>1e-9)
        first=None if len(positive)==0 else float(a[positive[0],0]-r['release_time_s'])
        if 'first_positive_nonfoot_s' in r:assert first==r['first_positive_nonfoot_s']
        assert r['warnings']==r['release_aux_max']==r['release_external_max']==0
        rows.append(dict(result=str(p),label=r['label'],J=r['J'],foot_support=foot,COM_min=com,first_positive_nonfoot_s=first,D_pass=r['validation']['passed']))
    result=dict(importance='重要辅助',passed=True,manifest_entries=len(manifest),hash_mismatches=bad,releases=len(rows),D_passes=sum(x['D_pass'] for x in rows),method='Independent parent hash audit and raw-array support/COM/positive contact/zero residual/dt checks; no simulation or old scoring function',results=rows)
    dest=O/'PARENT_VERIFICATION.json';assert not dest.exists()
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'},ensure_ascii=False))
if __name__=='__main__':main()
