"""Independent parent check: evidence counts, raw support, robustness and selection."""
from pathlib import Path
import json
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-robust-readout/20261106T031010Z')
def read(p): return json.loads(Path(p).read_text())
def main():
    summary=read(O/'SUMMARY.json'); rows=summary['results']; identity=read(O/'TRANSLATOR_IDENTITIES.json')
    limits=np.asarray(identity['limits']); checks=[]; table=[]
    assert len(rows)==summary['physical_releases']==23
    assert summary['selected_tau']==.005 and summary['final_skipped']
    for r in rows:
        with np.load(Path(r['folder'])/'NATIVE.npz') as z: a=z['rows']
        with np.load(Path(r['folder'])/'NEURAL.npz') as z: applied=z['applied']; action=z['action']; residual=z['residual']
        assert a.shape[0]==16001 and np.all(np.isfinite(a))
        assert np.max(np.abs(np.diff(a[:,0])-.0000125))<1e-12
        assert np.all(action==0) and np.all(residual==0)
        assert np.max(np.abs(applied)-limits)<=1e-12
        foot=float(np.mean(np.sum(a[:,1:7],axis=1))/24.446354755878005)
        com=float(a[:,10].min())
        assert abs(foot-r['parts']['foot_support_mean'])<1e-10
        assert abs(com-r['parts']['com_z_min'])<1e-10
        positive=np.flatnonzero(a[:,7]>0)
        first=None if not len(positive) else float(a[positive[0],0]-r['release_time_s'])
        assert r['validation']['checks']['D1_no_nonfoot_ground_force']==(len(positive)==0)
        assert r['validation']['passed']==all(r['validation']['checks'].values())
        checks.append(dict(folder=r['folder'],tau=r['tau_s'],J=r['J'],foot=foot,COM_min=com,first_positive_nonfoot_s=first,D_pass=r['validation']['passed']))
        if r['label'].endswith('dev1.3'): table.append(checks[-1])
    robust=[]
    for tau in [.005,.020,.050,.100]:
        key=f'cmd_{tau:.3f}'
        with np.load(O/'REPLAY_sigma0.0_seed0.npz') as z: ref=z[key]
        errors=[]
        for sigma in [.001,.01]:
            for seed in [1,2,3]:
                with np.load(O/f'REPLAY_sigma{sigma}_seed{seed}.npz') as z: cmd=z[key]
                for start in [0,100]: errors.append(float(np.sqrt(np.mean((cmd[start:]-ref[start:])**2))/(np.sqrt(np.mean(ref[start:]**2))+1e-12)))
        actual=max(errors); row=next(r for r in summary['ranking'] if r['tau']==tau)
        assert abs(actual-row['max_relative_rms'])<1e-12
        assert row['robust']==(actual<=.15)
        assert row['passes']==sum(r['validation']['passed'] for r in rows if r['tau_s']==tau and '_dev' in r['label'])
        robust.append(dict(tau=tau,max_relative_rms=actual))
    assert not any(r['validation']['passed'] for r in rows)
    result=dict(importance='重要辅助',passed=True,releases_checked=len(checks),total_D_passes=0,guard_comparison=table,robustness=robust,raw_support_checks=checks)
    dest=O/'PARENT_VERIFICATION.json';assert not dest.exists()
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='raw_support_checks'},ensure_ascii=False))
if __name__=='__main__':main()
