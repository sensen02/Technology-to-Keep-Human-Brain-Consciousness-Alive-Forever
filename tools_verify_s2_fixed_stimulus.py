"""Independent offline S2 case checks; no MuJoCo imports or simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from engine.embodied.source_snapshot import verify_source_snapshot


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def verify(bundle):
    bundle=Path(bundle).resolve();report=json.loads((bundle/'report.json').read_text())
    checks={'runner_pass':report['status']=='PASS','hashes_recorded_unchanged':report['hashes_unchanged']}
    verify_source_snapshot(bundle/'source_snapshot.zip',report['source_hashes_before'])
    checks['source_zip']=True
    checks['live_source_input_hashes']=all(sha(p)==h for group in ('source_hashes_before','input_hashes_before') for p,h in report[group].items())
    poses=np.load(bundle/'initial_pose.npz',allow_pickle=False)
    pose=poses['projected_seated_qpos'];cases=0;maximum=0.;summary={};metas={}
    artifacts={}
    for branch, info in report['branches'].items():
        meta=json.loads(Path(info['metadata']).read_text());metas[branch]=meta
        nu,nv=meta['nu'],meta['nv'];base=None
        checks[branch+'_case_count']=len(info['cases'])==nu+2
        branchmax=0.
        for label,entry in info['cases'].items():
            p=Path(entry['npz']);artifacts[str(p)]=sha(p)
            with np.load(p,allow_pickle=False) as z:
                a={k:z[k] for k in z.files}
            cases+=1
            expected=np.zeros(nu)
            if label.startswith('actuator_'):expected[int(label.split('_')[1])]=.1
            prefix=branch+'/'+label
            checks[prefix+'/clock']=a['time_s'].shape==(100,) and np.allclose(a['time_s'],np.arange(1,101)*.0001,rtol=0,atol=1e-14)
            checks[prefix+'/control']=a['ctrl'].shape==(100,nu) and np.array_equal(a['ctrl'],np.broadcast_to(expected,(100,nu)))
            checks[prefix+'/initial']=np.array_equal(a['initial_qpos'],pose) and not a['initial_act'].any() and not a['initial_qvel'].any()
            checks[prefix+'/finite']=all(np.isfinite(v).all() for v in a.values())
            checks[prefix+'/warning']=not a['warning_counts'].any()
            # Rebuild from full geometry+force, ignoring saved factorized/check arrays.
            q=np.einsum('tij,ti->tj',a['moment'],a['actuator_force'])
            err=float(np.max(np.abs(q-a['qfrc_actuator'])));maximum=max(maximum,err);branchmax=max(branchmax,err)
            checks[prefix+'/factorization']=np.allclose(q,a['qfrc_actuator'],rtol=1e-10,atol=1e-10)
            checks[prefix+'/initial_factorization']=np.allclose(a['initial_moment'].T@a['initial_actuator_force'],a['initial_qfrc_actuator'],rtol=1e-10,atol=1e-10)
            if label=='zero_0':base=a
            elif label=='zero_1':checks[prefix+'/repeat']=all(np.array_equal(a[k],base[k]) for k in a)
        summary[branch]={'cases':len(info['cases']),'zero_peak_activation':float(np.max(np.abs(base['act']))),
                         'zero_peak_force_model':float(np.max(np.abs(base['actuator_force']))),'max_factorization_error':branchmax}
    n,z=metas['native_floor'],metas['zero_floor']
    checks['branch_metadata_only_ctrlrange_changed']=all(n[k]==z[k] for k in n if k!='actuator_ctrlrange')
    checks['native_floor_preserved']=n['actuator_ctrlrange']==n['native_ctrlrange']
    checks['zero_floor_zero']=all(v[0]==0 for v in z['actuator_ctrlrange'])
    checks['zero_floor_upper_preserved']=all(a[1]==b[1] for a,b in zip(n['actuator_ctrlrange'],z['actuator_ctrlrange']))
    return dict(pass_all=all(checks.values()),checks=checks,cases=cases,max_factorization_error=maximum,
                summary=summary,artifact_sha256=artifacts,bundle=str(bundle),biological_validation=False,
                scope='offline engineering identity/inputs/repeat/factorization, not live execution reproduction or convergence')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=verify(a.bundle)
    with a.output.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:r[k] for k in ('pass_all','cases','max_factorization_error','summary')},indent=2))
    raise SystemExit(0 if r['pass_all'] else 1)


if __name__=='__main__':main()
