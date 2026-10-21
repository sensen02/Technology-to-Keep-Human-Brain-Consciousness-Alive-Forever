"""V2: persist raw endpoint before mass decode; no integration or retries."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import numpy as np
from run_s2_forward_adapter import capture, sha, ROOT, POSE, ARTIFICIAL_XML
from tools_s2_mass_layout import decode_tree, decode_csr
from tools_s2_stationarity import analyze, UNITS


def save_numeric(path, arrays):
    if not all(v.dtype.kind in 'biuf' and np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('nonfinite or nonnumeric capture')
    count=sum(v.nbytes for v in arrays.values())
    if count>64*1024*1024: raise ValueError('numeric capture budget')
    np.savez_compressed(path,**arrays)
    return dict(path=path.name,sha256=sha(path),numeric_bytes=count)


def main():
    import mujoco
    import engine.canonical_physics as cp
    assert mujoco.__version__=='3.9.0'
    prior=Path('/home/sensen/Desktop/engine修复验收/S2-mass-layout-contract-20261014-v1/verification.json')
    old=json.loads(prior.read_text())['frozen_hashes']
    assert all(sha(p)==h for p,h in old.items())
    names=('run_s2_stationarity_v2.py','S2_STATIONARITY_V2_SCOPE.zh-CN.md','tools_s2_mass_layout.py','test_s2_mass_layout.py','S2_MASS_LAYOUT_CONTRACT.zh-CN.md','tools_s2_stationarity.py','test_s2_stationarity.py','tools_s2_final_state_contract.py','run_s2_forward_adapter.py')
    sources={str(ROOT/n):sha(ROOT/n) for n in names}
    out=Path('/home/sensen/Desktop/engine修复验收')/('S2-stationarity-v2-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(exist_ok=False)
    (out/'artificial-model.xml').write_text(ARTIFICIAL_XML)
    report=dict(policy='REPORT_ONLY',environment=dict(mujoco=mujoco.__version__,python=sys.executable),frozen_hashes=old,source_hashes=sources,artificial_xml_sha256=sha(out/'artificial-model.xml'),cases=[],status='STARTED',api_input='mj_fullM receives d.qM; CSR decoder receives d.M',mass_api_doc=mujoco.mj_fullM.__doc__)
    def checkpoint(): (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    checkpoint();print(out,flush=True)
    plotdata=[]
    try:
        for name in ('canonical','artificial'):
            rec=dict(case=name,phase='ALLOCATING',forward_calls=0,step_calls=0);report['cases'].append(rec);checkpoint()
            m=cp.load_model() if name=='canonical' else mujoco.MjModel.from_xml_string(ARTIFICIAL_XML)
            assert 0<m.nv<=256
            d=mujoco.MjData(m)
            if name=='canonical':
                with np.load(POSE,allow_pickle=False) as p:d.qpos[:]=p['projected_seated_qpos']
                d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0
            initial={k:getattr(d,k).copy() for k in ('qpos','qvel','act','ctrl')};time=float(d.time)
            rec['phase']='FORWARD_CALL_ENTERED';rec['forward_calls']=1;checkpoint()
            mujoco.mj_forward(m,d)
            rec['phase']='FORWARD_RETURNED';checkpoint()
            assert d.nefc<=4096 and d.nisland<=mujoco.mjNISLAND
            a,base=capture(m,d,mujoco)
            a.update(qM=d.qM.copy(),M_sparse=d.M.copy(),parent=m.dof_parentid.copy(),M_rowadr=m.M_rowadr.copy(),M_rownnz=m.M_rownnz.copy(),M_colind=m.M_colind.copy(),qacc_smooth=d.qacc_smooth.copy(),qfrc_smooth=d.qfrc_smooth.copy(),qfrc_constraint=d.qfrc_constraint.copy(),dof_jntid=m.dof_jntid.copy(),jnt_type=m.jnt_type.copy(),jnt_dofadr=m.jnt_dofadr.copy())
            units=[]
            for dof,j in enumerate(m.dof_jntid):
                t=int(m.jnt_type[j]);local=dof-int(m.jnt_dofadr[j])
                units.append(UNITS[0] if t==2 or (t==0 and local<3) else UNITS[1])
            a['dof_unit_code']=np.array([UNITS.index(u) for u in units],dtype=np.int64)
            for k,v in initial.items():a['initial_'+k]=v;a['after_'+k]=getattr(d,k).copy()
            a['time_before_after']=np.array([time,d.time])
            rec.update(raw=save_numeric(out/(name+'-raw.npz'),a),capture=base,nv=int(m.nv),ncon=int(d.ncon),options={k:float(getattr(m.opt,k)) for k in ('integrator','timestep','cone','enableflags','disableflags')},phase='RAW_SAVED');checkpoint()
            assert d.time==time and not np.any(d.warning.number)
            assert all(v.tobytes()==getattr(d,k).tobytes() for k,v in initial.items())
            snapshot={k:getattr(d,k).copy() for k in ('qpos','qvel','act','ctrl','qM','M','qacc','qacc_smooth','qfrc_smooth','qfrc_constraint','efc_J','efc_force','efc_aref','efc_D','efc_R','efc_state','iefc_state')}
            full=np.empty((m.nv,m.nv));mujoco.mj_fullM(m,full,d.qM)
            tree=decode_tree(a['qM'],a['parent']);csr=decode_csr(a['M_sparse'],a['M_rowadr'],a['M_rownnz'],a['M_colind'],int(m.nv))
            basis=np.empty_like(full)
            for j in range(m.nv):
                e=np.zeros(m.nv);e[j]=1;v=np.empty(m.nv);mujoco.mj_mulM(m,d,v,e);basis[:,j]=v
            mul=np.empty(m.nv);mujoco.mj_mulM(m,d,mul,d.qacc)
            assert d.time==time and all(v.tobytes()==getattr(d,k).tobytes() for k,v in snapshot.items())
            matrices=dict(M_fullM=full,M_tree=tree,M_csr=csr,M_basis=basis,mulM_qacc=mul)
            rec['mass_comparison_absmax']={k:float(np.max(np.abs(v-full))) for k,v in matrices.items() if v.ndim==2}
            save_numeric(out/(name+'-mass.npz'),matrices)
            for k,v in matrices.items():
                if v.ndim==2:assert np.allclose(v,full,rtol=1e-12,atol=1e-14),k
            z=analyze(full,a['J'],a['qacc'],a['qacc_smooth'],a['qfrc_smooth'],a['qfrc_constraint'],a['force'],units)
            result=dict(a,**matrices)
            for k,v in z['arrays'].items():result['computed_'+k]=v
            for k,v in z['denominators'].items():result['denominator_'+k]=v
            rec.update(result=save_numeric(out/(name+'-result.npz'),result),mass=dict(path=name+'-mass.npz',sha256=sha(out/(name+'-mass.npz'))),groups=z['groups'],mulM_qacc_difference=(mul-z['arrays']['Ma']).tolist(),phase='COMPLETE',state_unchanged=True,mass_calls_snapshot_unchanged=True);checkpoint()
            plotdata.append((name,np.array(units),z['arrays']['g_rows'].copy()))
        assert all(sha(p)==h for p,h in old.items()) and all(sha(p)==h for p,h in sources.items())
        report['hashes_unchanged']=True;report['status']='COMPLETE';checkpoint()
    except Exception as exc:
        report.update(status='FAILED',error=repr(exc));checkpoint();raise
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for row,(name,labels,g) in enumerate(plotdata):
        for col,unit in enumerate(UNITS):
            ids=np.flatnonzero(labels==unit);ax=axes[row,col];ax.plot(ids,g[ids],'o-');ax.set_title(name+' / '+unit);ax.set_xlabel('DOF');ax.set_ylabel('stationarity (model units)');ax.grid(alpha=.3)
    fig.suptitle('REPORT ONLY: single-forward, not static equilibrium / full KKT');fig.tight_layout();fig.savefig(out/'stationarity.png');plt.close(fig)
    for r in report['cases']:print(r['case'],json.dumps(r['groups'],indent=2),flush=True)


if __name__=='__main__':main()
