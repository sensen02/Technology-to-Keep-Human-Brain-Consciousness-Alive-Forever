"""Bounded two single-forward stationarity observations, no integration."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import numpy as np
from run_s2_forward_adapter import capture, frozen, sha, ROOT, POSE, ARTIFICIAL_XML
from tools_s2_stationarity import analyze, UNITS


def dynamic_snapshot(d):
    names=('qpos','qvel','act','ctrl','qacc','qacc_smooth','qfrc_smooth','qfrc_constraint','M','efc_J','efc_force','efc_aref','efc_D','efc_R','efc_state','iefc_state')
    return {k:getattr(d,k).copy() for k in names}


def main():
    import mujoco
    import engine.canonical_physics as cp
    assert mujoco.__version__=='3.9.0'
    prior=Path('/home/sensen/Desktop/engine修复验收/S2-forward-adapter-20261014T215241Z')
    pm=json.loads((prior/'report.json').read_text())
    old=frozen();old.update(pm['source_hashes']);old.update({str(prior/r['npz']):r['sha256'] for r in pm['cases']})
    assert all(sha(p)==h for p,h in old.items())
    sources={str(ROOT/n):sha(ROOT/n) for n in ('run_s2_stationarity_capture.py','tools_s2_stationarity.py','test_s2_stationarity.py','S2_STATIONARITY_PROTOCOL.zh-CN.md','S2_STATIONARITY_CAPTURE_SCOPE.zh-CN.md')}
    out=Path('/home/sensen/Desktop/engine修复验收')/('S2-stationarity-real-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(exist_ok=False)
    (out/'artificial-model.xml').write_text(ARTIFICIAL_XML)
    records=[];plotdata=[]
    for name in ('canonical','artificial'):
        m=cp.load_model() if name=='canonical' else mujoco.MjModel.from_xml_string(ARTIFICIAL_XML)
        assert 0<m.nv<=256
        d=mujoco.MjData(m)
        if name=='canonical':
            with np.load(POSE,allow_pickle=False) as p:d.qpos[:]=p['projected_seated_qpos']
            d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0
        initial={k:getattr(d,k).copy() for k in ('qpos','qvel','act','ctrl')};t=float(d.time)
        mujoco.mj_forward(m,d)
        assert d.time==t and not np.any(d.warning.number)
        assert all(v.tobytes()==getattr(d,k).tobytes() for k,v in initial.items())
        assert d.nefc<=4096 and d.nisland<=mujoco.mjNISLAND
        a,base=capture(m,d,mujoco)
        snapshot=dynamic_snapshot(d)
        M=np.empty((m.nv,m.nv));mujoco.mj_fullM(m,M,d.M)
        mul=np.empty(m.nv);mujoco.mj_mulM(m,d,mul,d.qacc)
        assert d.time==t and all(v.tobytes()==getattr(d,k).tobytes() for k,v in snapshot.items())
        units=[]
        for dof,joint in enumerate(m.dof_jntid):
            jt=int(m.jnt_type[joint]);local=dof-int(m.jnt_dofadr[joint])
            units.append(UNITS[0] if jt==2 or (jt==0 and local<3) else UNITS[1])
        z=analyze(M,a['J'],d.qacc,d.qacc_smooth,d.qfrc_smooth,d.qfrc_constraint,a['force'],units)
        a.update(M_dense=M,M_sparse=d.M.copy(),M_rownnz=m.M_rownnz.copy(),M_rowadr=m.M_rowadr.copy(),M_colind=m.M_colind.copy(),qacc_smooth=d.qacc_smooth.copy(),qfrc_smooth=d.qfrc_smooth.copy(),qfrc_constraint=d.qfrc_constraint.copy(),mulM_qacc=mul,dof_unit_code=np.array([UNITS.index(u) for u in units],dtype=np.int64),dof_jntid=m.dof_jntid.copy(),jnt_type=m.jnt_type.copy(),jnt_dofadr=m.jnt_dofadr.copy())
        for k,v in z['arrays'].items():a['computed_'+k]=v
        for k,v in z['denominators'].items():a['denominator_'+k]=v
        for k,v in initial.items():a['initial_'+k]=v;a['after_'+k]=getattr(d,k).copy()
        a['time_before_after']=np.array([t,d.time])
        assert all(v.dtype.kind in 'biuf' and np.isfinite(v).all() for v in a.values())
        numeric=sum(v.nbytes for v in a.values());assert numeric<=64*1024*1024
        filename=name+'.npz';np.savez_compressed(out/filename,**a)
        records.append(dict(case=name,npz=filename,sha256=sha(out/filename),nv=int(m.nv),ncon=int(d.ncon),metadata=base['metadata'],capture=base,groups=z['groups'],numeric_bytes=numeric,mulM_dense_difference=[float(x) for x in (mul-z['arrays']['Ma'])],forward_calls=1,step_calls=0,mass_operations_do_not_mutate_snapshot=True,state_unchanged=True,options={k:float(getattr(m.opt,k)) for k in ('integrator','timestep','cone','enableflags','disableflags')},mass_storage_shape=list(d.M.shape)))
        plotdata.append((name,np.array(units),z['arrays']['g_rows'].copy()))
    assert all(sha(p)==h for p,h in old.items()) and all(sha(p)==h for p,h in sources.items())
    report=dict(policy='REPORT_ONLY',environment={'mujoco':mujoco.__version__,'python':sys.executable},mass_api={'fullM':mujoco.mj_fullM.__doc__,'mulM':mujoco.mj_mulM.__doc__,'input':'d.M, not legacy assumed qM'},cases=records,frozen_hashes=old,source_hashes=sources,artificial_xml_sha256=sha(out/'artificial-model.xml'),limitations='Single forward only, not static equilibrium, full KKT, convergence, biological calibration or behavior')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for row,(name,labels,g) in enumerate(plotdata):
        for col,unit in enumerate(UNITS):
            ids=np.flatnonzero(labels==unit);axes[row,col].plot(ids,g[ids],'o-');axes[row,col].set_title(name+' / '+unit);axes[row,col].set_xlabel('DOF index');axes[row,col].set_ylabel('stationarity residual (model units)');axes[row,col].grid(alpha=.3)
    fig.suptitle('REPORT ONLY: single-forward stationarity, not static standing or full KKT');fig.tight_layout();fig.savefig(out/'stationarity.png');plt.close(fig)
    print(out)
    for r in records:print(r['case'],json.dumps(r['groups'],indent=2))


if __name__=='__main__':main()
