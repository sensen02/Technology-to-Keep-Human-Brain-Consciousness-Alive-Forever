"""Two fresh single-forward observations, no integration; separate from frozen runner."""
from pathlib import Path
import json
import hashlib
import sys
from datetime import datetime, timezone
import numpy as np
from tools_s2_final_state_contract import validate_capture, STAT_COLUMNS

ROOT = Path(__file__).resolve().parent
OLD = Path('/home/sensen/Desktop/engine修复验收/S2-dynamic-residual-20261014T211552Z-7fbf1130')
POSE = ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
ARTIFICIAL_XML = '''<mujoco model="artificial_two_spheres_interface_only">
<option solver="Newton" cone="pyramidal" gravity="0 0 -9.81"/>
<worldbody><geom name="floor" type="plane" size="5 5 .1"/>
<body name="left" pos="-1 0 .49"><freejoint/><geom type="sphere" size=".5" mass="1"/></body>
<body name="right" pos="1 0 .49"><freejoint/><geom type="sphere" size=".5" mass="1"/></body>
</worldbody></mujoco>'''


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frozen():
    manifest=json.loads((OLD/'manifest.json').read_text())
    paths=dict(manifest['source_hashes_before'])
    paths.update({str(OLD/r['npz']):r['sha256'] for r in manifest['records']})
    assert all(sha(p)==h for p,h in paths.items())
    return paths


def capture(m, d, mujoco):
    n=int(d.nefc);ni=int(d.nisland);nv=int(m.nv)
    assert n<=4096 and ni<=128 and nv<=256
    assert not mujoco.mj_isSparse(m), 'this adapter probe implements actual dense layout only'
    cap=int(mujoco.mjNSOLVER);slots=int(mujoco.mjNISLAND)
    enabled=not bool(int(m.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_ISLAND))
    meta=dict(mujoco_version=mujoco.__version__,solver=int(m.opt.solver),noslip_iterations=int(m.opt.noslip_iterations),islands_enabled=enabled,nefc=n,nisland=ni,mjNSOLVER=cap,mjNISLAND=slots)
    assert len(d.solver)==slots*cap
    for key in STAT_COLUMNS: assert np.asarray(getattr(d.solver,key)).shape==(slots*cap,)
    fields=dict(efc_state_global_raw=d.efc_state[:n].copy(),solver_niter=d.solver_niter.copy())
    stats=np.zeros((slots*cap,7)); observed=np.zeros(slots*cap,dtype=bool)
    used=ni if enabled and ni>0 else 1 if n else 0
    for island in range(used):
        for iteration in range(min(int(d.solver_niter[island]),cap)):
            address=island*cap+iteration
            for col,key in enumerate(STAT_COLUMNS):
                value=getattr(d.solver[address],key)
                assert value==getattr(d.solver,key)[address], 'list/vector indexing disagreement'
                stats[address,col]=value
            observed[address]=True
    fields['solver_stats_flat']=stats
    if enabled and ni>0:
        for key,size in (('island_iefcadr',ni),('island_nefc',ni),('map_iefc2efc',n),('map_efc2iefc',n),('iefc_state',n)):
            fields[key]=getattr(d,key)[:size].copy()
    result=validate_capture(meta,fields)
    a=dict(fields)
    a.update(final_state=result['final_state'],solver_stats_observed_mask=observed,
             qacc=d.qacc.copy(),J=d.efc_J.copy().reshape(n,nv),aref=d.efc_aref[:n].copy(),D=d.efc_D[:n].copy(),R=d.efc_R[:n].copy(),force=d.efc_force[:n].copy(),type=d.efc_type[:n].copy(),id=d.efc_id[:n].copy(),warnings=d.warning.number.copy())
    a['solver_slot_offsets']=np.array([s['slot']*cap for s in result['statistics']],dtype=np.int64)
    r=a['J']@a['qacc']-a['aref'];mask=np.isin(a['type'],[3,4,5,6])
    error=a['force']-np.maximum(0,-a['D']*r)
    a['r']=r;a['piecewise_error']=error;a['piecewise_applicable']=mask
    report=dict(metadata=meta,source=result['source'],mismatch_count=result['mismatch_count'],solver_wrapper={'container':type(d.solver).__name__,'length':len(d.solver),'field_shape':list(np.asarray(d.solver.gradient).shape),'unused_slots':'zero placeholders; observed mask authoritative'},statistics=[{k:v for k,v in s.items() if k!='values'} for s in result['statistics']],force_error_policy='REPORT_ONLY',piecewise_absmax=float(np.max(np.abs(error[mask]))) if np.any(mask) else None,non_applicable_rows=int(np.sum(~mask)),final_vs_r_branch_mismatch=int(np.sum(a['final_state'][mask]!=np.where(r[mask]<0,1,0))))
    assert sum(v.nbytes for v in a.values())<=64*1024*1024
    return a,report


def main():
    import mujoco
    import engine.canonical_physics as cp
    assert mujoco.__version__=='3.9.0'
    before=frozen()
    sources={str(ROOT/n):sha(ROOT/n) for n in ('run_s2_forward_adapter.py','tools_s2_final_state_contract.py','S2_FORWARD_ADAPTER_PROTOCOL.zh-CN.md','S2_FINAL_STATE_CAPTURE_CONTRACT.zh-CN.md')}
    out=Path('/home/sensen/Desktop/engine修复验收')/('S2-forward-adapter-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(exist_ok=False)
    (out/'artificial-model.xml').write_text(ARTIFICIAL_XML)
    reports=[];plots=[]
    for name in ('canonical','artificial'):
        m=cp.load_model() if name=='canonical' else mujoco.MjModel.from_xml_string(ARTIFICIAL_XML)
        d=mujoco.MjData(m)
        if name=='canonical':
            with np.load(POSE,allow_pickle=False) as pose:d.qpos[:]=pose['projected_seated_qpos']
            d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0
        initial={k:getattr(d,k).copy() for k in ('qpos','qvel','act','ctrl')};t=float(d.time)
        mujoco.mj_forward(m,d)  # exactly one per model, never step
        assert float(d.time)==t and not np.any(d.warning.number)
        assert all(getattr(d,k).tobytes()==v.tobytes() for k,v in initial.items())
        a,report=capture(m,d,mujoco)
        for k,v in initial.items():a['initial_'+k]=v;a['after_'+k]=getattr(d,k).copy()
        a['time_before_after']=np.array([t,d.time])
        filename=name+'.npz';np.savez_compressed(out/filename,**a)
        report.update(case=name,npz=filename,sha256=sha(out/filename),forward_calls=1,step_calls=0,state_unchanged=True,nv=int(m.nv),ncon=int(d.ncon),options={k:float(getattr(m.opt,k)) for k in ('integrator','timestep','disableflags','enableflags','cone')},multisland_coverage=bool(d.nisland>=2))
        reports.append(report);plots.append((name,a['efc_state_global_raw'].copy(),a['final_state'].copy(),a['piecewise_error'].copy()))
    assert before==frozen() and all(sha(p)==h for p,h in sources.items())
    manifest=dict(environment={'python':sys.executable,'mujoco':mujoco.__version__},cases=reports,frozen_hashes=before,source_hashes=sources,artificial_xml_sha256=sha(out/'artificial-model.xml'),scope='Two single-forward observations only; no integration; not retrospective recovery of old island state')
    (out/'report.json').write_text(json.dumps(manifest,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row,(name,raw,final,error) in enumerate(plots):
        axes[row,0].plot(raw,'o',label='global raw');axes[row,0].plot(final,'x',label='final mapped');axes[row,0].set_title(name+' state labels (single forward)');axes[row,0].legend()
        axes[row,1].plot(error,'o-');axes[row,1].set_title(name+' piecewise force error / REPORT ONLY')
        for ax in axes[row]:ax.set_xlabel('Endpoint-local constraint row');ax.grid(alpha=.3)
    fig.suptitle('Interface probe, no integration; artificial case is not fly behavior');fig.tight_layout();fig.savefig(out/'adapter.png');plt.close(fig)
    print(out);print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
