"""NEW bounded two-case same-forward endpoint residual observation."""
from pathlib import Path
import json, hashlib, sys, uuid
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
import numpy as np
from tools_s2_dynamic_residual import entries, analyze
ROOT=Path(__file__).resolve().parent
POSE=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
BUDGET=256*1024*1024


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def provenance(cp):
    from flygym.utils.assets_lazy_loading import get_cache_root
    cache=get_cache_root()/'neuromechfly_musculoskeletal_meshes_20260623a'
    assets=[]
    for node in ET.parse(cp.CANONICAL_XML).getroot().findall('./asset/mesh'):
        p=(cp.CANONICAL_XML.parent/node.attrib['file']).resolve()
        if not p.is_file(): p=cache/Path(node.attrib['file']).name
        if not p.is_file(): raise RuntimeError(f'Local asset missing; no download: {p}')
        assets.append(p)
    # Original availability-probe set: 71 mesh plus six source/compile inputs =77.
    paths=[Path(cp.__file__),Path(cp.CANONICAL_XML),POSE,ROOT/'venv_body/lib/python3.12/site-packages/flygym/compose/fly/musculoskeletal.py',ROOT/'venv_body/lib/python3.12/site-packages/flygym/utils/assets_lazy_loading.py',Path('/home/sensen/Desktop/engine修复验收/S2-dynamic-availability-20261014/availability_probe.py')]+assets
    paths+= [ROOT/n for n in ('run_s2_dynamic_residual.py','tools_s2_dynamic_residual.py','test_s2_dynamic_residual.py','tools_verify_s2_dynamic_residual.py','S2_DYNAMIC_RESIDUAL_PROTOCOL.zh-CN.md')]
    return {str(p):sha(p) for p in dict.fromkeys(paths)}


def main():
    import mujoco
    import engine.canonical_physics as cp
    assert mujoco.__version__=='3.9.0'
    out=Path('/home/sensen/Desktop/engine修复验收')/('S2-dynamic-residual-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]);out.mkdir(exist_ok=False)
    before=provenance(cp);assert sha(cp.CANONICAL_XML)==cp.CANONICAL_XML_SHA256
    with np.load(POSE,allow_pickle=False) as a: pose=a['projected_seated_qpos'].copy()
    records=[]
    for dt,ratio in ((25e-6,1),(12.5e-6,2)):
        m=cp.load_model();m.opt.timestep=dt;m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
        assert int(m.opt.solver)==2 and m.opt.noslip_iterations==0
        assert m.nv<=256
        d=mujoco.MjData(m);d.qpos[:]=pose;d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0
        initial={'initial_'+k:getattr(d,k).copy() for k in ('qpos','qvel','act','ctrl')}
        fixed=('time_s','qpos','qvel','act','ctrl','qacc','qacc_smooth','qfrc_smooth','qfrc_constraint','qacc_warmstart','solver_niter','warning_counts','ne','nf','nl','nefc','nJ','ncon','nA','sparse','step_time_s','step_ctrl')
        data={k:[] for k in fixed}
        efcfields=('aref','D','R','force','state','type','id','b','vel','pos','margin','KBIP','frictionloss')
        contactfields=('geom','pos','dist','frame','dim','exclude','efc_address','friction','solref','solreffriction','solimp','includemargin','mu','local6')
        ragged={**{'efc_'+k:[] for k in efcfields},**{'contact_'+k:[] for k in contactfields},**{k:[] for k in ('J_values','J_rownnz','J_rowadr','J_colind','solver_stats')}}
        offsets={k:[0] for k in ('efc','contact','J','solver_stats')};used=sum(v.nbytes for v in initial.values())
        def add(k,v,target=data):
            nonlocal used
            v=np.asarray(v).copy()
            if not np.isfinite(v).all(): raise RuntimeError('nonfinite '+k)
            used+=v.nbytes
            if used>BUDGET: raise RuntimeError('case byte budget exceeded')
            target[k].append(v)
        for step in range(round(.01/dt)):
            d.ctrl[:]=0;d.ctrl[0]=.1 if step<round(.001/dt) else 0
            add('step_time_s',d.time);add('step_ctrl',d.ctrl)
            mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
            if np.any(d.warning.number): raise RuntimeError('MuJoCo warning')
            if (step+1)%ratio: continue
            if d.ncon>512 or d.nefc>4096 or d.efc_J.size>1048576: raise RuntimeError('capture size bound')
            sparse=bool(mujoco.mj_isSparse(m));n=int(d.nefc)
            vals=d.efc_J.copy();nn=d.efc_J_rownnz[:n].copy() if sparse else np.zeros(n,dtype=int);adr=d.efc_J_rowadr[:n].copy() if sparse else np.zeros(n,dtype=int);cols=d.efc_J_colind.copy() if sparse else np.zeros(len(vals),dtype=int)
            entries(vals,m.nv,n,sparse,nn,adr,cols,d.nJ)
            for k in fixed:
                if k.startswith('step_'): continue
                v=d.time if k=='time_s' else sparse if k=='sparse' else d.warning.number if k=='warning_counts' else getattr(d,k)
                add(k,v)
            for k in efcfields: add('efc_'+k,getattr(d,'efc_'+k)[:n],ragged)
            for k,v in zip(('J_values','J_rownnz','J_rowadr','J_colind'),(vals,nn,adr,cols)): add(k,v,ragged)
            contacts={k:[] for k in contactfields}
            for i in range(d.ncon):
                c=d.contact[i];f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
                for k in contactfields: contacts[k].append([c.geom1,c.geom2] if k=='geom' else f if k=='local6' else np.asarray(getattr(c,k)).copy())
            widths={'geom':2,'pos':3,'frame':9,'friction':5,'solref':2,'solreffriction':2,'solimp':5,'local6':6}
            ints={'geom','dim','exclude','efc_address'}
            for k in contactfields: add('contact_'+k,np.asarray(contacts[k],dtype=int if k in ints else float).reshape((-1,widths[k]) if k in widths else (-1,)),ragged)
            # Bounded active solver records, not all allocated stats slots.
            stats=[]
            for s in d.solver[:int(np.max(d.solver_niter))]: stats.append([s.improvement,s.gradient,s.lineslope,s.nactive,s.nchange,s.neval,s.nupdate])
            add('solver_stats',np.asarray(stats).reshape(-1,7),ragged)
            for k,size in (('efc',n),('contact',d.ncon),('J',len(vals)),('solver_stats',len(stats))): offsets[k].append(offsets[k][-1]+size)
            used+=4*8
            if used>BUDGET: raise RuntimeError('budget')
        a={**initial,**{k:np.asarray(v) for k,v in data.items()},**{k:np.concatenate(v,axis=0) for k,v in ragged.items()},**{k+'_offsets':np.asarray(v,dtype=np.int64) for k,v in offsets.items()}}
        for prefix in ('geom','pair'):
            for k in ('solref','solimp','friction','margin','gap','priority','solmix','solreffriction'):
                key=prefix+'_'+k
                if hasattr(m,key): a['compiled_'+key]=np.asarray(getattr(m,key)).copy()
        for k in ('o_solref','o_solimp','o_friction','o_margin'): a['option_'+k]=np.asarray(getattr(m.opt,k)).copy()
        assert sum(v.nbytes for v in a.values())<=BUDGET
        assert len(a['time_s'])==400 and np.allclose(a['time_s'],np.arange(1,401)*25e-6,rtol=0,atol=1e-14)
        name=f'native_floor_pulse_{dt}.npz';np.savez_compressed(out/name,**a)
        opts={k:float(getattr(m.opt,k)) for k in ('solver','jacobian','cone','integrator','iterations','tolerance','ls_iterations','ls_tolerance','noslip_iterations','noslip_tolerance','enableflags','disableflags','timestep','impratio')}
        doftypes=[]
        for j in m.dof_jntid:
            t=int(m.jnt_type[j]);local=len(doftypes)-int(m.jnt_dofadr[j]);doftypes.append('translation/model_force' if t==2 or (t==0 and local<3) else 'rotation/model_torque')
        records.append(dict(npz=name,sha256=sha(out/name),dt_s=dt,nv=int(m.nv),dof_types=doftypes,options=opts,numeric_bytes=sum(v.nbytes for v in a.values()),AR='not_constructed',M='not_requested',J_metadata='endpoint-local; dense metadata placeholders have no sparse semantics'))
        del a,data,ragged,d,m
    after=provenance(cp);assert before==after
    manifest=dict(records=records,source_hashes_before=before,source_hashes_after=after,original_availability_provenance_count=77,environment={'python':sys.executable,'mujoco':mujoco.__version__},fixed_cases=2,policy='REPORT_ONLY',physical_cause='unknown')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');analyze(out);print(out)


if __name__=='__main__': main()
