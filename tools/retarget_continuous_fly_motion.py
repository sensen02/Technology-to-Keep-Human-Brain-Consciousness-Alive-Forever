"""Offline NMF2 geometry -> bounded native-model targets; never a runtime controller.
Run fit, verify (fresh process, no optimization), then seal after all review.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import traceback
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/sensen/Desktop/engine修复验收/s1-continuous-fly-kinematics/20261102T004326Z')
INTERFACE = Path('/home/sensen/Desktop/engine修复验收/s4-kinematic-training-interface/20261102T004326Z')
MODEL = Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/RUN/MODEL.xml')
OUT = Path('/home/sensen/Desktop/engine修复验收/s4-real-motion-retarget/20261102T011240Z')
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
MARKERS = ['ThC', 'CTr', 'FTi', 'TiTa', 'Claw']
TEMPORAL = 1e-6


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, obj):
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=encode) + '\n')


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def inputs():
    return [Path(__file__), MODEL, SOURCE/'NMF2_CONTINUOUS.npz', SOURCE/'annotated_3d_walking_kinematics.csv',
            SOURCE/'NMF2_FIELDS_AND_TRANSFORMS.json', INTERFACE/'JOINT_AND_MARKER_MAPPING.json',
            INTERFACE/'LOSS_SPEC.zh-CN.md', INTERFACE/'target_schema.json']


def prereg():
    return dict(stage='S1/S4 offline target conversion; not physics walking or learning',
        source_units='NOT_ESTABLISHED', absolute_physical_units='NOT_ESTABLISHED',
        world_root=None, root_valid=False, source_Th='fixed (0,0,200); never used as measured world root',
        model_root='canonical offline Thorax frame: qpos[:7]=[0,0,0,1,0,0,0]; no world/ground interpretation',
        calibration='one proper Kabsch rotation of centered training-only mean ThC layout to model Coxa layout; independent centroid/norm normalization; no scale or translation applied to trajectories; never refit on validation/test',
        targets='source first three unit segment directions rotated by locked R, multiplied by corresponding fixed model marker-chord lengths; cumulative points start at exact model Coxa anchor',
        excluded='Claw and TiTa->Claw: distal claw is not Tarsus5 origin; no established tip offset, no distal direction loss either',
        morphology='front CTr->FTi is Trochanter+Femur composite chord; middle/hind CTr is Femur origin at virtual Trochanter hinge anchor. Source TiTa->Claw spans multiple tarsal bodies; no 1:1 bone claim',
        objective='per-leg first-three unit-direction component residuals plus sqrt(1e-6)*(q-previous)/native_range; first frame uses bounded spring reference. No angle copying, no root optimization',
        solver=dict(method='scipy least_squares trf, 2-point numerical Jacobian', max_nfev=300, ftol=1e-9, xtol=1e-9, gtol=1e-9,
                    warmstart='chronological all frames; validation/test may inherit previous conversion q, not learned parameters'),
        split='exact source chronological 22/7/8; per-frame IK is target conversion, not held-out learning generalization',
        diagnostics=dict(saturation_rad=1e-5, branch_jump_rad=0.5, rank_relative_tolerance=1e-6),
        validation='fresh process one compile via flygym._load_mjcf; saved qpos FK via mj_forward only, no optimization; independently reconstruct source/R/targets/XML joint definitions and bounds; atol 1e-10',
        planned_calls=dict(fit_model_compiles=1, verification_model_compiles=1, verification_forward=37, body_steps=0, neural_steps=0, training=0),
        runtime_marker_loss_enabled=False, physiological_joint_angles=False)


def compile_model():
    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    spec = _load_mjcf(MODEL)
    assets = {}
    for mesh in spec.meshes:
        if mesh.file:
            p = Path(mesh.file)
            assets[str(p)] = sha(p)
    model = spec.compile()
    return mj, model, assets


def setup(mj, m):
    data = mj.MjData(m)
    thorax = m.body('Thorax').id
    bodies = []
    joint_ids = []
    for leg in LEGS:
        bodies.append([m.body(leg+'Coxa').id, m.body(leg+('Trochanter' if leg[1]=='F' else 'Femur')).id,
                       m.body(leg+'Tibia').id, m.body(leg+'Tarsus1').id, m.body(leg+'Tarsus5').id])
        joint_ids.append([m.joint('joint_'+leg+suffix).id for suffix in
                          ['Coxa_yaw','Coxa_pitch','Coxa_roll','Trochanter_yaw','Trochanter_pitch','Trochanter_roll','Tibia_pitch']])
    bodies, joint_ids = np.array(bodies), np.array(joint_ids)
    addresses = m.jnt_qposadr[joint_ids]
    bounds = m.jnt_range[joint_ids].copy()
    q = m.qpos_spring.copy()
    q[:7] = [0,0,0,1,0,0,0]
    q[addresses] = np.clip(q[addresses], bounds[:,:,0], bounds[:,:,1])
    data.qpos[:] = q
    mj.mj_kinematics(m, data)
    p = data.xpos[bodies].copy()
    lengths = np.linalg.norm(np.diff(p[:,:4], axis=1), axis=-1)
    return data, thorax, bodies, joint_ids, addresses, bounds, q, p, lengths


def unit(a):
    return a / np.maximum(np.linalg.norm(a,axis=-1,keepdims=True), 1e-15)


def calibration(z, anchors):
    train = z['split']==0
    a = np.mean(z['xyz_source'][train,:,0],axis=0)
    ac = a-a.mean(axis=0)
    bc = anchors-anchors.mean(axis=0)
    an, bn = ac/np.linalg.norm(ac), bc/np.linalg.norm(bc)
    u,s,vh = np.linalg.svd(an.T@bn)
    correction = np.diag([1.,1.,np.linalg.det(u@vh)])
    # row vectors source @ R.T
    R = (u@correction@vh).T
    return R, dict(method='proper Kabsch on normalized centered training ThC mean',
        training_source_frames=z['source_frame'][train], source_anchor_mean=a,
        source_centroid=a.mean(axis=0), model_centroid=anchors.mean(axis=0),
        source_layout_frobenius_norm=np.linalg.norm(ac),model_layout_frobenius_norm=np.linalg.norm(bc),
        singular_values=s, R_source_to_model=R, determinant=np.linalg.det(R),
        normalized_layout_residual=np.linalg.norm(an@R.T-bn,axis=1),
        translation_applied=False,scale_applied=False,world_root=None,root_valid=False,
        absolute_physical_units='NOT_ESTABLISHED',
        source_training_bone_length_median=np.median(z['bone_lengths_source'][train],axis=0),
        source_training_bone_length_cv=np.std(z['bone_lengths_source'][train],axis=0)/np.mean(z['bone_lengths_source'][train],axis=0))


def targets(z,R,anchors,lengths):
    diff = np.diff(z['xyz_source'],axis=2)
    smask = z['marker_valid'][:,:,:-1] & z['marker_valid'][:,:,1:] & (np.linalg.norm(diff,axis=-1)>1e-12)
    dirs = unit(diff)@R.T
    dmask = smask.copy(); dmask[:,:,3]=False
    refs = np.full(z['xyz_source'].shape,np.nan)
    pmask = z['marker_valid'].copy();pmask[:,:,4]=False
    refs[:,:,0] = anchors
    for k in range(3):
        pmask[:,:,k+1] &= pmask[:,:,k] & dmask[:,:,k]
        refs[:,:,k+1] = refs[:,:,k]+dirs[:,:,k]*lengths[:,k,None]
    refs[~pmask] = np.nan
    return dirs,dmask,refs,pmask


def fit():
    from scipy.optimize import least_squares
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'RETARGETED_TARGETS.npz').exists(), 'Refuse to overwrite completed targets'
    dump(OUT/'PREREGISTRATION.json',prereg())
    dump(OUT/'SOURCE_IDENTITY.json',{str(p):sha(p) for p in inputs()})
    (OUT/'EXECUTION_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    counters=dict(model_compile_attempts=0,model_compile=0,kinematics=0,forward=0,body_steps=0,neural_steps=0,training=0,ik_solves=0)
    try:
        z=load(SOURCE/'NMF2_CONTINUOUS.npz')
        assert z['xyz_source'].shape==(37,6,5,3)
        assert z['leg_names'].tolist()==LEGS and z['marker_names'].tolist()==MARKERS
        assert np.bincount(z['split']).tolist()==[22,7,8]
        counters['model_compile_attempts']+=1
        mj,m,assets=compile_model();counters['model_compile']+=1
        dump(OUT/'ASSET_IDENTITY.json',assets)
        data,thorax,bodies,jids,adr,bounds,base,p0,lengths=setup(mj,m);counters['kinematics']+=1
        assert (m.nq,m.nv,m.nbody,m.nmesh,m.nu,m.nmocap,m.neq)==(49,48,73,71,90,0,0)
        R,ledger=calibration(z,p0[:,0]);dump(OUT/'TRANSFORM_AND_UNIT_LEDGER.json',ledger)
        dirs,dmask,refs,pmask=targets(z,R,p0[:,0],lengths)
        mapping=[]
        for li,leg in enumerate(LEGS):
            rows=[]
            for k,name in enumerate(MARKERS):
                body=m.body(int(bodies[li,k])).name
                rows.append(dict(source_marker=name,model_body=body,local_point=[0,0,0],
                                 position_loss_valid=k<4,meaning='body origin' if k<4 else 'diagnostic tarsus5 origin; NOT source Claw',
                                 joint_anchor=m.jnt_pos[jids[li,3]].tolist() if k==1 else None))
            mapping.append(dict(leg=leg,markers=rows,model_chord_lengths=lengths[li],
                                excluded_tarsus_chain=[leg+'Tarsus'+str(n) for n in range(1,6)]))
        dump(OUT/'MODEL_MARKER_CONTRACT.json',dict(legs=mapping,root_frame='canonical Thorax',
            model_counts={k:int(getattr(m,k)) for k in ['nq','nv','nbody','njnt','ngeom','nsite','nmesh','nu','nmocap','neq']},
            joints=[dict(name=m.joint(int(j)).name,body=m.body(int(m.jnt_bodyid[j])).name,
                joint_id=int(j),qpos_address=int(m.jnt_qposadr[j]),dof_address=int(m.jnt_dofadr[j]),
                axis=m.jnt_axis[j],pos=m.jnt_pos[j],range_rad=m.jnt_range[j],limited=bool(m.jnt_limited[j])) for j in jids.flat],
            joint_angle_targets='derived regularized IK branch; redundant DOFs not uniquely observed',
            runtime_marker_loss_enabled=False))
        qposes=[];achieved=[];statuses=[];ranks=[];singular=[]
        previous=base.copy()
        for t in range(37):
            q=previous.copy();frame_status=[];frame_rank=[];frame_singular=[]
            for li,leg in enumerate(LEGS):
                address=adr[li];lo,hi=bounds[li].T;span=hi-lo
                prior=previous[address].copy()
                def geometric(v):
                    data.qpos[:]=q;data.qpos[address]=v
                    mj.mj_kinematics(m,data);counters['kinematics']+=1
                    pts=data.xpos[bodies[li,:4]]
                    direction=unit(np.diff(pts,axis=0))
                    return (direction-dirs[t,li,:3])[dmask[t,li,:3]].ravel()
                def fun(v):
                    return np.r_[geometric(v),np.sqrt(TEMPORAL)*(v-prior)/span]
                result=least_squares(fun,np.clip(prior,lo+1e-12,hi-1e-12),bounds=(lo,hi),
                    method='trf',max_nfev=300,ftol=1e-9,xtol=1e-9,gtol=1e-9)
                q[address]=result.x;counters['ik_solves']+=1
                sv=np.linalg.svd(result.jac[:-7],compute_uv=False)
                rank=int(np.sum(sv>sv[0]*1e-6)) if len(sv) and sv[0]>0 else 0
                frame_rank.append(rank);frame_singular.append(sv.tolist())
                frame_status.append(dict(leg=leg,success=bool(result.success),status=int(result.status),
                    nfev=int(result.nfev),cost=float(result.cost),optimality=float(result.optimality),message=str(result.message)))
            data.qpos[:]=q;mj.mj_kinematics(m,data);counters['kinematics']+=1
            qposes.append(q.copy());achieved.append(data.xpos[bodies].copy());statuses.append(frame_status)
            ranks.append(frame_rank);singular.append(frame_singular);previous=q.copy()
            print('converted source frame',int(z['source_frame'][t]),flush=True)
        qposes=np.array(qposes);achieved=np.array(achieved)
        native=qposes[:,adr]
        sat=np.minimum(native-bounds[None,:,:,0],bounds[None,:,:,1]-native)<=1e-5
        achieved_dirs=unit(np.diff(achieved,axis=2))
        angular=np.arccos(np.clip(np.sum(achieved_dirs*dirs,axis=-1),-1,1));angular[~dmask]=np.nan
        point_res=np.linalg.norm(achieved-refs,axis=-1);point_res[~pmask]=np.nan
        jumps=np.abs(np.diff(native,axis=0))
        speed=np.diff(native,axis=0)/np.diff(z['time_relative_s'])[:,None,None]
        solver_ok=np.array([[r['success'] for r in f] for f in statuses])
        np.savez_compressed(OUT/'RETARGETED_TARGETS.npz',source_xyz=z['xyz_source'],source_marker_mask=z['marker_valid'],
            source_coordinate_mask=z['coordinate_valid'],source_frame=z['source_frame'],source_row_index=np.arange(37),
            source_trial_id=z['trial_id'],time_relative_s=z['time_relative_s'],split=z['split'],leg_names=z['leg_names'],
            marker_names=z['marker_names'],qpos=qposes,native_joint_qpos=native,joint_qpos_addresses=adr,
            native_joint_target_mask=np.broadcast_to(solver_ok[:,:,None],native.shape),
            joint_bounds_rad=bounds,model_body_ids=bodies,reference_points=refs,achieved_points=achieved,
            model_target_mask=pmask,reference_directions=dirs,direction_target_mask=dmask,
            achieved_directions=achieved_dirs,point_residual_model_length=point_res,direction_residual_rad=angular,
            bound_saturation=sat,delta_q_rad=np.diff(native,axis=0),velocity_rad_per_nominal_s=speed,
            branch_jump_mask=jumps>0.5,rotation_source_to_model=R,model_chord_lengths=lengths,
            model_anchor_points=p0[:,0],geometric_jacobian_rank=np.array(ranks),
            root_valid=np.zeros(37,dtype=bool),runtime_marker_loss_enabled=np.array(False))
        dump(OUT/'SOLVER_DIAGNOSTICS.json',dict(statuses=statuses,geometric_singular_values=singular,
            rank_note='rank computed from numerical geometric Jacobian before temporal regularizer; IK joint targets choose a branch, not measured joint angles'))
        summary=dict(solver_successes=int(solver_ok.sum()),solver_solves=222,
            direction_error_mean_deg=float(np.nanmean(angular)*180/np.pi),direction_error_max_deg=float(np.nanmax(angular)*180/np.pi),
            point_error_mean_model_length=float(np.nanmean(point_res)),point_error_max_model_length=float(np.nanmax(point_res)),
            saturation_count=int(sat.sum()),jump_over_half_rad_count=int((jumps>.5).sum()),max_abs_delta_q_rad=float(jumps.max()),
            max_abs_velocity_rad_per_nominal_s=float(np.abs(speed).max()),geometric_rank_min=int(np.min(ranks)),geometric_rank_max=int(np.max(ranks)),
            absolute_physical_units='NOT_ESTABLISHED',world_root=None,physics_walk=False,training=False,
            per_leg={},per_split={})
        for li,leg in enumerate(LEGS):
            summary['per_leg'][leg]=dict(direction_mean_deg=float(np.nanmean(angular[:,li])*180/np.pi),
                direction_max_deg=float(np.nanmax(angular[:,li])*180/np.pi),point_mean_model_length=float(np.nanmean(point_res[:,li])),
                saturation_count=int(sat[:,li].sum()),failed_solves=int((~solver_ok[:,li]).sum()))
        for code,name in enumerate(['train','validation','test']):
            sel=z['split']==code
            summary['per_split'][name]=dict(frames=int(sel.sum()),direction_mean_deg=float(np.nanmean(angular[sel])*180/np.pi),
                point_mean_model_length=float(np.nanmean(point_res[sel])))
        dump(OUT/'RESULT.json',summary)
        dump(OUT/'EXECUTION.json',dict(success=True,counters=counters,python=sys.version,executable=sys.executable))
    except Exception:
        dump(OUT/'FAILED_EXECUTION.json',dict(traceback=traceback.format_exc(),counters=counters))
        raise


def verify():
    """Independent path: XML contracts + raw source + saved qpos -> full mj_forward."""
    assert not (OUT/'MANIFEST_SHA256.json').exists(), 'Sealed evidence is read-only'
    z=load(SOURCE/'NMF2_CONTINUOUS.npz');a=load(OUT/'RETARGETED_TARGETS.npz')
    identity=json.loads((OUT/'SOURCE_IDENTITY.json').read_text())
    assert all(sha(p)==h for p,h in identity.items())
    mj,m,assets=compile_model()
    assert assets==json.loads((OUT/'ASSET_IDENTITY.json').read_text())
    data=mj.MjData(m)
    tree=ET.parse(MODEL);xmlj={j.get('name'):j for j in tree.iter('joint') if j.get('name')}
    contract=json.loads((OUT/'MODEL_MARKER_CONTRACT.json').read_text())
    oldmapping=json.loads((INTERFACE/'JOINT_AND_MARKER_MAPPING.json').read_text())['joints']
    for j in contract['joints']:
        jid=m.joint(j['name']).id;x=xmlj[j['name']]
        old=next(k for k in oldmapping if k['name']==j['name'])
        for key,value in [('axis',m.jnt_axis[jid]),('pos',m.jnt_pos[jid]),('range',m.jnt_range[jid])]:
            assert np.allclose(np.fromstring(x.get(key),sep=' '),value,atol=1e-14,rtol=0)
        assert np.array_equal(m.jnt_axis[jid],np.array(j['axis']))
        assert np.allclose(m.jnt_axis[jid],old['axis_body_local'],atol=1e-14,rtol=0)
        assert np.array_equal(m.jnt_range[jid],j['range_rad'])
        assert m.jnt_qposadr[jid]==j['qpos_address']==old['qpos_address']
        assert m.jnt_dofadr[jid]==old['dof_address'] and bool(m.jnt_limited[jid])
    assert np.array_equal(a['source_xyz'],z['xyz_source'],equal_nan=True)
    for own,original in [('source_frame','source_frame'),('time_relative_s','time_relative_s'),('split','split'),('source_marker_mask','marker_valid')]:
        assert np.array_equal(a[own],z[original])
    assert np.array_equal(a['source_row_index'],np.arange(37))
    assert np.bincount(a['split']).tolist()==[22,7,8]
    assert np.array_equal(a['qpos'][:,:7],np.tile([0,0,0,1,0,0,0],(37,1)))
    assert not a['root_valid'].any() and not a['runtime_marker_loss_enabled'].item()
    assert not a['model_target_mask'][:,:,4].any() and not a['direction_target_mask'][:,:,3].any()
    assert np.isnan(a['reference_points'][:,:,4]).all()
    pts=[]
    for q in a['qpos']:
        data.qpos[:]=q;mj.mj_forward(m,data);pts.append(data.xpos[a['model_body_ids']].copy())
    pts=np.array(pts);fkerr=float(np.max(np.abs(pts-a['achieved_points'])))
    assert fkerr<1e-10
    adr=a['joint_qpos_addresses'];q=a['qpos'][:,adr];b=a['joint_bounds_rad']
    assert np.all(q>=b[None,:,:,0]-1e-12) and np.all(q<=b[None,:,:,1]+1e-12)
    assert np.array_equal(q,a['native_joint_qpos'])
    # Reconstruct Kabsch independently (not calling calibration/targets/setup).
    src=z['xyz_source'][z['split']==0,:,0].mean(axis=0)
    sx=src-src.mean(axis=0);sx/=np.linalg.norm(sx)
    anchors=pts[0,:,0];mx=anchors-anchors.mean(axis=0);mx/=np.linalg.norm(mx)
    u,_,vh=np.linalg.svd(sx.T@mx);u[:,-1]*=np.linalg.det(u@vh);r=(u@vh).T
    assert np.max(np.abs(r-a['rotation_source_to_model']))<1e-12
    assert abs(np.linalg.det(r)-1)<1e-12 and np.max(np.abs(r@r.T-np.eye(3)))<1e-12
    seg=z['xyz_source'][:,:,1:]-z['xyz_source'][:,:,:-1]
    dr=seg/np.linalg.norm(seg,axis=-1,keepdims=True)@r.T
    assert np.max(np.abs(dr-a['reference_directions']))<1e-12
    lengths=np.linalg.norm(np.diff(pts[:,:,:4],axis=2),axis=-1)
    assert np.max(np.abs(lengths-a['model_chord_lengths'][None]))<1e-10
    ref=np.zeros((37,6,4,3));ref[:,:,0]=anchors
    for k in range(3):ref[:,:,k+1]=ref[:,:,k]+dr[:,:,k]*a['model_chord_lengths'][:,k,None]
    terr=float(np.max(np.abs(ref-a['reference_points'][:,:,:4])))
    assert terr<1e-10
    ar=np.linalg.norm(pts[:,:,:4]-ref,axis=-1)
    assert np.max(np.abs(ar-a['point_residual_model_length'][:,:,:4]))<1e-10
    d=np.diff(pts,axis=2);d/=np.linalg.norm(d,axis=-1,keepdims=True)
    ang=np.arccos(np.clip((d*dr).sum(axis=-1),-1,1))
    assert np.max(np.abs(ang[:,:,:3]-a['direction_residual_rad'][:,:,:3]))<1e-7
    vel=np.diff(q,axis=0)/np.diff(z['time_relative_s'])[:,None,None]
    assert np.array_equal(vel,a['velocity_rad_per_nominal_s'])
    sat=np.minimum(q-b[None,:,:,0],b[None,:,:,1]-q)<=1e-5
    assert np.array_equal(sat,a['bound_saturation'])
    assert np.array_equal(np.abs(np.diff(q,axis=0))>.5,a['branch_jump_mask'])
    dump(OUT/'INDEPENDENT_FK_VERIFICATION.json',dict(passed=True,fresh_process=True,optimization_rerun=False,
        counters=dict(model_compile_attempts=1,model_compile=1,forward=37,body_steps=0,neural_steps=0,training=0,ik_solves=0),
        fk_max_abs_model_length=fkerr,target_max_abs_model_length=terr,frames=37,joints=42,
        checks=['raw source / frame / masks / splits','XML + prior mapping axes / addresses / bounds','fixed canonical root',
            'training-only single proper rotation','model fixed marker-chord lengths','saved qpos FK / target / residual',
            'native limits / saturation / jumps / nominal speeds','Claw missing mask; root unavailable'],
        absolute_physical_units='NOT_ESTABLISHED'))


def seal():
    assert not (OUT/'MANIFEST_SHA256.json').exists()
    assert json.loads((OUT/'INDEPENDENT_FK_VERIFICATION.json').read_text())['passed']
    assert json.loads((OUT/'REVIEW.json').read_text())['passed']
    assert all(sha(p)==h for p,h in json.loads((OUT/'SOURCE_IDENTITY.json').read_text()).items())
    files=[p for p in OUT.rglob('*') if p.is_file()]
    dump(OUT/'MANIFEST_SHA256.json',{str(p):sha(p) for p in sorted(files)})
    print('sealed',len(files),'files; no further writes permitted')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['fit','verify','seal'])
    args=parser.parse_args()
    {'fit':fit,'verify':verify,'seal':seal}[args.mode]()
