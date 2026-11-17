"""Parent-owned zero-step real eye-image probe. Default text-only, no heavy imports."""
from pathlib import Path
import argparse,json,hashlib,os,sys
ROOT=Path(__file__).resolve().parent

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent-resource-released',action='store_true');args=parser.parse_args()
    if not args.parent_resource_released:print('CODE_ONLY no image/model/physics execution');return
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError(name+'=1 required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    import numpy as np
    import mujoco as mj
    from PIL import Image,ImageDraw
    from camera import eye_frames,render_eye_pair,add_decor,PendingVision
    pr=json.loads((ROOT/'PREREG_CAMERA_PROBE.json').read_text());base=Path(pr['body_base']);run=Path(pr['trajectory'])
    sys.path.insert(0,str(base));from physics import Core
    from identity import verify,verify_runtime,capture
    ident=json.loads((run/'IDENTITY.json').read_text());verify(ident)
    files=[ROOT/'camera.py',ROOT/'probe.py',ROOT/'PREREG_CAMERA_PROBE.json',run/'IDENTITY.json',run/'trace_1ms.npz',Path(pr['head_geometry_audit'])]
    sourcehashes={str(p):sha(p) for p in files}
    out=ROOT/'CAMERA_PROBE';out.mkdir(exist_ok=False)
    calls={'integration_attempts':0}
    def forbidden(*a,**k):calls['integration_attempts']+=1;raise RuntimeError('zero-step optical probe forbids mj_step')
    mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
    c=Core(adhesion=False);model=c.plant;verify_runtime(ident,c)
    if (model.nq,model.nv,model.nu)!=(49,48,42):raise ValueError('body dimensions drift')
    mesh=model.geom_type==mj.mjtGeom.mjGEOM_MESH
    if mesh.sum()!=71 or not np.all(model.geom_rgba[mesh,3]==1.):raise ValueError('all71opaque required')
    probeidentity=capture(c,dict(kind=pr['kind'],source_hashes=sourcehashes,training=False,physics_steps=0),extra_paths=files)
    with (out/'IDENTITY.json').open('x') as f:json.dump(probeidentity,f,indent=2)
    with np.load(run/'trace_1ms.npz',allow_pickle=False) as z:
        qpos=z['qpos'][pr['poses_1ms_indices']].copy();qvel=z['qvel'][pr['poses_1ms_indices']].copy();times=z['time'][pr['poses_1ms_indices']].copy()
    audit=json.loads(Path(pr['head_geometry_audit']).read_text());head=model.body('Head').id
    def restore(j):
        c.d.qpos[:]=qpos[j];c.d.qvel[:]=qvel[j];c.d.time=float(times[j]);mj.mj_forward(model,c.d)
        if not np.array_equal(c.d.qpos,qpos[j]) or not np.array_equal(c.d.qvel,qvel[j]):raise ValueError('forward changed saved pose')
        return c.d.xpos[head].copy(),c.d.xmat[head].reshape(3,3).copy()
    hp,hr=restore(0);frames=eye_frames(hp,hr,audit,pr['eyes']);mid=(frames[0]['pos']+frames[1]['pos'])/2;ex=hr[:,0].copy();ex[2]=0.;ex/=np.linalg.norm(ex);ey=np.array([-ex[1],ex[0],0.]);basis=np.column_stack((ex,ey,np.array([0.,0.,1.])))
    targetspec=pr['visual_targets'];target_centers={side:mid+targetspec['forward_from_handover_proxy']*ex+sgn*targetspec['side_offset']*ey for side,sgn in [('left',1),('right',-1)]}
    for center in target_centers.values():center[2]=targetspec['center_z']
    wall=(mid+target_centers['left'])/2;wall[2]=1.4
    renderer=None;records=[];images=[]
    cases=[(name,0) for name in pr['cases_at_handover']]+[('left',1),('left',2)]
    try:
        renderer=mj.Renderer(model,height=32,width=32)
        for case,j in cases:
            hp,hr=restore(j);yaw=20. if case.endswith('plus20') else -20. if case.endswith('minus20') else 0.
            frames=eye_frames(hp,hr,audit,pr['eyes'],yaw)
            target='right' if case=='right' else 'left' if case.startswith('left') else None
            def decorate(scene):
                if target is not None:add_decor(scene,mj,mj.mjtGeom.mjGEOM_CYLINDER,[targetspec['radius'],targetspec['height']/2,0],target_centers[target],np.eye(3),targetspec['color_rgba'])
                if 'occluded' in case:add_decor(scene,mj,mj.mjtGeom.mjGEOM_BOX,[.04,.70,1.4],wall,basis,[.12,.12,.12,1.])
            rgb,meta=render_eye_pair(renderer,model,c.d,frames,pr['eyes'],decorate)
            filename=f'{len(images):02d}_{case}_pose{j}.png'
            image=np.concatenate((rgb[0],rgb[1]),axis=1);Image.fromarray(image).resize((512,256),Image.Resampling.NEAREST).save(out/filename)
            images.append(rgb);records.append(dict(case=case,pose_index=j,actual_saved_time_s=float(times[j]),camera_metadata=meta,image_file=str(out/filename),rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),optical_counterfactual_yaw_deg=yaw,actual_head_position=hp.tolist(),actual_head_rotation=hr.tolist()))
    finally:
        if renderer is not None:renderer.close()
    verify_runtime(ident,c);verify_runtime(probeidentity,c)
    for p,h in sourcehashes.items():
        if sha(p)!=h:raise ValueError('source/input changed '+p)
    clock=PendingVision();timeline=[]
    for k in range(17):
        t=k*.005
        if k%4==0:clock.capture(t,images[(k//4)%len(images)])
        observation=clock.query(t)
        valid=k>=4
        if observation['valid']!=valid:raise ValueError('20ms pending validity incorrect')
        if valid:
            expected=((k//4)-1)%len(images)
            if not np.array_equal(observation['image'],images[expected]) or observation['capture_time_s']>t-.020+1e-10:raise ValueError('futureimage leakage/heldframe mismatch')
        timeline.append({key:value for key,value in observation.items() if key!='image'}|dict(query_time_s=t))
    optical={}
    for case in ('left','right'):
        i=next(i for i,r in enumerate(records) if r['case']==case and r['pose_index']==0);optical[case+'_vs_off_meanabs']=float(np.abs(images[i].astype(float)-images[0]).mean())
    ia=next(i for i,r in enumerate(records) if r['case']=='left_occluded');ib=next(i for i,r in enumerate(records) if r['case']=='off_occluded');optical['occludedtarget_vs_occludedoff_meanabs']=float(np.abs(images[ia].astype(float)-images[ib]).mean())
    np.savez_compressed(out/'RAW_RGB.npz',rgb=np.stack(images))
    panel=Image.new('RGB',(512,288*len(images)),color='white');draw=ImageDraw.Draw(panel)
    for i,rec in enumerate(records):
        with Image.open(rec['image_file']) as im:panel.paste(im,(0,i*288+32))
        draw.text((4,i*288+5),f'{i}: {rec["case"]} savedpose{rec["pose_index"]} LEFT | RIGHT',fill='black')
    panel.save(out/'EYE_PANEL.png')
    report=dict(status='REAL_RGB_OPTICAL_PROBE_COMPLETE_REQUIRES_PARENT_IMAGE_REVIEW',Core_constructions=1,physical_steps=0,integration_attempts=calls['integration_attempts'],model_forward_calls=0,training_updates=0,mesh_count=71,all_mesh_opaque=True,source_hashes=sourcehashes,compiled_physics_before_after_unchanged=True,only_saved_pose_reconstruction=True,render_only_geometry_no_collision=True,diagnostics=optical,pending_20ms_release_5ms_hold_checked=True,pending_timeline=timeline,records=records,parameters=pr,biological_compound_eye_validity=False,behavior_vision_qualified=False)
    with (out/'PREFLIGHT.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(physical_steps=0,integration_attempts=0,poses=3,case_count=len(images),diagnostics=optical,panel=str(out/'EYE_PANEL.png'))))
if __name__=='__main__':main()
