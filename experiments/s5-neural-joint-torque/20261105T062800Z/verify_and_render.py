"""Independent saved-array checks and pure trajectory replay for the S5 pilot."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,hashlib
from pathlib import Path
import numpy as np
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path[:0]=[str(R),str(R/'tools')]
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T062800Z')
P=Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z')
def js(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import mujoco,imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from render_dynamic_support_video import load
    summary=js(O/'SUMMARY.json');cov=js(O/'COVERAGE.json');S=np.array(cov['sign_matrix']);limit=np.array([r['limit'] for r in cov['joints']])
    checked=[]
    for f in sorted(O.glob('*/RESULT.json')):
        folder=f.parent;r=js(f);cfg=js(folder/'PARAMETERS.json');z=np.array(cfg['z']);b=np.load(folder/'BODY.npz');n=np.load(folder/'NEURAL.npz');native=np.load(folder/'NATIVE.npz')['rows']
        proposals=np.array([np.clip(limit/100*np.exp(z[66:])*(S@(np.exp(z[:66])*rr)),-limit,limit) for rr in n['source_rates']])
        if cfg['cut']:proposals[:]=0
        assert np.allclose(proposals,n['applied'],rtol=0,atol=1e-13)
        assert np.array_equal(n['source_rates'][1:],n['rates'][:-1])
        assert np.isfinite(b['qpos']).all() and np.isfinite(b['qvel']).all()
        assert native.shape==(16001,20) and np.allclose(np.diff(native[:,0]),.0000125,rtol=0,atol=1e-12)
        assert n['rates'].shape==(200,66) and n['currents'].shape==(200,12000)
        assert np.allclose(b['ctrl'][1::2],n['applied'],rtol=0,atol=0)
        warm=np.load(O/('WARM_'+cfg['arm'])/'BODY.npz')
        assert np.array_equal(b['qpos'][0],warm['qpos'][-1]) and np.array_equal(b['qvel'][0],warm['qvel'][-1])
        checked.append(dict(folder=str(folder),readout_error=float(np.max(abs(proposals-n['applied']))),sensory_varies=bool(np.any(np.ptp(n['currents'],axis=0)>0)),pool_active_count=int((np.max(n['rates'],axis=0)>1e-9).sum()),zero_output_joints=int((np.max(abs(n['applied']),axis=0)==0).sum())))
    assert len(checked)==38
    assert js(O/'INPUT_HASHES_BEFORE.json')==js(O/'INPUT_HASHES_AFTER.json')
    for path,h in js(O/'INPUT_HASHES_BEFORE.json').items():assert sha(path)==h
    selected=[('P0 original',summary['P0']['best'],'P0'),('P1 readout cut',summary['P0']['cut'],'P0'),('P2 random',summary['P2']['best'],'P2')]
    fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for label,r,arm in selected:
        a=np.load(Path(r['folder'])/'NATIVE.npz')['rows'];t=a[:,0]-1.3
        axs[0].plot(t,a[:,10],label=label);axs[1].plot(t,a[:,1:7].sum(1)/24.446354755878005,label=label);axs[2].plot(t,a[:,7]/24.446354755878005,label=label)
    axs[0].axhline(.8*1.2072528224791905,color='k',ls='--',label='D8 floor');axs[0].set_ylabel('Thorax subtree COM z')
    axs[1].set_ylabel('Foot load / weight');axs[2].set_ylabel('Positive nonfoot / weight');axs[2].set_xlabel('Time after release (s)')
    for ax in axs:ax.legend();ax.grid(alpha=.3)
    fig.suptitle('Joint-torque SNN pilot: 1.3 s algorithm warmup, 0.2 s release\nNo walking certification; absolute force/SI NOT_ESTABLISHED')
    fig.tight_layout();fig.savefig(O/'COMPARISON.png',dpi=140);plt.close(fig)
    model,_,inventory=load(P/'MODEL_JOINT_TORQUE.xml');dump(O/'VISUAL_INVENTORY.json',inventory)
    renderer=mujoco.Renderer(model,height=480,width=640);data=mujoco.MjData(model)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera);camera.distance=8.;camera.azimuth=125.;camera.elevation=-28.
    opt=mujoco.MjvOption();mujoco.mjv_defaultOption(opt);opt.sitegroup[:]=0
    for flag in ('mjVIS_TENDON','mjVIS_ACTUATOR','mjVIS_JOINT'):opt.flags[getattr(mujoco.mjtVisFlag,flag)]=0
    series=[]
    for label,r,arm in selected:
        w=np.load(O/('WARM_'+arm)/'BODY.npz');b=np.load(Path(r['folder'])/'BODY.npz')
        full={k:np.concatenate([w[k][:-1],b[k]],axis=0) for k in ['time','qpos','qvel','ctrl','com']}
        series.append((label,full))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    # 5ms simulation per frame at 25fps = 8x slow. Entire warmup included.
    times=np.arange(301)*.005;maxerr=0.;samples=[];frame_map=[]
    with imageio.get_writer(O/'P0_P1_P2_8x_slow.mp4',fps=25,codec='libx264',quality=7,macro_block_size=16) as writer:
        for fi,t in enumerate(times):
            panels=[];frame_indices=[]
            for label,b in series:
                ix=int(np.argmin(abs(b['time']-t)));frame_indices.append(ix)
                data.qpos[:]=b['qpos'][ix];data.qvel[:]=b['qvel'][ix];data.ctrl[:]=b['ctrl'][ix];mujoco.mj_forward(model,data)
                maxerr=max(maxerr,float(np.max(abs(data.subtree_com[model.body('Thorax').id]-b['com'][ix]))))
                camera.lookat[:]=[data.qpos[0],data.qpos[1],.9]
                renderer.update_scene(data,camera=camera,scene_option=opt);image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                draw.rectangle([0,0,640,102],fill=(20,20,20));phase='ALGORITHM WARMUP' if t<1.3 else ('READOUT CUT' if label.startswith('P1') else 'NEURAL TORQUE ONLY')
                draw.text((10,8),label,font=font,fill='white');draw.text((10,34),f't={t:.3f}s | 8x slow | {phase}',font=font,fill=(255,220,90));draw.text((10,62),'Pilot failed support gate; not autonomous walking',font=font,fill='white')
                panels.append(np.asarray(image))
            combined=np.concatenate(panels,axis=1);writer.append_data(combined);frame_map.append(dict(frame=fi,time_s=float(t),source_rows=frame_indices))
            if fi in [0,250,260,270,300]:samples.append(combined)
    renderer.close();assert maxerr<1e-10
    imageio.imwrite(O/'VIDEO_SAMPLES.png',np.concatenate(samples,axis=0))
    dump(O/'VIDEO_FRAME_MAP.json',frame_map)
    reader=imageio.get_reader(O/'P0_P1_P2_8x_slow.mp4');assert reader.count_frames()==301
    for i in [0,260,300]:assert reader.get_data(i).shape==(480,1920,3)
    reader.close()
    dump(O/'VERIFICATION.json',dict(evaluations_checked=checked,total=38,input_hashes_unchanged=True,video_frames=301,video_fps=25,slow_factor=8,simulated_duration_s=1.5,replay_com_max_error=maxerr,visible_meshes=inventory['visible'],physics_arrays_unchanged=inventory['physics_arrays_unchanged'],model_sha256=sha(P/'MODEL_JOINT_TORQUE.xml'),trajectory_sources=[dict(label=label,body=str(Path(r['folder'])/'BODY.npz'),body_sha256=sha(Path(r['folder'])/'BODY.npz'),warm=str(O/('WARM_'+arm)/'BODY.npz'),warm_sha256=sha(O/('WARM_'+arm)/'BODY.npz')) for label,r,arm in selected]))
    print('VERIFIED',len(checked),'trajectories; replay error',maxerr,flush=True)
if __name__=='__main__':main()
