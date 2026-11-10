"""Independent inference, filter, timing, score and replay verification."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np
import pickle,hashlib,json
import runtime as rt
from translator import SpikeHistory,NumpyTranslator
O=rt.O;js=rt.js;dump=rt.dump;sha=rt.sha

def main():
    summary=js(O/'SUMMARY.json');identity=js(O/'TRANSLATOR_IDENTITIES.json');cells=np.array(identity['cells']);limits=np.array(identity['limits']);matrix=np.array(identity['old_matrix'])
    cid=js(O/'DATA/CACHE_IDENTITY.json');assert sha(cid['file'])==cid['sha256']
    with Path(cid['file']).open('rb') as f:cache=pickle.load(f)
    assert all(int(c['net'].spike_count.sum())>0 and np.any(c['history']['rates']) for c in cache.values()), 'sensory-warmed neural cache required for every handover'
    import train_dynamic_support as dst
    import dynamic_support_metrics as dsm
    checked=[];models={}
    for folder in sorted((O/'RUNS').iterdir()):
        if not(folder/'RESULT.json').exists():continue
        cfg=js(folder/'PREREGISTRATION.json');result=js(folder/'RESULT.json');assert sha(cfg['model_file'])==cfg['model_sha256']
        if cfg['model_file'] not in models:models[cfg['model_file']]=NumpyTranslator(cfg['model_file'])
        model=models[cfg['model_file']];n=np.load(folder/'NEURAL.npz');body=np.load(folder/'BODY.npz');native=np.load(folder/'NATIVE.npz')['rows'];c=cache[cfg['release_time_s']]
        assert np.isfinite(native).all() and np.isfinite(body['qpos']).all()
        assert abs(native[-1,0]-native[0,0]-cfg['duration_s'])<1e-9
        assert np.allclose(np.diff(native[:,0]),rt.DT,atol=1e-12,rtol=0)
        old=np.array([np.clip(limits/100*(matrix@r),-limits,limits) for r in n['source_rates']]);assert np.array_equal(old,n['old_torque'])
        residual=2*limits[None,:]*np.tanh(n['action']);assert np.array_equal(residual,n['residual'])
        command=np.zeros_like(old) if cfg['cut'] else np.clip(old+residual,-limits,limits)
        assert np.array_equal(command,n['applied']) and np.array_equal(command,body['ctrl'][1::2])
        hist=SpikeHistory(cells);hist.load_state_dict(c['history']);maxerr=0.
        for i,pair in enumerate(n['spikes204']):
            assert np.array_equal(hist.features(),n['features'][i])
            expected_features=np.zeros(612,np.float32) if cfg['translator_cut'] else hist.features()
            assert np.array_equal(expected_features,n['policy_features'][i])
            if not cfg['stochastic']:
                err=float(np.max(abs(model(expected_features)-n['action'][i])));maxerr=max(maxerr,err);assert err<1e-6
            for sample in pair:
                full=np.zeros(12000,bool);full[cells]=sample;hist.consume(full)
        if cfg['sensory_cut']:assert not np.any(n['sensory_current_norm'])
        if cfg['translator_cut']:assert not np.any(n['residual'])
        score=native.copy();score[:,0]-=cfg['release_time_s']-1.3;previous=dst.RELEASE;dst.RELEASE=cfg['duration_s']
        try:J,parts=dst.objective(score,c['assist'])
        finally:dst.RELEASE=previous
        assert J==result['J'];gate=dsm.evaluate_native(c['assist'],native);gate['checks']['D1_no_nonfoot_ground_force']=bool(np.max(native[:,7])==0)
        assert gate['checks']==result['validation']['checks']
        # At every actual release time the warmup ends at an exact saved row.
        warm=np.load(Path(c['warm_folder'])/'BODY.npz');ix=int(np.argmin(abs(warm['time']-cfg['release_time_s'])))
        assert abs(warm['time'][ix]-cfg['release_time_s'])<1e-9
        assert np.array_equal(body['qpos'][0],warm['qpos'][ix]) and np.array_equal(body['qvel'][0],warm['qvel'][ix])
        checked.append(dict(folder=str(folder),J_recomputed=J,deterministic=not cfg['stochastic'],action_recompute_max_error=maxerr,filter_exact=True,state_continuity_exact=True))
    before=js(O/'INPUT_HASHES_BEFORE.json');assert all(sha(p)==h for p,h in before.items())
    for mode in ['prepare','fit','train']:assert js(O/('INPUT_HASHES_AFTER_'+mode+'.json'))==before
    # Saved-data diagnostic only: shuffle complete spike intervals and recompute causal filters.
    diagnostic=[]
    for label,r in [('selected',summary['representative']),('trained_MLP',js(O/'FINAL_CONTROLS.json')['representative'])]:
        folder=Path(r['folder']);cfg=js(folder/'PREREGISTRATION.json');n=np.load(folder/'NEURAL.npz');model=NumpyTranslator(cfg['model_file']);hist=SpikeHistory(cells);hist.load_state_dict(cache[1.3]['history']);rng=np.random.default_rng(20261105);order=rng.permutation(len(n['spikes204']));altered=[]
        for idx in order:
            altered.append(model(hist.features()))
            for sample in n['spikes204'][idx]:
                full=np.zeros(12000,bool);full[cells]=sample;hist.consume(full)
        diagnostic.append(dict(label=label,mode='offline shuffled saved spike intervals, not autonomous simulation or intent proof',seed=20261105,order=order,mean_squared_latent_change=float(np.mean((np.array(altered)-n['action'])**2))))
    dump(O/'HISTORY_DIAGNOSTIC.json',diagnostic)
    render(summary)
    dump(O/'VERIFICATION.json',dict(total=len(checked),checks=checked,input_hashes_unchanged=True,spike_history_recomputed=True,all_scores_recomputed=True,deterministic_inference_checked=True,stochastic_training='Recorded sampled latent action verified through torque mapping; action is exploration, not deterministic prediction.',intent_preservation='NOT_ESTABLISHED'))
    print('VERIFIED',len(checked),'live trajectories',flush=True)

def render(summary):
    import mujoco,imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from render_dynamic_support_video import load
    eng=js(O/'ENGINEERING.json');zero=js(Path(eng['reproduced_folder'])/'RESULT.json')
    controls=js(O/'FINAL_CONTROLS.json')
    selected=[('OLD neural readout',zero),('TRAINED MLP (development choice)',controls['representative']),('DEPLOYMENT choice: '+summary['incumbent']['name'],summary['representative']),('ALL neural output CUT',summary['motor_cut'])]
    fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for label,r in selected:
        a=np.load(Path(r['folder'])/'NATIVE.npz')['rows'];x=a[:,0]-r['release_time_s'];axs[0].plot(x,a[:,10],label=label);axs[1].plot(x,a[:,1:7].sum(1)/24.446354755878005,label=label);axs[2].plot(x,a[:,7]/24.446354755878005,label=label)
    for ax,title in zip(axs,['Thorax COM z','Foot load / weight','Positive nonfoot / weight']):ax.set_ylabel(title);ax.legend();ax.grid(alpha=.3)
    axs[-1].set_xlabel('Time after release (s)');fig.suptitle('Neural translator: support and motion intent assessed separately');fig.tight_layout();fig.savefig(O/'COMPARISON.png',dpi=140);plt.close(fig)
    model,_,inventory=load(rt.b.P/'MODEL_JOINT_TORQUE.xml');dump(O/'VISUAL_INVENTORY.json',inventory)
    renderer=mujoco.Renderer(model,height=480,width=640);data=mujoco.MjData(model);camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera);camera.distance=8.;camera.azimuth=125.;camera.elevation=-28.
    opt=mujoco.MjvOption();mujoco.mjv_defaultOption(opt);opt.sitegroup[:]=0
    for name in ['mjVIS_TENDON','mjVIS_ACTUATOR','mjVIS_JOINT']:opt.flags[getattr(mujoco.mjtVisFlag,name)]=0
    series=[];warm=np.load(O/'DATA/train_offset_0/BODY.npz');mask=warm['time']<1.3-1e-9
    for label,r in selected:
        b=np.load(Path(r['folder'])/'BODY.npz');full={k:np.concatenate([warm[k][mask],b[k]]) for k in ['time','qpos','qvel','ctrl','com']};series.append((label,r,full))
    end=min(b['time'][-1] for _,_,b in series);times=np.arange(round(end/.005)+1)*.005;frames=[];mapping=[];maxerr=0.;font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    with imageio.get_writer(O/'OLD_NEW_CUT_8x_slow.mp4',fps=25,codec='libx264',quality=7,macro_block_size=16) as writer:
        for fi,t in enumerate(times):
            panels=[];ids=[]
            for label,r,b in series:
                ix=int(np.argmin(abs(b['time']-t)));ids.append(ix);data.qpos[:]=b['qpos'][ix];data.qvel[:]=b['qvel'][ix];data.ctrl[:]=b['ctrl'][ix];mujoco.mj_forward(model,data)
                maxerr=max(maxerr,float(np.max(abs(data.subtree_com[model.body('Thorax').id]-b['com'][ix]))));camera.lookat[:]=[data.qpos[0],data.qpos[1],.9];renderer.update_scene(data,camera=camera,scene_option=opt)
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image);draw.rectangle([0,0,640,100],fill=(20,20,20));phase='ALGORITHM WARMUP' if t<1.3 else ('ALL OUTPUT CUT' if 'CUT' in label else 'NEURAL ONLY')
                draw.text((10,8),label,font=font,fill='white');draw.text((10,34),f't={t:.3f}s | 8x slow | {phase}',font=font,fill=(255,220,90));draw.text((10,62),'Walking and intent preservation NOT ESTABLISHED',font=font,fill='white');panels.append(np.asarray(image))
            frame=np.concatenate(panels,axis=1);writer.append_data(frame);mapping.append(dict(frame=fi,time_s=float(t),source_rows=ids))
            if fi in [0,250,260,280,len(times)-1]:frames.append(frame)
    renderer.close();assert maxerr<1e-10;imageio.imwrite(O/'VIDEO_SAMPLES.png',np.concatenate(frames))
    reader=imageio.get_reader(O/'OLD_NEW_CUT_8x_slow.mp4');assert reader.count_frames()==len(times)
    for i in [0,260,len(times)-1]:assert reader.get_data(i).shape==(480,2560,3)
    reader.close();dump(O/'VIDEO_FRAME_MAP.json',mapping);dump(O/'VIDEO_VERIFICATION.json',dict(frames=len(times),fps=25,slow_factor=8,replay_com_error=maxerr,mesh_visible=inventory['visible'],physics_arrays_unchanged=inventory['physics_arrays_unchanged'],no_visual_inspection_claim=True,trajectory_sources=[dict(label=label,folder=r['folder'],body_sha256=sha(Path(r['folder'])/'BODY.npz')) for label,r,_ in series]))
if __name__=='__main__':main()
