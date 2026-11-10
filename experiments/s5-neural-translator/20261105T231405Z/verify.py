"""Independent offline label, neural inference, physical score and video checks."""
from pathlib import Path
import pickle
import numpy as np
import continuation as c
from translator import SpikeHistory,NumpyTranslator
O=c.O;OLD=c.OLD;dump=c.dump;js=c.js;sha=c.sha

def arrays(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}

def verify_labels(t):
    checks=[];loaded={}
    for name in ['warm_train','warm_dev','closed_train','closed_dev','aggregate_1','aggregate_2']:
        ds=c.load(name);sources=np.unique(ds['source']);nchecked=0;maxerr=0.;eligible_after_earlier_contact=0
        for source in sources:
            ids=np.flatnonzero(ds['source']==source);pick=np.unique(ids[np.linspace(0,len(ids)-1,min(5,len(ids))).round().astype(int)])
            p=Path(source);folder=p.parent if p.name=='TEACHER.npz' else p
            if str(folder) not in loaded:loaded[str(folder)]=arrays(folder/'BODY.npz')
            b=loaded[str(folder)];offset=js(folder/'PREREGISTRATION.json').get('phase_offset_rad',0.)
            d=t.mj.MjData(t.m)
            for i in pick:
                ix=int(ds['body_index'][i]);time=float(b['time'][ix]);assert abs(ds['time'][i]-time)<1e-9
                d.qpos[:]=b['qpos'][ix];d.qvel[:]=b['qvel'][ix];d.ctrl[:]=b['ctrl'][ix];t.mj.mj_forward(t.m,d)
                phi=(max(0,time-.15)*10*np.pi+offset)%(2*np.pi);ramp=float(np.clip((time-.15)/.10,0,1));qref=t.gait(phi);goal=(1-ramp)*t.q0[t.qa]+ramp*qref
                velocity=ramp*t.gait(phi,1)*10*np.pi+(qref-t.q0[t.qa])*(10 if .15<time<.25 else 0)
                tau=t.gains['kp']*(goal-d.qpos[t.qa])+np.asarray(t.gains['kd'])*(velocity-d.qvel[t.va])+d.qfrc_bias[t.va]-d.qfrc_passive[t.va]
                expected=np.clip(np.clip(tau,-t.gains['clip'],t.gains['clip']),-t.limits,t.limits).astype(np.float32)
                error=float(np.max(abs((expected-ds['y'][i])/t.limits)));maxerr=max(maxerr,error);assert error<1e-6
                nchecked+=1
            if p.name!='TEACHER.npz':
                a=arrays(folder/'NATIVE.npz')['rows'];contact_ids=np.flatnonzero(a[:,7]>0);first_contact=int(contact_ids[0]) if len(contact_ids) else len(a)
                for i in ids:
                    k=int(ds['body_index'][i])//2;eligible=np.max(a[80*k:80*k+81,7])==0 and np.min(a[80*k:80*k+81,10])>=.9658022579833524
                    assert eligible==ds['eligible'][i]
                    eligible_after_earlier_contact+=int(eligible and 80*k>first_contact)
        checks.append(dict(dataset=name,rows=len(ds['x']),eligible=int(ds['eligible'].sum()),labels_recomputed=nchecked,normalized_max_error=maxerr,all_eligibility_masks_recomputed=True,eligible_after_earlier_nonfoot_contact=eligible_after_earlier_contact))
    # Brute-force float64 audit independent of optimized float32 pairwise formula.
    warm=c.load('warm_train');closed=c.load('closed_train');dist=arrays(O/'DATA/DISTANCES.npz');scale=dist['scale'].astype(np.float64);errors=[]
    for idx in np.linspace(0,len(closed['x'])-1,24).round().astype(int):
        rms=np.sqrt(np.mean(((warm['x'].astype(np.float64)-closed['x'][idx])/scale)**2,axis=1));errors.append(abs(float(rms.min())-float(dist['closed_distance'][idx])))
    assert max(errors)<.002
    split=js(O/'DATA/SPLIT.json');assert not(set(split['train_sources'])&set(split['dev_sources']))
    dump(O/'LABEL_VERIFICATION.json',dict(checks=checks,nearest_distance_float64_max_error=max(errors),no_shared_closed_train_dev_source=True,teacher_is_offline_reference_not_guaranteed_action=True))

def main():
    t=c.runtime(fresh=True);verify_labels(t);identity=js(O/'TRANSLATOR_IDENTITIES.json');cells=np.array(identity['cells']);limits=np.array(identity['limits']);matrix=np.array(identity['old_matrix']);pool_local=[np.searchsorted(cells,np.array(p)) for p in identity['pools']]
    import train_dynamic_support as dst
    import dynamic_support_metrics as dsm
    checks=[]
    for folder in sorted((O/'RUNS').iterdir()):
        cfg=js(folder/'PREREGISTRATION.json');r=js(folder/'RESULT.json');assert sha(cfg['model_file'])==cfg['model_sha256'];model=NumpyTranslator(cfg['model_file']);n=arrays(folder/'NEURAL.npz');body=arrays(folder/'BODY.npz');a=arrays(folder/'NATIVE.npz')['rows'];state=t.cache[cfg['release_time_s']]
        assert not cfg['stochastic'] and np.isfinite(a).all()
        hist=SpikeHistory(cells);hist.load_state_dict(state['history'])
        for i,pair in enumerate(n['spikes204']):
            feature=hist.features();assert np.array_equal(feature,n['features'][i]);policy=np.zeros_like(feature) if cfg['translator_cut'] else feature
            assert np.array_equal(policy,n['policy_features'][i]);assert np.array_equal(model(policy),n['action'][i])
            # Independent original-pool rates from individual5ms filters, allowing roundoff.
            rates=np.array([hist.rates[0,idx].mean() for idx in pool_local]);np.testing.assert_allclose(rates,n['source_rates'][i],rtol=1e-12,atol=1e-10)
            for sp in pair:
                full=np.zeros(12000,bool);full[cells]=sp;hist.consume(full)
        old=np.array([np.clip(limits/100*(matrix@rr),-limits,limits) for rr in n['source_rates']]);assert np.array_equal(old,n['old_torque'])
        residual=2*limits*np.tanh(n['action']);assert np.array_equal(residual,n['residual']);tau=np.zeros_like(old) if cfg['cut'] else np.clip(old+residual,-limits,limits);assert np.array_equal(tau,n['applied']) and np.array_equal(tau,body['ctrl'][1::2])
        if cfg['translator_cut']:assert not residual.any()
        if cfg['sensory_cut']:assert not n['sensory_current_norm'].any()
        assert np.allclose(np.diff(a[:,0]),c.rt.DT,atol=1e-12,rtol=0);assert abs(a[-1,0]-a[0,0]-.2)<1e-9
        shifted=a.copy();shifted[:,0]-=cfg['release_time_s']-1.3;J,parts=dst.objective(shifted,state['assist']);assert J==r['J']
        gate=dsm.evaluate_native(state['assist'],a);gate['checks']['D1_no_nonfoot_ground_force']=bool(np.max(a[:,7])==0);assert gate['checks']==r['validation']['checks']
        warm=arrays(Path(state['warm_folder'])/'BODY.npz');idx=int(np.argmin(abs(warm['time']-cfg['release_time_s'])));assert np.array_equal(body['qpos'][0],warm['qpos'][idx]) and np.array_equal(body['qvel'][0],warm['qvel'][idx])
        checks.append(dict(folder=str(folder),J=J,neural_filter_and_pool_rates_verified=True,commands_exact=True,continuity_exact=True))
    before=js(O/'INPUT_HASHES_BEFORE.json');assert all(sha(p)==h for p,h in before.items())
    assert before==js(O/'INPUT_HASHES_AFTER_prepare.json')==js(O/'INPUT_HASHES_AFTER_run.json')
    import sys
    reused='--reuse-video' in sys.argv
    if reused:
        previous_manifest=js(O/'MANIFEST_SHA256.json');vv=js(O/'VIDEO_VERIFICATION.json')
        for name in ['OLD_PREVIOUS_NEW_CUT_8x_slow.mp4','VIDEO_SAMPLES.png','VIDEO_FRAME_MAP.json','VIDEO_VERIFICATION.json','VISUAL_INVENTORY.json','COMPARISON.png','INPUT_SHIFT.png']:assert sha(O/name)==previous_manifest[str(O/name)]
        for source in vv['sources']:assert sha(Path(source['folder'])/'BODY.npz')==source['body_sha256']
    else:render(t)
    dump(O/'VERIFICATION.json',dict(total=len(checks),checks=checks,inputs_unchanged=True,labels_independently_checked=True,all_policy_inputs_neural_only=True,body_compensation_added=False,existing_video_reused_after_hash_and_source_checks=reused))
    print('VERIFIED',len(checks),'trajectories',flush=True)

def render(t):
    import mujoco,imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from render_dynamic_support_video import load
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=js(O/'SUMMARY.json');previous=js(OLD/'FINAL_CONTROLS.json')['representative'];selected=[('OLD readout',s['zero_reproduction']),('PREVIOUS trained MLP',previous),('NEW corrective-data MLP',s['representative']),('ALL OUTPUT CUT',s['motor_cut'])]
    fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for label,r in selected:
        a=arrays(Path(r['folder'])/'NATIVE.npz')['rows'];x=a[:,0]-1.3;axs[0].plot(x,a[:,10],label=label);axs[1].plot(x,a[:,1:7].sum(1)/24.446354755878005);axs[2].plot(x,a[:,7]/24.446354755878005)
    axs[0].legend();axs[0].set_ylabel('COM z');axs[1].set_ylabel('Foot load / W');axs[2].set_ylabel('Nonfoot / W');axs[2].set_xlabel('Time after release (s)')
    for ax in axs:ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(O/'COMPARISON.png',dpi=140);plt.close(fig)
    ds=arrays(O/'DATA/DISTANCES.npz');data=c.load('closed_train');fig,ax=plt.subplots(figsize=(9,5));ax.hist(ds['warmdev_distance'],bins=40,density=True,alpha=.6,label='Warm validation');ax.hist(ds['closed_distance'][data['eligible']],bins=40,density=True,alpha=.6,label='Closed loop before failure');ax.hist(ds['closed_distance'][~data['eligible']],bins=40,density=True,alpha=.4,label='Excluded failure states');ax.set_xlabel('Nearest warm-training feature RMS distance');ax.legend();fig.tight_layout();fig.savefig(O/'INPUT_SHIFT.png',dpi=140);plt.close(fig)
    model,_,inventory=load(c.rt.b.P/'MODEL_JOINT_TORQUE.xml');dump(O/'VISUAL_INVENTORY.json',inventory);renderer=mujoco.Renderer(model,height=480,width=640);d=mujoco.MjData(model);camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera);camera.distance=8.;camera.azimuth=125.;camera.elevation=-28.
    opt=mujoco.MjvOption();mujoco.mjv_defaultOption(opt);opt.sitegroup[:]=0
    for name in ['mjVIS_TENDON','mjVIS_ACTUATOR','mjVIS_JOINT']:opt.flags[getattr(mujoco.mjtVisFlag,name)]=0
    warm=arrays(Path(t.cache[1.3]['warm_folder'])/'BODY.npz');mask=warm['time']<1.3-1e-9;series=[]
    for label,r in selected:
        b=arrays(Path(r['folder'])/'BODY.npz');full={k:np.concatenate([warm[k][mask],b[k]]) for k in ['time','qpos','qvel','ctrl','com']};series.append((label,r,full))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);frames=[];mapping=[];error=0.
    with imageio.get_writer(O/'OLD_PREVIOUS_NEW_CUT_8x_slow.mp4',fps=25,codec='libx264',quality=7,macro_block_size=16) as writer:
        for fi in range(301):
            time=fi*.005;panels=[];indices=[]
            for label,r,b in series:
                i=int(np.argmin(abs(b['time']-time)));indices.append(i);d.qpos[:]=b['qpos'][i];d.qvel[:]=b['qvel'][i];d.ctrl[:]=b['ctrl'][i];mujoco.mj_forward(model,d);error=max(error,float(np.max(abs(d.subtree_com[model.body('Thorax').id]-b['com'][i]))));camera.lookat[:]=[d.qpos[0],d.qpos[1],.9];renderer.update_scene(d,camera=camera,scene_option=opt)
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image);draw.rectangle([0,0,640,100],fill=(20,20,20));phase='ALGORITHM WARMUP' if time<1.3 else ('ALL OUTPUT CUT' if 'CUT' in label else 'NEURAL ONLY');draw.text((10,8),label,font=font,fill='white');draw.text((10,34),f't={time:.3f}s | 8x slow | {phase}',font=font,fill=(255,220,90));draw.text((10,62),'Walking / intent preservation NOT ESTABLISHED',font=font,fill='white');panels.append(np.array(image))
            frame=np.concatenate(panels,axis=1);writer.append_data(frame);mapping.append(dict(frame=fi,time_s=time,source_rows=indices))
            if fi in [0,250,260,280,300]:frames.append(frame)
    renderer.close();assert error==0.;imageio.imwrite(O/'VIDEO_SAMPLES.png',np.concatenate(frames));reader=imageio.get_reader(O/'OLD_PREVIOUS_NEW_CUT_8x_slow.mp4');assert reader.count_frames()==301
    for i in [0,260,300]:assert reader.get_data(i).shape==(480,2560,3)
    reader.close();dump(O/'VIDEO_FRAME_MAP.json',mapping);dump(O/'VIDEO_VERIFICATION.json',dict(frames=301,fps=25,slow_factor=8,replay_com_error=error,mesh_visible=inventory['visible'],physics_arrays_unchanged=inventory['physics_arrays_unchanged'],sources=[dict(label=l,folder=r['folder'],body_sha256=sha(Path(r['folder'])/'BODY.npz')) for l,r,_ in series],visual_inspection='NOT_COMPLETED'))
if __name__=='__main__':main()
