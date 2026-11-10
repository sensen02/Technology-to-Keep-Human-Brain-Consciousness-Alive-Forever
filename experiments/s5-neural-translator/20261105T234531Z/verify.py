"""Independent causal-history reconstruction, amplitude, scoring and replay audit."""
from pathlib import Path
import pickle
import numpy as np
import timing as c
from translator import SpikeHistory,NumpyTranslator
O=c.O;dump=c.dump;js=c.js;sha=c.sha;arrays=c.arrays

def common_lags(limits):
    answer=[]
    for case in js(O/'TIMING_DIAGNOSIS.json')['cases']:
        p=Path(case['folder']);manifest=js(c.PREV/'MANIFEST_SHA256.json')
        for name in ['BODY.npz','NEURAL.npz','NATIVE.npz']:assert sha(p/name)==manifest[str(p/name)]
        d=arrays(O/'DATA'/('ALIGNMENT_'+str(case['phase'])+'.npz'));pred=(d['actual']-d['old'])/limits;target=(d['target']-d['old'])/limits;clean=d['strict_prefix'];n=len(clean)
        chosen=None
        for span in [20,10,5,2,1,0]:
            lags=[v for v in [-20,-10,-5,-2,-1,0,1,2,5,10,20] if abs(v)<=span];ids=np.arange(n);valid=clean.copy()
            for lag in lags:
                jj=ids+lag;valid&=(jj>=0)&(jj<n);valid&=clean[np.clip(jj,0,n-1)]
            ii=ids[valid]
            if len(ii)>=10:
                curve=[dict(lag_ms=lag,n=len(ii),MSE=float(np.mean((pred[ii]-target[ii+lag])**2))) for lag in lags];chosen=dict(curve=curve,best=min(curve,key=lambda row:row['MSE']),same_prediction_rows=ii);break
        answer.append(dict(phase=case['phase'],common_support=chosen))
    dump(O/'TIMING_COMMON_SUPPORT.json',dict(cases=answer,interpretation='All lag comparisons per case use the same prediction rows. Remaining association is not a causal delay estimate. Strict windows may be too short for large lags.'))

def main():
    t=c.runtime(True);base=t.cache;limits=t.limits;ident=js(O/'TRANSLATOR_IDENTITIES.json');cells=np.array(ident['cells']);matrix=np.array(ident['old_matrix']);pool=[np.searchsorted(cells,np.array(p)) for p in ident['pools']];warmsp=arrays(c.FIRST/'DATA/train_offset_0/TEACHER.npz')['spikes204'];sourceweights=arrays(c.MODEL)
    import train_dynamic_support as dst
    import dynamic_support_metrics as dsm
    histories={}
    # Recompute each required release history independently from archived spikes.
    for mode,taus in c.TAUS.items():
        decay=np.exp(-.0005/taus)[:,None];state=np.zeros((3,204));stops={round((phase-.2)/.0005):phase for phase in base}
        for step,sp in enumerate(warmsp,1):
            state=decay*state+(1-decay)*sp[None,:]/.0005
            if step in stops:histories[(mode,stops[step])]=state.copy()
    checked=[];models={}
    for folder in sorted((O/'RUNS').iterdir()):
        cfg=js(folder/'PREREGISTRATION.json');r=js(folder/'RESULT.json');path=cfg['model_file'];assert sha(path)==cfg['model_sha256'];weights=arrays(path)
        for key in ['W0','W1','W2']:assert np.array_equal(weights[key],sourceweights[key])
        gain=float(weights['residual_gain']);taus=weights['taus_s'];mode=next(k for k,v in c.TAUS.items() if np.array_equal(v,taus));state=base[cfg['release_time_s']];n=arrays(folder/'NEURAL.npz');body=arrays(folder/'BODY.npz');a=arrays(folder/'NATIVE.npz')['rows']
        if path not in models:models[path]=NumpyTranslator(path)
        model=models[path];rates=histories[(mode,cfg['release_time_s'])].copy();oldrates=state['history']['rates'][0].copy();decay=np.exp(-.0005/taus)[:,None];olddecay=np.exp(-.0005/.005);maxgainerr=0.
        for i,pair in enumerate(n['spikes204']):
            features=(rates/100).astype(np.float32).ravel();assert np.array_equal(features,n['features'][i]);effective=np.zeros_like(features) if cfg['translator_cut'] else features
            assert np.array_equal(effective,n['policy_features'][i]);raw=model(effective)
            expected=np.zeros(42,np.float32) if gain==0 else raw if gain==1 else np.arctanh(gain*np.tanh(raw));assert np.array_equal(expected,n['action'][i])
            maxgainerr=max(maxgainerr,float(np.max(abs(np.tanh(expected)-gain*np.tanh(raw)))))
            projected=np.array([oldrates[idx].mean() for idx in pool]);np.testing.assert_allclose(projected,n['source_rates'][i],atol=1e-10,rtol=1e-12)
            for spikes in pair:
                rates=decay*rates+(1-decay)*spikes[None,:]/.0005;oldrates=olddecay*oldrates+(1-olddecay)*spikes/.0005
        assert maxgainerr<1e-6
        old=np.array([np.clip(limits/100*(matrix@rr),-limits,limits) for rr in n['source_rates']]);res=2*limits*np.tanh(n['action']);ctrl=np.zeros_like(old) if cfg['cut'] else np.clip(old+res,-limits,limits)
        assert np.array_equal(old,n['old_torque']) and np.array_equal(res,n['residual']) and np.array_equal(ctrl,n['applied']) and np.array_equal(ctrl,body['ctrl'][1::2])
        if cfg['sensory_cut']:assert not n['sensory_current_norm'].any()
        if cfg['translator_cut'] or gain==0:assert not res.any()
        assert np.isfinite(a).all() and np.isfinite(body['qpos']).all();assert np.allclose(np.diff(a[:,0]),c.rt.DT,rtol=0,atol=1e-12);assert abs(a[-1,0]-a[0,0]-.2)<1e-9
        normalized=a.copy();normalized[:,0]-=cfg['release_time_s']-1.3;J,_=dst.objective(normalized,state['assist']);assert J==r['J'];g=dsm.evaluate_native(state['assist'],a);g['checks']['D1_no_nonfoot_ground_force']=bool(np.max(a[:,7])==0);assert g['checks']==r['validation']['checks']
        warm=arrays(Path(state['warm_folder'])/'BODY.npz');ix=int(np.argmin(abs(warm['time']-cfg['release_time_s'])));assert np.array_equal(warm['qpos'][ix],body['qpos'][0]) and np.array_equal(warm['qvel'][ix],body['qvel'][0])
        checked.append(dict(folder=str(folder),J=J,causal_history_verified=True,original_pool_rates_verified=True,gain_roundoff_max=maxgainerr,command_exact=True,body_continuity_exact=True))
    summary=js(O/'SUMMARY.json');assert len(checked)==summary['total']
    before=js(O/'INPUT_HASHES_BEFORE.json');assert before==js(O/'INPUT_HASHES_AFTER_prepare.json')==js(O/'INPUT_HASHES_AFTER_run.json');assert all(sha(p)==h for p,h in before.items())
    common_lags(limits);render(t,summary)
    dump(O/'VERIFICATION.json',dict(total=len(checked),checks=checked,all_weights_frozen=True,original_decoder_5ms_unchanged=True,history_rebuilt_causally_before_release=True,input_hashes_unchanged=True))
    print('VERIFIED',len(checked),'timing and amplitude trajectories',flush=True)

def render(t,s):
    import mujoco,imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from render_dynamic_support_video import load
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    zero=next(r for r in s['candidates'][0]['results'] if r['release_time_s']==1.3);parent=js(c.PREV/'SUMMARY.json')['representative'];safe=next(r for r in s['guarded']['results'] if r['release_time_s']==1.3)
    selected=[('OLD readout',zero),('PREVIOUS full correction',parent),('BEST nonzero: '+s['best_nonzero']['name'],s['representative']),('GUARDED: '+s['guarded']['name'],safe)]
    fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for label,r in selected:
        a=arrays(Path(r['folder'])/'NATIVE.npz')['rows'];x=a[:,0]-1.3;axs[0].plot(x,a[:,10],label=label);axs[1].plot(x,a[:,1:7].sum(1)/24.446354755878005);axs[2].plot(x,a[:,7]/24.446354755878005)
    axs[0].legend(fontsize=8);axs[0].set_ylabel('COM z');axs[1].set_ylabel('Foot load/W');axs[2].set_ylabel('Nonfoot/W');axs[2].set_xlabel('Time after release(s)')
    for ax in axs:ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(O/'COMPARISON.png',dpi=140);plt.close(fig)
    grid=np.empty((4,6));guards=np.zeros((4,6),bool)
    for candidate in s['candidates'][1:]:
        i=list(c.TAUS).index(candidate['tau_mode']);j=c.GAINS.index(candidate['gain']);grid[i,j]=max(r['J'] for r in candidate['results']);guards[i,j]=candidate['guard_pass']
    fig,ax=plt.subplots(figsize=(10,5));im=ax.imshow(grid,aspect='auto');ax.set_xticks(range(6),c.GAINS);ax.set_yticks(range(4),list(c.TAUS));ax.set_xlabel('Correction amplitude');ax.set_ylabel('Translator filter time factor');ax.set_title('Worst loss across five states; * passes1.300s regression guard')
    for i in range(4):
        for j in range(6):ax.text(j,i,f'{grid[i,j]:.3f}'+('*' if guards[i,j] else ''),ha='center',va='center',color='white')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(O/'TIMING_GAIN_GRID.png',dpi=140);plt.close(fig)
    model,_,inventory=load(c.rt.b.P/'MODEL_JOINT_TORQUE.xml');dump(O/'VISUAL_INVENTORY.json',inventory);renderer=mujoco.Renderer(model,height=480,width=640);d=mujoco.MjData(model);cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.distance=8.;cam.azimuth=125.;cam.elevation=-28.;opt=mujoco.MjvOption();mujoco.mjv_defaultOption(opt);opt.sitegroup[:]=0
    for name in ['mjVIS_TENDON','mjVIS_ACTUATOR','mjVIS_JOINT']:opt.flags[getattr(mujoco.mjtVisFlag,name)]=0
    warm=arrays(Path(t.cache[1.3]['warm_folder'])/'BODY.npz');mask=warm['time']<1.3-1e-9;series=[]
    for label,r in selected:
        b=arrays(Path(r['folder'])/'BODY.npz');series.append((label,r,{k:np.concatenate([warm[k][mask],b[k]]) for k in ['time','qpos','qvel','ctrl','com']}))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);frames=[];mapping=[];err=0.
    with imageio.get_writer(O/'TIMING_GAIN_GUARD_8x_slow.mp4',fps=25,codec='libx264',quality=7,macro_block_size=16) as writer:
        for fi in range(301):
            when=fi*.005;panels=[];indices=[]
            for label,r,b in series:
                ix=int(np.argmin(abs(b['time']-when)));indices.append(ix);d.qpos[:]=b['qpos'][ix];d.qvel[:]=b['qvel'][ix];d.ctrl[:]=b['ctrl'][ix];mujoco.mj_forward(model,d);err=max(err,float(np.max(abs(d.subtree_com[model.body('Thorax').id]-b['com'][ix]))));cam.lookat[:]=[d.qpos[0],d.qpos[1],.9];renderer.update_scene(d,camera=cam,scene_option=opt)
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image);draw.rectangle([0,0,640,100],fill=(20,20,20));draw.text((10,8),label,font=font,fill='white');draw.text((10,34),f't={when:.3f}s | 8x slow | '+('ALGORITHM WARMUP' if when<1.3 else 'NEURAL ONLY'),font=font,fill=(255,220,90));draw.text((10,62),'Walking / intention NOT ESTABLISHED',font=font,fill='white');panels.append(np.array(image))
            frame=np.concatenate(panels,axis=1);writer.append_data(frame);mapping.append(dict(frame=fi,time_s=when,rows=indices))
            if fi in [0,250,260,280,300]:frames.append(frame)
    renderer.close();assert err==0.;imageio.imwrite(O/'VIDEO_SAMPLES.png',np.concatenate(frames));reader=imageio.get_reader(O/'TIMING_GAIN_GUARD_8x_slow.mp4');assert reader.count_frames()==301
    for idx in [0,260,300]:assert reader.get_data(idx).shape==(480,2560,3)
    reader.close();dump(O/'VIDEO_FRAME_MAP.json',mapping);dump(O/'VIDEO_VERIFICATION.json',dict(frames=301,fps=25,slow_factor=8,replay_com_error=err,visible_meshes=inventory['visible'],physics_arrays_unchanged=inventory['physics_arrays_unchanged'],sources=[dict(label=label,folder=r['folder'],sha256=sha(Path(r['folder'])/'BODY.npz')) for label,r,_ in series],visual_inspection='NOT_COMPLETED'))
if __name__=='__main__':main()
