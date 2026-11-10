"""Continuous physical SNN runtime. Translator sees only completed spike histories.
Old Trial is reused solely for validated model identities, contacts and observation helpers.
No source rewriting, teacher evaluation, pose writes or external forces during release.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,hashlib,copy,time,pickle,resource
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path[:0]=[str(R),str(R/'tools')]
import importlib.util
LEGACY=R/'experiments/s5-neural-joint-torque/20261105T062800Z/run_trial.py'
spec=importlib.util.spec_from_file_location('legacy_physics',LEGACY);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
from translator import SpikeHistory
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261105T200021Z')
PRIOR=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T153657Z')
DOSE=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T182927Z')
TRAIN=[1.2,1.25,1.3,1.35,1.4];DEV=[1.225,1.275,1.325,1.375];TEST=[1.212,1.262,1.312,1.362]  # whole 1ms command boundaries; no half-window future spikes
DT=b.DT;NDT=b.NDT;CDT=b.CDT
js=b.js;dump=b.dump;sha=b.sha

def files():
    return b.sources()+[PRIOR/'SUMMARY.json',PRIOR/'COVERAGE.json',Path(js(PRIOR/'SUMMARY.json')['P0']['best']['matrix_file']),DOSE/'SUMMARY.json',Path(__file__),Path(__file__).with_name('translator.py'),Path(__file__).with_name('experiment.py')]

class Runtime(b.Trial):
    def __init__(self):
        b.O=O;super().__init__()
        self.cells=np.array(sorted(set(np.concatenate(self.pools))),int);assert len(self.cells)==204
        old=js(PRIOR/'SUMMARY.json')['P0']['best'];self.matrix=np.array(js(old['matrix_file'])['matrix'])
        self.history=SpikeHistory(self.cells);self.last_spikes=[];self.zero_sensory=False
        manifest=js(b.B/'RUN/MAPPING.json')['rows'];self.rh=np.array(sorted({i for row in manifest if row['leg']=='RH' for i in row['local_indices']}))
        self.cache={};dump(O/'TRANSLATOR_IDENTITIES.json',dict(cells=self.cells,RH_cells=self.rh,old_matrix=self.matrix,limits=self.limits,pools=[p.tolist() for p in self.pools],joint_names=[self.m.joint(int(j)).name for j in self.jids]))
    def observe_neural(self,d,net,enc,rates):
        from run_neural_muscle_closure import observe_quantities
        if self.zero_sensory:current=np.zeros(12000)
        else:current,_=enc.encode(observe_quantities(self.m,d,self.q0,self.joints,self.geom));current=np.asarray(current)
        decay=np.exp(-NDT/.005);counts=np.zeros(12000,dtype=np.int32);self.last_spikes=[]
        for _ in range(2):
            sp=net.step(current_nA=current);counts+=sp;self.last_spikes.append(sp[self.cells].copy());self.history.consume(sp)
            rates=decay*rates+(1-decay)*np.asarray(self.poolmat@sp,dtype=float)/NDT
        return rates,counts,current
    def old_command(self,rates):return np.clip(self.limits/100*(self.matrix@rates),-self.limits,self.limits)
    def warmup(self,key,phase_offset=0.,capture=()):
        from engine.neural_cond import ConductanceNetwork,ConductanceParams
        from engine.embodied.sensory_channels import ChannelSensoryEncoder,ChannelConfig
        folder=O/'DATA'/key;folder.mkdir(parents=True,exist_ok=False)
        dump(folder/'PREREGISTRATION.json',dict(phase_offset_rad=phase_offset,capture_times=list(capture),teacher_only=True,stop_s=1.4))
        np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=NDT*1000)
        net.set_connectivity(*[sparse.load_npz(b.AB/'A'/('ACTUAL_'+k+'.npz')) for k in ['We','Wi']]);enc=ChannelSensoryEncoder(self.groups,12000,ChannelConfig())
        m=self.m;mj=self.mj;d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);mj.mj_forward(m,d)
        self.history.reset();rates=np.zeros(66);self.zero_sensory=False
        body=[self.bodyrow(d)];native=[];xs=[];oldctrl=[];targets=[];sample_times=[];spikes=[];teacher_acc=np.zeros(42);cache_steps={round(t/DT):t for t in capture}
        kp=self.gains['kp'];kd=np.asarray(self.gains['kd']);clip=self.gains['clip']
        for step in range(round(1.4/DT)):
            if step%80==0:
                source=self.history.features();source_old=self.old_command(rates);teacher_acc[:]=0
                if step>=round(.2/DT):rates,counts,current=self.observe_neural(d,net,enc,rates);spikes.extend(self.last_spikes)
            t=float(d.time);ramp=np.clip((t-.15)/.10,0,1);phi=max(0,t-.15)*2*np.pi*5+phase_offset
            qg=self.gait(phi%(2*np.pi));des=(1-ramp)*self.q0[self.qa]+ramp*qg
            vel=ramp*self.gait(phi%(2*np.pi),1)*2*np.pi*5+(qg-self.q0[self.qa])*(10. if .15<t<.25 else 0.)
            torque=kp*(des-d.qpos[self.qa])+kd*(vel-d.qvel[self.va])+d.qfrc_bias[self.va]-d.qfrc_passive[self.va]
            d.ctrl[:]=np.clip(np.clip(torque,-clip,clip),-self.limits,self.limits);d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;teacher_acc+=d.ctrl
            mj.mj_step(m,d);mj.mj_forward(m,d)
            if step>=round(1./DT):native.append(self.contact(d))
            if (step+1)%40==0:body.append(self.bodyrow(d))
            if (step+1)%80==0 and step>=round(.2/DT):
                xs.append(source);oldctrl.append(source_old);targets.append(teacher_acc/80);sample_times.append(float(d.time)-CDT)
            assert np.isfinite(d.qpos).all() and not d.warning.number.any()
            if step+1 in cache_steps:
                duration=cache_steps[step+1];arr=np.asarray(native);win=arr[-round(.2/DT):]
                state=dict(integration=self.snapshot(d),net=copy.deepcopy(net),enc=copy.deepcopy(enc),rates=rates.copy(),history=self.history.state_dict(),duration=duration,phase_offset=phase_offset,assist=dict(foot_support_mean=float(win[:,1:7].sum(1).mean()/(m.body_mass.sum()*abs(m.opt.gravity[2]))),com_z_mean=float(win[:,10].mean())),warm_folder=str(folder))
                self.cache[duration]=state
        np.savez_compressed(folder/'BODY.npz',**{k:np.array([r[k] for r in body]) for k in body[0]})
        np.savez_compressed(folder/'NATIVE.npz',rows=np.asarray(native))
        np.savez_compressed(folder/'TEACHER.npz',features=xs,old_torque=oldctrl,target_torque=targets,time=sample_times,spikes204=np.array(spikes),cells=self.cells)
        print('WARM',key,'offset',phase_offset,'captures',len(capture),flush=True)
        return folder
    def prepare(self):
        self.warmup('train_offset_0',0.,TRAIN+DEV+TEST)
        for label,offset in [('train_offset_04',.4),('train_offset_08',.8),('train_offset_12',1.2),('dev_offset_02',.2),('dev_offset_06',.6)]:self.warmup(label,offset)
        with (O/'DATA/CACHE.pkl').open('wb') as f:pickle.dump(self.cache,f,protocol=5)
        dump(O/'DATA/CACHE_IDENTITY.json',dict(file=str(O/'DATA/CACHE.pkl'),sha256=sha(O/'DATA/CACHE.pkl'),owned_local_pickle=True))
    def load_cache(self):
        ident=js(O/'DATA/CACHE_IDENTITY.json');assert sha(ident['file'])==ident['sha256']
        with Path(ident['file']).open('rb') as f:self.cache=pickle.load(f)
    def episode(self,label,release_time,actor,model_file,duration=.2,stochastic=False,cut=False,sensory_cut=False,translator_cut=False):
        import train_dynamic_support as dst
        import dynamic_support_metrics as dsm
        folder=O/'RUNS'/(time.strftime('%Y%m%dT%H%M%S',time.gmtime())+str(time.time_ns()%1000000000).zfill(9)+'Z_'+label);folder.mkdir(parents=True)
        dump(folder/'PREREGISTRATION.json',dict(label=label,release_time_s=release_time,duration_s=duration,model_file=str(model_file),model_sha256=sha(model_file),stochastic=stochastic,cut=cut,sensory_cut=sensory_cut,translator_cut=translator_cut,RH_exc_gain=.375,importance='核心必需'))
        c=self.cache[release_time];net=copy.deepcopy(c['net']);enc=copy.deepcopy(c['enc']);rates=c['rates'].copy();self.history.load_state_dict(c['history']);self.zero_sensory=sensory_cut
        rowids=np.repeat(np.arange(net.n),np.diff(net.We.indptr));mask=np.isin(rowids,self.rh);net.We.data[mask]*=.375
        d=self.mj.MjData(self.m);self.mj.mj_setState(self.m,d,c['integration'],self.mj.mjtState.mjSTATE_INTEGRATION);self.mj.mj_forward(self.m,d)
        beforeq=d.qpos.copy();beforev=d.qvel.copy();d.ctrl[:]=0 if cut else self.old_command(rates);d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;self.mj.mj_forward(self.m,d)
        assert np.array_equal(beforeq,d.qpos) and np.array_equal(beforev,d.qvel)
        body=[self.bodyrow(d)];native=[self.contact(d)];log=[];start=time.monotonic();nsteps=round(duration/CDT)
        phi_prev=0.;training=[]
        def potential(row):
            foot=row[1:7].sum()/dsm.WEIGHT;nf=row[7]/dsm.WEIGHT
            return -(max(0.,1-foot)+max(0.,1-row[10]/c['assist']['com_z_mean'])+2*nf/max(foot+nf,1e-9))
        phi_prev=potential(native[0]);phi_initial=phi_prev
        try:
            for k in range(nsteps):
                features=self.history.features();policy_features=np.zeros_like(features) if translator_cut else features.copy();old=self.old_command(rates)
                action,logprob,value=actor(policy_features,stochastic)
                if translator_cut:assert np.max(abs(action))==0 or stochastic
                residual=2*self.limits*np.tanh(action);ctrl=np.zeros(42) if cut else np.clip(old+residual,-self.limits,self.limits)
                source_rates=rates.copy();rates,sp,cur=self.observe_neural(d,net,enc,rates)
                d.ctrl[:]=ctrl
                for j in range(80):
                    assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                    self.mj.mj_step(self.m,d);self.mj.mj_forward(self.m,d);native.append(self.contact(d))
                    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and not d.warning.number.any()
                    if (j+1)%40==0:body.append(self.bodyrow(d))
                phi=potential(native[-1]);reward=phi-phi_prev;phi_prev=phi
                training.append((policy_features,action,logprob,value,reward))
                log.append(dict(time=float(d.time),features=features,policy_features=policy_features,action=action,old_torque=old,residual=residual,applied=ctrl,source_rates=source_rates,rates=rates.copy(),spikes204=np.asarray(self.last_spikes),sensory_current_norm=float(np.linalg.norm(cur))))
            arr=np.asarray(native);scored=arr.copy();scored[:,0]-=release_time-1.3
            # Legacy time penalty uses a .2s denominator: set it explicitly to this run length.
            prior_release=dst.RELEASE;dst.RELEASE=duration
            try:J,parts=dst.objective(scored,c['assist'])
            finally:dst.RELEASE=prior_release
            gate=dsm.evaluate_native(c['assist'],arr);gate['checks']['D1_no_nonfoot_ground_force']=bool(np.max(arr[:,7])==0);gate['passed']=all(gate['checks'].values())
            last=training[-1];training[-1]=(*last[:4],last[4]-J-phi_prev)
            assert abs(sum(r[4] for r in training)-(-J-phi_initial))<1e-8
            contact=arr[:,1:7]>1e-9;liftoffs=np.sum(np.diff(contact.astype(int),axis=0)==-1,axis=0);landings=np.sum(np.diff(contact.astype(int),axis=0)==1,axis=0)
            result=dict(label=label,J=J,parts=parts,validation=gate,release_time_s=release_time,duration_s=duration,folder=str(folder),model_file=str(model_file),wall_s=time.monotonic()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,finite=True,warnings=int(d.warning.number.sum()),release_aux_max=0.,release_external_max=0.,displacement_xy=(d.qpos[:2]-beforeq[:2]),liftoffs=liftoffs,landings=landings,saturation_fraction=float(np.mean(np.abs([r['applied'] for r in log])>=self.limits[None,:])),walking_certified=False,intent_preserved='NOT_ESTABLISHED',stochastic=stochastic)
            dump(folder/'RESULT.json',result)
        except BaseException:
            import traceback
            dump(folder/'FAILURE.json',dict(traceback=traceback.format_exc()));raise
        finally:
            np.savez_compressed(folder/'NATIVE.npz',rows=np.asarray(native));np.savez_compressed(folder/'BODY.npz',**{k:np.array([r[k] for r in body]) for k in body[0]})
            if log:np.savez_compressed(folder/'NEURAL.npz',**{k:np.array([r[k] for r in log]) for k in log[0]})
            self.zero_sensory=False
        print(label,release_time,round(J,6),'pass',gate['passed'],flush=True)
        return result,training
