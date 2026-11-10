"""Preregistered CPU translator pretraining, live PPO, and held-out evaluation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import sys,time,copy,json,subprocess,platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from translator import Translator,NumpyTranslator,SpikeHistory
import runtime as rt
O=rt.O;dump=rt.dump;js=rt.js;sha=rt.sha

def prereg():
    O.mkdir(parents=True,exist_ok=False)
    dump(O/'PREREGISTRATION.json',dict(stage='S5 learned neural-only translator; support is a prerequisite, not intent preservation or walking',
        architecture='204 mapped motor cells, causal 5/20/50ms filters; 612->64->64->42 tanh MLP, no biases. 2L*tanh residual added to frozen old torque and clipped to unchanged L; zero last layer exact baseline.',
        frozen='12000-cell archived A network, encoder, old174 coefficients, RH incoming excitation0.375 applied at release; no body state/target/time/phase input to actor or critic; no online teacher',
        teacher_data=dict(train_offsets_rad=[0.,.4,.8,1.2],dev_offsets_rad=[.2,.6],window_s=[.3,1.4],labels='mean actual torque over next1ms, input prior completed neural window; whole trajectory groups; offset-specific full physical PD warmup only'),
        pretrain=dict(seed=20261105,adam_lr=.001,weight_decay=.0001,max_epochs=100,patience=12,batch_size=256,loss='mean squared normalized torque error after old+bounded residual',comparison='unrestricted bias-free linear residual vs MLP vs zero residual; select deployment by live performance, not teacher loss'),
        PPO=dict(seed=20261105,actor_lr=.0001,critic_lr=.0003,gamma=1.,gae_lambda=.95,clip=.2,epochs=5,minibatch=256,episodes_per_update=10,entropy_coefficient=.0001,initial_log_std=-3.,log_std_bounds=[-5.,-1.5],gradient_norm_max=.5,stage_training_counts=[160,64,32],stage_release_s=[.2,.5,1.],validation_cap=64),
        reward='Terminal -existing J, dense potential differences Phi(next)-Phi(now), terminal Phi=0, gamma1. Phi=-(max(0,1-foot/W)+max(0,1-height/assist_height)+2*positive_nonfoot/(foot+positive_nonfoot)). Thus total=-J-Phi(initial); does not reward motionlessness.',
        selection='lexicographic: count of full D1-D8 passes, then lower worst J, then lower mean J; compare at all four dev phases every40 episodes and end. Best old/linear candidates retained.',
        stages='advance only if3/4 dev phases pass current support gate; if .2s fails stop and keep evidence, no movement reward or longer training silently enabled. If support passes, require next separate preregistration of movement reward formulas before training that branch; standing alone never walking.',
        phases=dict(train=rt.TRAIN,dev=rt.DEV,final_heldout=rt.TEST,implementation_adjustment='Planned half-millisecond heldout times rounded down .5ms to whole1ms command boundaries; otherwise pending neural window would include future spikes at capture. Registered before all simulations.'),
        controls='repeat, clear new-branch input/history (old retained), all motor cut, sensory cut; inference exported NumPy vs PyTorch parity; input history permutation only as saved-data diagnostic',
        intent='verify current neuron identities and independent evidence first; no source-confirmed walk/stop variation means intent preservation NOT_ESTABLISHED; cannot give labels to translator',
        scope='artificial learned interface, not anatomical discovery. No online body compensation, no new biological neurons/synapses, no higher actuator authority.',
        random_graph='only if .2s support gate passes, require matched-budget P2 before any connectome-specific claim',
        resource='isolated PyTorch2.6CPU environment referencing old site-packages read-only; one process/one heavy job; numerical release dt12.5us/neural.5ms/command1ms',
        units='NOT_ESTABLISHED',importance='核心必需'))
    dump(O/'INPUT_HASHES_BEFORE.json',{str(p):sha(p) for p in rt.files()})
    dump(O/'CHANGE_LEDGER.json',dict(old_file=str(rt.DOSE/'PREREGISTRATION.json'),old_sha256=sha(rt.DOSE/'PREREGISTRATION.json'),new_file=str(O/'PREREGISTRATION.json'),new_sha256=sha(O/'PREREGISTRATION.json'),reason='User approved nonlinear neural translator with short history and intent-preservation tests; body compensation deferred. Fixed174 anatomical readout restriction relaxed only in artificial interface.'))
    (O/'ENVIRONMENT.txt').write_text('executable='+sys.executable+'\npython='+sys.version+'\nplatform='+platform.platform()+'\n'+subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    roots=np.load(rt.b.B/'IDENTITIES.npz')['root_ids'];dn=[720575941433155799,720575941566493282]
    dump(O/'INTENT_AUDIT.json',dict(candidate_DNp09_roots=dn,present_local_indices={str(r):np.flatnonzero(roots==r).tolist() for r in dn},source_file=str(rt.R/'engine/embodied/descending_command.py'),source_sha256=sha(rt.R/'engine/embodied/descending_command.py'),walk_stop_causal_evidence_in_this_network='NOT_ESTABLISHED',policy='No added cells or fabricated intent labels. Without independent validated command condition, no intent-preservation claim even if motion succeeds.'))
    print('PREREGISTERED',O,flush=True)

def np_actor(path):
    model=NumpyTranslator(path)
    return lambda x,stochastic:(model(x),0.,0.)

def prepare():
    torch.manual_seed(20261105);model=Translator();(O/'MODELS').mkdir();model.export(O/'MODELS/zero.npz')
    sample=torch.randn(16,612)
    assert torch.count_nonzero(model(sample))==0 and torch.count_nonzero(model(torch.zeros_like(sample)))==0
    t=rt.Runtime();t.prepare()
    reference=js(rt.DOSE/'SUMMARY.json')['best'];result,_=t.episode('zero_reproduction',1.3,np_actor(O/'MODELS/zero.npz'),O/'MODELS/zero.npz')
    ref=np.load(Path(reference['folder'])/'BODY.npz');new=np.load(Path(result['folder'])/'BODY.npz')
    exact=all(np.array_equal(ref[k],new[k]) for k in ref.files);assert exact and result['J']==reference['J'],(exact,result['J'],reference['J'])
    # Independent filter reproduction from archived warmup binary spike sequence.
    data=np.load(O/'DATA/train_offset_0/TEACHER.npz');hist=SpikeHistory(t.cells);features=[]
    for i,pair in enumerate(data['spikes204'].reshape(-1,2,204)):
        features.append(hist.features())
        for sample in pair:
            full=np.zeros(12000,bool);full[t.cells]=sample;hist.consume(full)
    assert np.array_equal(np.asarray(features),data['features'])
    state=hist.state_dict();other=SpikeHistory(t.cells);other.load_state_dict(state);assert np.array_equal(hist.features(),other.features())
    dump(O/'ENGINEERING.json',dict(zero_residual_exact_reproduction=exact,reference_J=reference['J'],new_J=result['J'],filter_independent_recompute_exact=True,filter_state_restore_exact=True,zero_neural_features_zero_new_branch=True,model_parameters=sum(v.numel() for v in model.parameters()),reference_folder=reference['folder'],reproduced_folder=result['folder']))
    print('PREPARE VERIFIED EXACT',flush=True)

def dataset(prefix):
    paths=sorted((O/'DATA').glob(prefix+'*/TEACHER.npz'));xs=[];old=[];targets=[];groups=[];times=[]
    for p in paths:
        d=np.load(p);mask=(d['time']>=.3-1e-9)&(d['time']<1.4-1e-9);xs.append(d['features'][mask]);old.append(d['old_torque'][mask]);targets.append(d['target_torque'][mask]);groups.extend([str(p)]*int(mask.sum()));times.extend(d['time'][mask])
    return np.concatenate(xs),np.concatenate(old),np.concatenate(targets),groups,np.asarray(times)

def pretrain():
    torch.manual_seed(20261105);rng=np.random.default_rng(20261105)
    train=dataset('train_offset_');dev=dataset('dev_offset_');limits=np.array(js(O/'TRANSLATOR_IDENTITIES.json')['limits'],np.float32)
    X=torch.from_numpy(train[0]);old=torch.tensor(train[1]/limits,dtype=torch.float32);y=torch.tensor(train[2]/limits,dtype=torch.float32)
    VX=torch.from_numpy(dev[0]);vo=torch.tensor(dev[1]/limits,dtype=torch.float32);vy=torch.tensor(dev[2]/limits,dtype=torch.float32)
    dump(O/'DATA_SPLIT.json',dict(train_groups=sorted(set(train[3])),dev_groups=sorted(set(dev[3])),train_samples=len(X),dev_samples=len(VX),no_shared_trajectory=True,terminal_heldout_not_used=True))
    reports={}
    for kind in ['linear','mlp']:
        torch.manual_seed(20261105);model=Translator(kind);optim=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001);bestloss=float('inf');beststate=None;patience=0;hist=[]
        for epoch in range(100):
            order=rng.permutation(len(X))
            for start in range(0,len(X),256):
                ids=order[start:start+256];pred=torch.clamp(old[ids]+2*torch.tanh(model(X[ids])),-1,1);loss=((pred-y[ids])**2).mean();optim.zero_grad();loss.backward();optim.step()
            with torch.no_grad():
                vpred=torch.clamp(vo+2*torch.tanh(model(VX)),-1,1);v=((vpred-vy)**2).mean().item();tr=((torch.clamp(old+2*torch.tanh(model(X)),-1,1)-y)**2).mean().item()
            hist.append(dict(epoch=epoch,train_mse=tr,dev_mse=v))
            if v<bestloss-1e-6:bestloss=v;beststate=copy.deepcopy(model.state_dict());patience=0
            else:patience+=1
            if patience>=12:break
        model.load_state_dict(beststate);path=O/'MODELS'/('offline_'+kind+'.npz');model.export(path)
        npmodel=NumpyTranslator(path);np.testing.assert_allclose(npmodel(dev[0]),model(VX).detach().numpy(),rtol=1e-4,atol=1e-5)
        assert np.max(abs(npmodel(np.zeros((3,612),np.float32))))==0
        reports[kind]=dict(history=hist,best_dev_mse=bestloss,old_readout_dev_mse=float(((vo-vy)**2).mean()),model_file=str(path),torch_numpy_max_error=float(np.max(abs(npmodel(dev[0])-model(VX).detach().numpy()))))
        print('OFFLINE',kind,bestloss,'old',reports[kind]['old_readout_dev_mse'],flush=True)
    dump(O/'OFFLINE_FIT.json',reports)

class PPOActor:
    def __init__(self,kind,path):
        self.model=Translator(kind);self.model.restore_export(path)
        self.critic=nn.Sequential(nn.Linear(612,64),nn.Tanh(),nn.Linear(64,1));self.logstd=nn.Parameter(torch.full((42,),-3.))
        self.actor_optim=torch.optim.Adam(list(self.model.parameters())+[self.logstd],lr=.0001);self.critic_optim=torch.optim.Adam(self.critic.parameters(),lr=.0003)
    def __call__(self,x,stochastic):
        with torch.no_grad():
            xx=torch.from_numpy(x);mean=self.model(xx);dist=Normal(mean,self.logstd.exp());act=dist.sample() if stochastic else mean
            return act.numpy(),float(dist.log_prob(act).sum()),float(self.critic(xx).squeeze())
    def update(self,episodes):
        X=[];A=[];LP=[];ADV=[];RET=[]
        for traj in episodes:
            values=np.array([v[3] for v in traj]);rewards=np.array([v[4] for v in traj]);adv=np.zeros(len(traj));gae=0.
            for i in range(len(traj)-1,-1,-1):
                nxt=values[i+1] if i+1<len(traj) else 0.;gae=rewards[i]+nxt-values[i]+.95*gae;adv[i]=gae
            X.extend(v[0] for v in traj);A.extend(v[1] for v in traj);LP.extend(v[2] for v in traj);ADV.extend(adv);RET.extend(adv+values)
        x=torch.tensor(np.array(X),dtype=torch.float32);a=torch.tensor(np.array(A),dtype=torch.float32);oldlp=torch.tensor(LP,dtype=torch.float32);adv=torch.tensor(ADV,dtype=torch.float32);ret=torch.tensor(RET,dtype=torch.float32);adv=(adv-adv.mean())/(adv.std()+1e-8);metrics=[]
        for epoch in range(5):
            for idx in torch.randperm(len(x)).split(256):
                mean=self.model(x[idx]);dist=Normal(mean,self.logstd.exp());lp=dist.log_prob(a[idx]).sum(-1);ratio=torch.exp(lp-oldlp[idx]);loss=-torch.min(ratio*adv[idx],torch.clamp(ratio,.8,1.2)*adv[idx]).mean()-.0001*dist.entropy().sum(-1).mean()
                self.actor_optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(self.model.parameters())+[self.logstd],.5);self.actor_optim.step()
                with torch.no_grad():self.logstd.clamp_(-5.,-1.5)
                vl=((self.critic(x[idx]).squeeze(-1)-ret[idx])**2).mean();self.critic_optim.zero_grad();vl.backward();nn.utils.clip_grad_norm_(self.critic.parameters(),.5);self.critic_optim.step();metrics.append((float(loss.detach()),float(vl.detach())))
        return dict(actor_loss=float(np.mean([v[0] for v in metrics])),value_loss=float(np.mean([v[1] for v in metrics])),logstd_mean=float(self.logstd.mean().detach()))

def key(results):return (sum(r['validation']['passed'] for r in results),-max(r['J'] for r in results),-np.mean([r['J'] for r in results]))
def train():
    torch.manual_seed(20261105);np.random.seed(20261105);t=rt.Runtime();t.load_cache();validation=[];selection=[]
    def assess(path,label,duration=.2,phases=rt.DEV):
        results=[t.episode(label+'_'+str(phase),phase,np_actor(path),path,duration=duration)[0] for phase in phases];validation.extend(results);assert len(validation)<=64;dump(O/'VALIDATION_HISTORY.json',validation);return results
    for name in ['zero','offline_linear','offline_mlp']:
        path=O/'MODELS'/(name+'.npz');rs=assess(path,'initial_'+name);selection.append(dict(name=name,path=str(path),results=rs,key=key(rs)))
    incumbent=max(selection,key=lambda s:s['key']);mlp_seed=max((s for s in selection if s['name']!='offline_linear'),key=lambda s:s['key'])
    learner=PPOActor('mlp',mlp_seed['path']);updates=[];trainresults=[];rng=np.random.default_rng(20261105);trained=0;stage_reached=.2
    dump(O/'INITIAL_SELECTION.json',dict(candidates=selection,deployment_incumbent=incumbent,PPO_MLP_seed=mlp_seed))
    for duration,budget in [(.2,160),(.5,64),(1.,32)]:
        stage_reached=duration
        if duration!=.2:
            rs=assess(incumbent['path'],'stage_baseline_'+str(duration),duration);incumbent=dict(name='previous_stage',path=incumbent['path'],results=rs,key=key(rs))
        completed=0
        while completed<budget:
            take=min(10,budget-completed);modelpath=O/'MODELS'/('ppo_before_'+str(trained).zfill(4)+'.npz');learner.model.export(modelpath);batch=[]
            order=np.tile(rt.TRAIN,2);rng.shuffle(order)
            for j in range(take):
                result,traj=t.episode('ppo_'+str(trained).zfill(4),float(order[j]),learner,modelpath,duration=duration,stochastic=True);trainresults.append(result);batch.append(traj);trained+=1;completed+=1
            m=learner.update(batch);m.update(episodes=trained,duration_s=duration);updates.append(m)
            dump(O/'PPO_UPDATES.json',updates);dump(O/'TRAIN_HISTORY.json',trainresults)
            if completed%40==0 or completed==budget:
                path=O/'MODELS'/('ppo_after_'+str(trained).zfill(4)+'.npz');learner.model.export(path);rs=assess(path,'checkpoint_'+str(trained),duration);candidate=dict(name='ppo_'+str(trained),path=str(path),results=rs,key=key(rs))
                if candidate['key']>incumbent['key']:incumbent=candidate
                dump(O/'INCUMBENT.json',incumbent)
        if sum(r['validation']['passed'] for r in incumbent['results'])<3:
            print('STAGE GATE FAILED',duration,'no silent longer training',flush=True);break
        # Movement reward needs separate explicit registration if support succeeds.
        if duration==1.:break
    final=assess(incumbent['path'],'FINAL_HELDOUT',stage_reached,rt.TEST)
    # Fixed representative phase is not selected by outcome.
    path=incumbent['path'];base=t.episode('FINAL_base',1.3,np_actor(path),path,duration=stage_reached)[0];repeat=t.episode('FINAL_repeat',1.3,np_actor(path),path,duration=stage_reached)[0]
    a=np.load(Path(base['folder'])/'BODY.npz');bb=np.load(Path(repeat['folder'])/'BODY.npz');assert base['J']==repeat['J'] and all(np.array_equal(a[k],bb[k]) for k in a.files)
    cut=t.episode('FINAL_all_motor_cut',1.3,np_actor(path),path,duration=stage_reached,cut=True)[0];sens=t.episode('FINAL_sensory_cut',1.3,np_actor(path),path,duration=stage_reached,sensory_cut=True)[0];newcut=t.episode('FINAL_translator_cut',1.3,np_actor(path),path,duration=stage_reached,translator_cut=True)[0]
    dump(O/'SUMMARY.json',dict(incumbent=incumbent,final_heldout=final,representative=base,repeat=repeat,motor_cut=cut,sensory_cut=sens,translator_cut=newcut,training_evaluations=trained,validation_evaluations=len(validation)+5,stage_reached_s=stage_reached,walking_certified=False,intent_preservation='NOT_ESTABLISHED',P2_run=False,P2_reason='conditional on support gate; see actual final checks',all_results_retained=True))
    print('TRAIN COMPLETE',trained,incumbent['name'],flush=True)

def main():
    mode=sys.argv[1]
    if mode=='prereg':prereg();return
    before=js(O/'INPUT_HASHES_BEFORE.json');assert before=={str(p):sha(p) for p in rt.files()}
    try:
        if mode=='prepare':prepare()
        elif mode=='fit':pretrain()
        elif mode=='train':train()
    except BaseException:
        import traceback
        dump(O/('FAILURE_'+mode+'.json'),dict(traceback=traceback.format_exc()));raise
    finally:
        after={str(p):sha(p) for p in rt.files()};dump(O/('INPUT_HASHES_AFTER_'+mode+'.json'),after);assert before==after
if __name__=='__main__':main()
