"""Independent saved PPO training accounting; parent executes after FINAL_STATUS.
No trainer/runtime imports, Torch, model compilation, FK, physics or replay. Reward,
logprob, history/input contracts and retained checkpoints are checked from arrays.
Exact all-update GAE/critic reconstruction is explicitly NOT claimed: rollout
next_values and intermediate actor/critic weights were not retained in episodeNPZs.
"""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T123634Z')
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
TRAIN=ROOT/'experiments/s5-ann-walk/20261107T123634Z/train_residual.py'
TRAIN_HASH='05df6626bb6eb1335cc8be75b4fad56f299ff651ebb54ca7dc9980b571939950'
MODEL=Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z/MODEL_JOINT_TORQUE.xml')
W=24.446354755878005

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def encode(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    raise TypeError(type(v).__name__)
def error(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:return float('inf')
    return float(abs(a.astype(float)-b.astype(float)).max()) if a.size else 0.
def rotation(quat):
    w,x,y,z=quat
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def log_probability(latent,mean,logstd):
    latent,mean,logstd=[np.asarray(a,np.float64) for a in (latent,mean,logstd)]
    lognormal=-.5*((latent-mean)/np.exp(logstd))**2-logstd-.5*np.log(2*np.pi)
    logjac=2*(np.log(2)-latent-np.logaddexp(0.,-2*latent))
    return np.sum(lognormal-logjac,axis=-1)
def reward(metrics,action,terminal):
    value=float(np.clip(metrics['filtered_forward_velocity']/.3,-1,1.5)-2*metrics['Fn_weighted_slip_squared']/(W*.3**2)
        -.05*np.sum(metrics['body_omega']**2)-.01*np.sum(metrics['gravity_body'][:2]**2)-.01*np.mean(action**2)
        -5*(np.sum(metrics['loads'])/W-1)**2-50*metrics['nonfoot_force']/W)
    return value-25*terminal

def gae(rewards,values,nextvalues,terminated,done):
    out=np.zeros(len(rewards));carry=0.
    for i in range(len(rewards)-1,-1,-1):
        delta=rewards[i]+.995*nextvalues[i]*(not terminated[i])-values[i]
        carry=delta+.995*.95*(not done[i])*carry;out[i]=carry
    return out.astype(np.float32),(out+values).astype(np.float32)

def main():
    if not (OUT/'FINAL_STATUS.json').exists():print(json.dumps(dict(status='TRAINING_NOT_FINAL',physics_steps=0)));return 3
    final=read(OUT/'FINAL_STATUS.json');protocol=read(OUT/'PREREGISTRATION.json');logs=read(OUT/'TRAIN_LOG.json') if (OUT/'TRAIN_LOG.json').exists() else []
    config=protocol['config'];base=load(protocol['base_model']);reference=load(protocol['reference']);qa=reference['qpos_addresses'].astype(int);va=qa-1
    document=ET.parse(MODEL).getroot();key=document.find('keyframe/key');q0=np.fromstring(key.attrib['qpos'],sep=' ');v0=np.fromstring(key.attrib.get('qvel',' '.join(['0']*48)),sep=' ')
    joint={j.attrib['name']:j for j in document.iter('joint') if 'name' in j.attrib};motors=list(document.find('actuator'));names=[x.attrib['joint'] for x in motors]
    low=np.array([np.fromstring(joint[name].attrib['range'],sep=' ')[0] for name in names]);high=np.array([np.fromstring(joint[name].attrib['range'],sep=' ')[1] for name in names])
    native=np.array([min(-np.fromstring(x.attrib['ctrlrange'],sep=' ')[0],np.fromstring(x.attrib['ctrlrange'],sep=' ')[1]) for x in motors]);half=np.maximum((high-low)/2,.1)
    indices=np.array([names.index(name) for name in protocol['action']['joints']]);checkpoints={};checkpoint_reports=[];inputs={}
    for update in (8,16,24,32):
        path=OUT/('MODEL_RESIDUAL_u'+format(update,'02d')+'.npz')
        if not path.exists():continue
        weights=load(path);inputs[str(path)]=sha(path);metadata=json.loads(str(weights['metadata_json'].item()))
        c=dict(base_fields_bitwise=all(k in weights and np.array_equal(weights[k],a) for k,a in base.items()),
            input_shapes=weights['actor_w0'].shape==(128,477) and weights['actor_w1'].shape==(128,128) and weights['actor_w2'].shape==(6,128),
            scale=float(weights['delta_scale'])==.003,indices=np.array_equal(weights['indices'],indices),
            names=list(weights['proprio_names'].astype(str))==protocol['observations']['proprio']['names'],
            base_hash=str(weights['base_sha256'].item())==protocol['base_sha256'],input_order=str(weights['network_input_order'].item())=='proprio183 THEN basebuf[lags].reshape294',
            config=metadata['config']==config,finite=all(np.isfinite(a).all() for k,a in weights.items() if a.dtype.kind in 'fiu'))
        checkpoints[update]=weights;checkpoint_reports.append(dict(update=update,path=str(path),checks=c))
    checks=dict(frozen_trainer_source=sha(TRAIN)==TRAIN_HASH,full_final_status=final['status']=='TRAINED_NOT_PHYSICALLY_ACCEPTED',
        completed_32_updates=final['completed_updates']==len(logs)==32,four_checkpoints=len(checkpoints)==4,
        checkpoint_contract=all(all(x['checks'].values()) for x in checkpoint_reports),base_hash=sha(protocol['base_model'])==protocol['base_sha256'],
        declared_dimensions=protocol['observations']['dimensions']==477 and protocol['observations']['proprio']['dimensions']==183,
        feature_exclusions='worldposition' in protocol['observations']['proprio']['excluded'] and 'time' in protocol['observations']['proprio']['excluded'])
    maximum={};episode_reports=[];decision_records=[];commands_total=warm_total=neural_total=failures_total=0;first_load_fields_excluded=0
    def compare(name,a,b):maximum[name]=max(maximum.get(name,0.),error(a,b))
    for path in sorted((OUT/'episodes').glob('episode_*.npz')):
        inputs[str(path)]=sha(path);e=load(path)
        if 'observation' not in e:
            episode_reports.append(dict(path=str(path),partial_no_policy_rows=True,reason=str(e['end_reason'].item())));continue
        count=len(e['time']);commands=e['COMMANDS'];commands_total+=len(commands);history=None;filtered=0.;warm=neural=0
        compare('episode_policy_clock',e['time'],np.arange(count)*.005)
        compare('BODY_postpolicy_clock',e['BODY_time'],e['time']+.005)
        compare('native_limits',e['limits'],native);compare('kp128',e['kp'],native/.2*128);compare('kd128',e['kd'],native*.005*np.sqrt(128))
        target_after=np.clip(commands[:,1:43]+.001*np.clip(commands[:,43:85]+commands[:,85:127],-30,30),low,high)
        if len(commands)>1:compare('command_target_integrator',target_after[:-1],commands[1:,1:43])
        complete_commands=len(commands)==count*5;episode_errors={}
        for i in range(count):
            target=e['target_before'][i];compare('target_before_matches_command',target,commands[i*5,1:43]);compare('command_rate_held',commands[i*5:(i+1)*5,43:85],np.repeat(e['commanded_rate'][i:i+1],5,axis=0))
            normalized=(np.asarray(target,np.float32)-base['center'])/base['scale'];history=np.repeat(normalized[None],int(base['K']),axis=0) if history is None else np.concatenate((history[1:],normalized[None]),axis=0)
            hist=history[base['lags'].astype(int)].reshape(-1);compare('exact_once_normalized_history',e['observation'][i,183:],hist)
            preq=q0 if i==0 else e['BODY_qpos'][i-1];prev=v0 if i==0 else e['BODY_qvel'][i-1];R=rotation(preq[3:7]);loads=np.zeros(6) if i==0 else e['loads'][i-1]
            sensory=np.r_[(preq[qa]-q0[qa])/half,(target-preq[qa])/half,prev[va]/10,(target-q0[qa])/half,loads/W,R.T@[0,0,-1],R.T@prev[:3],prev[3:6]]
            sensory=np.clip(sensory,-10,10).astype(np.float32)
            if i==0:
                keep=np.r_[np.arange(168),np.arange(174,183)];first_load_fields_excluded+=6;compare('proprioception_except_initial_unlogged_load6',e['observation'][i,:183][keep],sensory[keep])
            else:compare('all183_proprioception_from_saved_prestate',e['observation'][i,:183],sensory)
            filtered+=float(-np.expm1(-.005/.1))*(e['body_linear_velocity'][i,0]-filtered);compare('reward_forward_filter',e['filtered_forward_velocity'][i],filtered)
            postR=rotation(e['BODY_qpos'][i,3:7]);compare('postmetric_linear_velocity',e['body_linear_velocity'][i],postR.T@e['BODY_qvel'][i,:3]);compare('postmetric_omega',e['body_omega'][i],e['BODY_qvel'][i,3:6]);compare('postmetric_gravity',e['gravity_body'][i],postR.T@[0,0,-1])
            if e['control'][i]=='teacher_warmup':
                warm+=1;compare('warm_reward_zero',e['reward'][i],0.);compare('warm_residual_zero',e['residual_delta_q'][i],np.zeros(42));continue
            neural+=1;update=1+neural_total//256;terminated=bool(e['terminated'][i]);done=bool(e['done'][i]);action=e['residual_action'][i];latent=e['latent_action'][i]
            # PhysicalFailure exceptions synthesize zero metrics and exactly-25;
            # distinguish them from ordinary finite COM/nonfoot terminal penalties.
            synthetic_exception=terminated and not np.any(e['loads'][i]) and not np.any(e['gravity_body'][i])
            metrics={key:e[key][i] for key in ('filtered_forward_velocity','Fn_weighted_slip_squared','body_omega','gravity_body','loads','nonfoot_force')}
            # Trainer saved body_omega as a live qvel view. Recover actual reward
            # omega from the separately copied poststep BODY_qvel, retain mismatch.
            metrics['body_omega']=e['BODY_qvel'][i,3:6]
            calculated=-25. if synthetic_exception else reward(metrics,action,terminated)
            compare('declared_reward_including_terminal',e['reward'][i],calculated);compare('tanh_action_from_latent',action,np.tanh(latent))
            z=hist.reshape(1,-1);z=np.tanh(z@base['w0'].T+base['b0']);z=np.tanh(z@base['w1'].T+base['b1']);baseq=base['center']+base['scale']*(z@base['w2'].T+base['b2'])[0]
            delta=np.zeros(42);delta[indices]=.003*np.asarray(action,float);compare('frozen_base_prediction',e['base_qnext'][i],baseq);compare('residual_delta_mapping',e['residual_delta_q'][i],delta)
            compare('neural_command_formula',e['commanded_rate'][i],np.clip((baseq+delta-target)/.005,-30,30))
            logstd=np.full(6,-3.) if update==1 else np.asarray(logs[update-2]['PPO']['logstd']);logstd=np.clip(logstd,-5,-1.5)
            lp=log_probability(latent,e['actor_mean_raw'][i],logstd);compare('squashed_Normal_logprob',e['logprob'][i],lp)
            if update==1:compare('initial_zero_actor_mean',e['actor_mean_raw'][i],np.zeros(6))
            cp=checkpoints.get(update-1)
            if cp is not None:
                h=np.tanh(e['observation'][i]@cp['actor_w0'].T+cp['actor_b0']);h=np.tanh(h@cp['actor_w1'].T+cp['actor_b1'])
                compare('retained_checkpoint_actor_mean',e['actor_mean_raw'][i],h@cp['actor_w2'].T+cp['actor_b2']);compare('retained_checkpoint_critic',e['critic_value'][i],np.asarray((h@cp['critic_w'].T+cp['critic_b']).item()))
            if terminated:failures_total+=1
            timeout=neural>=1400;episode_errors.setdefault('done_flags',True);episode_errors['done_flags']&=done==(terminated or timeout)
            episode_errors.setdefault('terminal_condition',True)
            actual_failure=bool(e['BODY_com'][i,2]<.95 or e['nonfoot_force'][i]>.05*W or synthetic_exception)
            episode_errors['terminal_condition']&=terminated==actual_failure
            decision_records.append(dict(update=update,reward=float(e['reward'][i]),value=float(e['critic_value'][i]),done=done,terminated=terminated,source_episode=str(path)))
            neural_total+=1
            if i==200 and 'HANDOVER_history' in e:
                compare('handover_history',e['HANDOVER_history'],history);compare('handover_target',e['HANDOVER_target'],target)
        warm_total+=warm
        episode_reports.append(dict(path=str(path),policy_rows=count,warm_policy_rows=warm,neural_decisions=neural,commands=len(commands),
            complete_command_groups=complete_commands,warm_is200=warm==200,flags=episode_errors,end_reason=str(e['end_reason'].item()),
            final_time=float(e['FINAL_time']),last_policy_endpoint_time=float(e['BODY_time'][-1]),input_hash=sha(path)))
    update_reports=[]
    for log in logs:
        u=log['update'];rows=[r for r in decision_records if r['update']==u];rewards=np.asarray([r['reward'] for r in rows],np.float32);values=np.asarray([r['value'] for r in rows],np.float32)
        c=dict(decisions256=len(rows)==256,global_decision_counter=log['counters']['neural_policy_decisions']==u*256,
            reward_mean=bool(len(rows) and abs(float(rewards.mean())-log['reward_mean'])<1e-5),reward_min=bool(len(rows) and abs(float(rewards.min())-log['reward_min'])<1e-5),reward_max=bool(len(rows) and abs(float(rewards.max())-log['reward_max'])<1e-5),
            episode_boundaries=sum(r['done'] for r in rows)==log['episode_boundaries'],advantage_length=log['advantage_length']==256,
            returns_advantage_value_mean=bool(len(rows) and np.isclose(log['returns_mean']-log['advantage_mean'],float(values.mean()),atol=5e-4,rtol=1e-4)),
            logstd_bounds=bool(np.all(np.asarray(log['PPO']['logstd'])>=-5) and np.all(np.asarray(log['PPO']['logstd'])<=-1.5)))
        update_reports.append(dict(update=u,checks=c))
    counts=final['counters'];complete=checks['full_final_status'];saved_steps=commands_total*80
    checks.update(decisions8192=neural_total==counts['neural_policy_decisions']==8192,warm_counts=warm_total==counts['warmup_policy_ticks'],
        commands_accounted=commands_total==counts['command_ticks'],native80_percommand=saved_steps==counts['physics_steps'],
        resource_bounds=counts['warmups_started']<=16 and counts['physics_steps']<=6000000,
        all_episodes_retained=len(episode_reports)==counts['episodes_saved']==counts['resets']==counts['warmups_started'],
        every_warmup200=all(x.get('warm_is200',False) for x in episode_reports),complete_policy_command_groups=all(x.get('complete_command_groups',False) for x in episode_reports),
        failure_records=failures_total==counts['failures']==len(final['failures']),update_summaries=all(all(x['checks'].values()) for x in update_reports),
        recorded_done_flags=all(all(x.get('flags',{}).values()) for x in episode_reports))
    thresholds={'squashed_Normal_logprob':2e-4,'retained_checkpoint_actor_mean':5e-7,'retained_checkpoint_critic':2e-6,
        'all183_proprioception_from_saved_prestate':5e-6,'proprioception_except_initial_unlogged_load6':5e-6,'reward_forward_filter':1e-10,'declared_reward_including_terminal':1e-8,
        'kp128':1e-10,'kd128':1e-12,'neural_command_formula':5e-4,'frozen_base_prediction':1e-5}
    checks['saved_array_mechanisms']=all(value<=thresholds.get(key,1e-6) for key,value in maximum.items() if key!='postmetric_omega')
    raw_metric_integrity=dict(ORIGINAL_logged_body_omega_matches_copied_post_BODY_qvel=maximum.get('postmetric_omega',0.)<=1e-6,
        original_logged_body_omega_max_error=maximum.get('postmetric_omega',0.),
        defect='train_residual.measure omega=self.d.qvel[3:6] is a liveview; record stores it withoutcopy, so episodestacking sees finalepisodevelocity. Actual immediate reward used currentomega correctly.',
        recovery='Reward recomputation uses copied BODY_qvel[:,3:6] at the same poststep time. Originalepisodearrays/model/source are preserved; no silent repair.',
        affects_training_reward_execution=False,affects_saved_body_omega_sidechannel=True)
    frozen={path:Path(path).exists() and sha(path)==digest for path,digest in protocol['source_hashes'].items()};checks['registered_sources_unchanged']=all(frozen.values())
    for path in [OUT/'PREREGISTRATION.json',OUT/'FINAL_STATUS.json',OUT/'TRAIN_LOG.json',Path(__file__),MODEL]:inputs[str(path)]=sha(path)
    status=('TRAINING_ACCOUNTING_VERIFIED_WITH_LOGGING_DEFECT_NOT_ACCEPTANCE' if not raw_metric_integrity['ORIGINAL_logged_body_omega_matches_copied_post_BODY_qvel'] else 'TRAINING_ACCOUNTING_VERIFIED_NOT_ACCEPTANCE') if all(checks.values()) else 'PARTIAL_OR_FAILED_TRAINING_AUDIT'
    report=dict(status=status,checks=checks,raw_metric_integrity=raw_metric_integrity,
        final_status=final['status'],completed_updates=final['completed_updates'],counts_recorded=counts,
        counts_reconstructed=dict(warm_policy=warm_total,neural_decisions=neural_total,command_ticks=commands_total,native_steps_if_commands_complete=saved_steps,terminated_decisions=failures_total),
        episodes=episode_reports,updates=update_reports,checkpoints=checkpoint_reports,maximum_array_errors=maximum,array_error_tolerances=thresholds,
        evidence_limits=dict(first_policy_initial_load_features_unverified=first_load_fields_excluded,
            exact_GAE_allupdates='NOT_RECONSTRUCTABLE: perdecision next_values/advantages/returns and intermediate critics not retained; source terminal/timeout mask logic reviewed, syntheticGAEcheckedseparately.',
            actor_weight_forward='Only initial zero mean androllouts9/17/25 canbecheckedagainstretainedpriorupdate8/16/24weights. No claim all32actorstatesforwardverified.',
            reward_contact_metrics='Recomputedrewarduses saved load/slipsquared metrics; materialcontactphysics itself requires parentstrictreplay.',
            exclusions='183contract rebuiltwithoutclock/worldposition/rootheight/yawreference/anchors;477 includes294declaredcausaltargethistory.'),
        source_hashes_unchanged=frozen,input_hashes=inputs,analysis_counts=dict(new_physics_steps=0,model_compile=0,FK_calls=0,replay=0),
        limitation='Training accounting/trace integrity is not rewardacceptance, physicalD/slip/heading success, neuralnecessity or biologicalvalidation. Partial/guardabortedtraining cannotbeclaimedcomplete.')
    output=OUT/'AUDIT_TRAINING.json'
    if output.exists():output.read_bytes();raise FileExistsError('preserve prior training audit')
    output.write_text(json.dumps(report,indent=2,allow_nan=False,default=encode)+'\n')
    print(json.dumps(dict(status=report['status'],checks=checks,maximum_array_errors=maximum,counts=report['counts_reconstructed'],output=str(output)),default=encode),flush=True)
    return 0 if all(checks.values()) else 2
if __name__=='__main__':raise SystemExit(main())
