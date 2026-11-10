"""Fixed64-update continuation, restoring actor/critic/Adam and both RNG states.

Physical state is deliberately fresh at an episode boundary, not checkpoint-resumed.
Only new diagnostic array copies and rollout GAE audit fields differ from123634.
"""
import argparse
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import torch

BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T123634Z/train_residual.py')
BASE_SHA256='05df6626bb6eb1335cc8be75b4fad56f299ff651ebb54ca7dc9980b571939950'
HERE=Path(__file__).resolve()
PARENT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T123634Z')
CHECKPOINT=PARENT/'CHECKPOINT_LAST.pt'
OUT=PARENT.parent/'20261107T130313Z'
source=BASE.read_text();assert hashlib.sha256(BASE.read_bytes()).hexdigest()==BASE_SHA256
patches=[
    ("OUT=BASE_MODEL.parent.parent/'20261107T123634Z'","OUT=BASE_MODEL.parent.parent/'20261107T130313Z'"),
    ('updates=32,rollout=256,decisions=8192','updates=64,rollout=256,decisions=16384'),
    ('maximum_warmups=16,maximum_native_steps=6000000','maximum_warmups=32,maximum_native_steps=12000000'),
    ('body_omega=omega,','body_omega=omega.copy(),'),
    ('body_linear_velocity=velocity,','body_linear_velocity=velocity.copy(),'),
    ('def train(steps=8192):','def train(steps=16384):'),
    ("Fixed protocol requires exactly8192neuraldecisions","Fixed continuation requires exactly16384neuraldecisions"),
    ("    try:\n        env=Environment();obs=env.reset()",
     "    restore_checkpoint(net,optimizer,rng)\n    try:\n        env=Environment();obs=env.reset()"),
    ("for update in range(1,CFG['updates']+1):","for update in range(33,97):"),
    ("rollout={k:[] for k in ('obs','latent','logprob','values','rewards','next_values','terminated','done')}",
     "rollout={k:[] for k in ('obs','latent','logprob','values','rewards','next_values','terminated','done','episode','episode_time')}"),
    ('                for key,item in dict(obs=obs.copy(),',
     "                rollout['episode'].append(env.episode);rollout['episode_time'].append(env.policy_time)\n                for key,item in dict(obs=obs.copy(),"),
    ('            arrays.update(advantage=advantage,returns=returns)',
     "            arrays.update(advantage=advantage,returns=returns,update=np.array(update),local_update=np.array(update-32),decision_indices=np.arange((update-33)*256,(update-32)*256))\n            np.savez_compressed(OUT/('ROLLOUT_u'+format(update,'03d')+'.npz'),**arrays)"),
    ("if update%8==0:exports.append(export(net,OUT/('MODEL_RESIDUAL_u'+format(update,'02d')+'.npz'),metadata))",
     "if update in (48,64,80,96):exports.append(export(net,OUT/('MODEL_RESIDUAL_u'+format(update,'03d')+'.npz'),metadata))"),
    ("update<CFG['updates']","update<96"),
    ('completed_updates=len(logs),partial_update_aborted=True',
     'completed_updates=len(logs),completed_updates_global=32+len(logs),partial_update_aborted=True'),
    ('counters=env.total,failures=env.failures,completed_updates=len(logs),exports=exports,episode_files_saved_before_final_status=True',
     'counters=env.total,prior_counters=prior_counters,physics_steps_including_parent=prior_counters[\'physics_steps\']+env.total[\'physics_steps\'],failures=env.failures,completed_updates=len(logs),completed_updates_global=32+len(logs),exports=exports,episode_files_saved_before_final_status=True'),
]
effective=source
for old,new in patches:
    assert effective.count(old)==1,('Expectedonefrozenmatch',old,effective.count(old))
    effective=effective.replace(old,new,1)
EFFECTIVE_SHA256=hashlib.sha256(effective.encode()).hexdigest()
runtime=types.ModuleType('continued_physical_residual');runtime.__file__=str(BASE)
exec(compile(effective,str(HERE)+'::effective','exec'),runtime.__dict__)
prior_counters=json.loads((PARENT/'FINAL_STATUS.json').read_text())['counters']
runtime.prior_counters=prior_counters


def restore_checkpoint(net,optimizer,rng):
    # Trusted local file produced by this project. No arbitrary downloaded pickle.
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert checkpoint['update']==32 and checkpoint['counters']['neural_policy_decisions']==8192
    net.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
    torch.set_rng_state(checkpoint['torch_rng']);rng.bit_generator.state=checkpoint['numpy_rng']
    return checkpoint


runtime.restore_checkpoint=restore_checkpoint
original_prereg=runtime.preregister


def preregister():
    # Newexperimentpre-registration is written before modelsetup or physicalFK.
    metadata=original_prereg()
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert checkpoint['update']==32
    path=OUT/'EFFECTIVE_CONTINUATION.py';assert not path.exists();path.write_text(effective)
    metadata.update(stage='S5 fixed residual-PPO continuation globalupdates33-96',
        continuation=dict(parent_source=str(BASE),parent_source_sha256=BASE_SHA256,
            checkpoint=str(CHECKPOINT),checkpoint_sha256=runtime.n.sha(CHECKPOINT),restore_update=32,
            restore='Network+Adam+TorchRNG+NumPyRNG restored AFTER newnetworkconstructor and BEFORE physicalwarmup. Nohiddenphysicalstatecontinuation; freshoriginalkeyframe+1srealteacherwarmup,newbasehistory.',
            global_updates=[33,96],new_updates=64,new_decisions=16384,prior_counters=prior_counters,
            new_resource_guard=dict(max_warmups=32,max_native_steps=12000000),
            unchanged='AllPPO/reward/actor477/.003/base/modelphysics/nativecaps/gain128hyperparamsunchanged; onlymoreexperience, nostudent/teachergeometry/gain/contactsearch.',
            diagnostic_copies='body_omega+body_linear_velocity copiedbeforetrace; oldsource/oldtraces unchanged.',
            new_rollout_audit='64ROLLOUT_u033..096.npz includesallobs/latents/logprobs/values/rewards/next_values/terminated/done/GAEadvantage/returns/update/decisionindices/episode/time. Prior8192decisionslackedthese; notretroactivelyreconstructed.',
            exports=[48,64,80,96],new_physics_counters='newrunonly; FINAL_STATUSincludesprior3756800steps+newactualstepsseparately'),
        adaptation=dict(wrapper=str(HERE),wrapper_sha256=runtime.n.sha(HERE),effective_source=str(path),effective_sha256=EFFECTIVE_SHA256,
            patches=[dict(old=old,new=new,required_matches=1) for old,new in patches]))
    metadata['source_hashes'].update({str(HERE):runtime.n.sha(HERE),str(CHECKPOINT):runtime.n.sha(CHECKPOINT),
        str(PARENT/'FINAL_STATUS.json'):runtime.n.sha(PARENT/'FINAL_STATUS.json')})
    metadata['network']=metadata['network'].replace('actorheadzeros; logstd−3clamped','actorhead/logstd restoredfromupdate32; logstdclamped')
    metadata['PPO']=metadata['PPO'].replace('32x256decisions','64x256additionaldecisions')
    metadata['exports']='FourcompositeNPZ globalu048/u064/u080/u096; optimizerCHECKPOINT_LAST overwrittennewdironly. No finalalias/bestselection.'
    metadata['resource_guard']='32newphysicalwarmups or12millionnewnative-steps; retainpartialcheckpoint/traces onfailure.'
    runtime.n.dump(OUT/'PREREGISTRATION.json',metadata);return metadata


runtime.preregister=preregister


def self_test():
    def forbidden(*a,**k):raise AssertionError('No physics in continuationpreflight')
    runtime.mj.mj_step=forbidden;torch.set_num_threads(4)
    net=runtime.ActorCritic();opt=torch.optim.Adam(net.parameters(),lr=runtime.CFG['lr']);rng=np.random.default_rng()
    ck=restore_checkpoint(net,opt,rng)
    assert all(torch.equal(net.state_dict()[k],v) for k,v in ck['model'].items())
    state=opt.state_dict();assert state['param_groups']==ck['optimizer']['param_groups']
    for key,value in state['state'].items():
        for field,v in value.items():
            expected=ck['optimizer']['state'][key][field]
            assert torch.equal(v,expected) if torch.is_tensor(v) else v==expected
    assert torch.equal(torch.get_rng_state(),ck['torch_rng'])
    assert json.dumps(rng.bit_generator.state,sort_keys=True)==json.dumps(ck['numpy_rng'],sort_keys=True)
    with torch.no_grad():
        obs=torch.from_numpy(np.random.default_rng(7).normal(0,.2,(16,477)).astype(np.float32));mean,value=net(obs)
    exported=runtime.export_arrays(net,ck['metadata']);npactor=runtime.NumpyActor.__new__(runtime.NumpyActor);npactor.w=exported
    a,v=npactor(obs.numpy());parity=float(max(abs(a-torch.tanh(mean).numpy()).max(),abs(v-value.numpy()).max()));assert parity<1e-5
    with np.load(PARENT/'MODEL_RESIDUAL_u32.npz') as old:
        assert all(np.array_equal(exported[k],old[k]) for k in old.files if k!='metadata_json')
    assert runtime.CFG['updates']==64 and runtime.CFG['decisions']==16384
    assert runtime.CFG['maximum_warmups']==32 and runtime.CFG['maximum_native_steps']==12000000
    result=dict(physics_steps=0,network_checkpoint_bitwise=True,Adam_state_bitwise=True,TorchRNG_restored=True,
        NumPyRNG_restored=True,composite_u32_arrays_bitwise=True,Torch_NumPy_parity=parity,
        source_sha256=runtime.n.sha(HERE),effective_sha256=EFFECTIVE_SHA256,checkpoint_sha256=runtime.n.sha(CHECKPOINT))
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'SELF_TEST.json').exists():raise FileExistsError('Preservepreflight')
    runtime.n.dump(OUT/'SELF_TEST.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--continue',dest='continue_training',action='store_true');group.add_argument('--self-test',action='store_true')
    ap.add_argument('--steps',type=int,default=16384);args=ap.parse_args()
    if args.steps!=16384:ap.error('Fixedcontinuation16384decisionsonly')
    runtime.train(args.steps) if args.continue_training else self_test()
