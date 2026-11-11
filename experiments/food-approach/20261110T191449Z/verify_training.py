"""Independent saved-checkpoint audit; no physics, no optimization, no source edits."""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import torch
from conditional_motor import HERE, load_motor, load_baseline_actor, sha256


def exact(a,b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return type(a) is type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def verify(checkpoint):
    torch.set_num_threads(1)
    checkpoint=Path(checkpoint).resolve()
    initial_digest=sha256(checkpoint)
    model,payload=load_motor(checkpoint)
    config=payload['config'];stage=config['stage']
    if stage not in ('adapter','finetune'):raise ValueError('unknown stage')
    total=100 if stage=='adapter' else 500
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise ValueError('failed: '+name)
    for category in ('source_hashes','data_hashes'):
        entries=config[category]
        check(category+'_nonempty',bool(entries))
        for path,digest in entries.items():
            check(category+':'+path,Path(path).is_absolute() and sha256(path)==digest)
    check('model_reload_all_tensors_bitexact',exact(model.state_dict(),payload['state_dict']))
    model.set_training_stage(stage)
    opt=torch.optim.Adam([model.command_encoder.weight],lr=.003 if stage=='adapter' else .001)
    if stage=='finetune':opt.add_param_group(dict(params=list(model.base.parameters()),lr=.0001))
    opt.load_state_dict(payload['optimizer'])
    check('optimizer_reload_all_state_bitexact',exact(opt.state_dict(),payload['optimizer']))
    expected_names=[['command_encoder.weight']]
    if stage=='finetune':expected_names.append(['base.'+name for name,_ in model.base.named_parameters()])
    named=dict(model.named_parameters());ids_by_object={id(p):name for name,p in named.items()}
    check('optimizer_group_count',len(opt.param_groups)==len(expected_names))
    counters={};group_names=[]
    expected_lrs=[.003] if stage=='adapter' else [.001,.0001]
    for group_index,(group,names) in enumerate(zip(opt.param_groups,expected_names)):
        actual_names=[ids_by_object[id(p)] for p in group['params']]
        group_names.append(actual_names)
        check('group_order_'+str(group_index),actual_names==names)
        check('group_hyperparameters_'+str(group_index),group['lr']==expected_lrs[group_index] and group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0 and not group['amsgrad'] and not group['maximize'])
        for name,p in zip(names,group['params']):
            state=opt.state[p]
            check('moment_keys:'+name,set(state)=={'step','exp_avg','exp_avg_sq'})
            step=state['step'];expected_step=total if group_index==0 else 400
            check('counter:'+name,torch.is_tensor(step) and step.numel()==1 and torch.isfinite(step).all() and float(step)==expected_step)
            counters[name]=float(step)
            for key in ('exp_avg','exp_avg_sq'):
                value=state[key]
                check(key+':'+name,value.shape==p.shape and value.dtype==p.dtype and bool(torch.isfinite(value).all()))
            check('nonnegative_second_moment:'+name,bool((state['exp_avg_sq']>=0).all()))
    flat=[name for names in group_names for name in names]
    expected_trainable={name for name,p in model.named_parameters() if p.requires_grad}
    check('optimizer_covers_trainable_once',len(flat)==len(set(flat)) and set(flat)==expected_trainable)
    check('no_orphan_optimizer_state',len(opt.state)==len(flat))
    curve=payload['curve']
    check('complete_global_curve',len(curve)==total+1 and [r['step'] for r in curve]==list(range(total+1)))
    check('finite_metrics_curve',all(math.isfinite(float(row['wall_s'])) and all(math.isfinite(float(v)) for v in row.get('train_MSE',{}).values()) and all(math.isfinite(float(v)) for m in row.get('metrics',{}).values() for v in m.values()) for row in curve))
    check('recorded_loss_full_sequence','full contiguous2880 BPTT' in config['loss'])
    parent_summary=None
    if stage=='finetune':
        parent_path=Path(config['resume'])
        check('parent_digest',parent_path.is_absolute() and sha256(parent_path)==config['resume_sha256'])
        _,parent=load_motor(parent_path)
        check('parent_stage_and_curve',parent['config']['stage']=='adapter' and len(parent['curve'])==101 and parent['curve'][-1]['step']==100)
        check('exact_parent_curve_prefix',exact(curve[:101],parent['curve']))
        check('same_parent_dataset',config['data_hashes']==parent['config']['data_hashes'])
        for category in ('source_hashes','data_hashes'):
            for path,digest in parent['config'][category].items():check('parent_'+category+':'+path,sha256(path)==digest)
        parent_opt=parent['optimizer']
        check('parent_optimizer_single_adapter',len(parent_opt['param_groups'])==1 and parent_opt['param_groups'][0]['params']==[0] and set(parent_opt['state'])=={0})
        check('parent_adapter_counter100',float(parent_opt['state'][0]['step'])==100.)
        check('adapter_counter_continuation',counters['command_encoder.weight']==float(parent_opt['state'][0]['step'])+400)
        parent_summary=dict(path=str(parent_path),sha256=sha256(parent_path),adapter_parent_step=100,additional_updates=400,base_new_step=400,
            moment_handover='Final state and metadata consistent; numeric equality of moments exactly at handover cannot be inferred from final moments. Trainer loads parent optimizer before adding base group; no saved handover-state artifact exists.')
    rng=payload['rng']
    torch_before=torch.get_rng_state();numpy_before=np.random.get_state()
    try:
        torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy'])
        check('rng_restore_roundtrip',exact(torch.get_rng_state(),rng['torch']) and exact(np.random.get_state(),rng['numpy']))
        t1=torch.rand(8,dtype=torch.float64);n1=np.random.random(8)
        torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy'])
        check('rng_draws_reproducible',torch.equal(t1,torch.rand(8,dtype=torch.float64)) and np.array_equal(n1,np.random.random(8)))
    finally:
        torch.set_rng_state(torch_before);np.random.set_state(numpy_before)
    if stage=='adapter':
        baseline=load_baseline_actor().eval()
        check('adapter_stage_all_base_tensors_unchanged',exact(model.base.state_dict(),baseline.state_dict()))
        reference=HERE.parent.parent/'banc-walking-recovery/20261110T152754Z/NATIVE_ROLE_2500/neural_decisions.npz'
        check('neutral_reference_in_data_identity',str(reference) in config['data_hashes'] and sha256(reference)==config['data_hashes'][str(reference)])
        with np.load(reference,allow_pickle=False) as z:
            x=torch.tensor(z['X238'],dtype=torch.float64)
            expected_mean=torch.tensor(z['mean'],dtype=torch.float64)
            expected_hidden=torch.tensor(z['hidden_after'],dtype=torch.float64)
        with torch.no_grad():
            mean,hidden=model.sequence(x[:,None],x.new_zeros((len(x),1,2)))
        check('adapter100_neutral_saved2880_means_bitexact',torch.equal(mean[:,0],expected_mean))
        check('adapter100_neutral_saved2880_hidden_bitexact',torch.equal(hidden[:,0],expected_hidden))
    check('checkpoint_unchanged_during_verification',sha256(checkpoint)==initial_digest)
    return dict(all_passed=all(checks.values()),checks=checks,checkpoint=str(checkpoint),checkpoint_sha256=initial_digest,
        stage=stage,global_step=total,parameter_groups=group_names,optimizer_counters=counters,parent=parent_summary,
        physical_steps=0,optimizer_updates=0,verifier_sha256=sha256(Path(__file__)),
        limitations=['Does not establish physical command response or behavior success.',
            'RNG restore is a saved-state roundtrip, not independent replay of full training.',
            'For finetune, exact optimizer moments at stage handover are not observable from final checkpoint; source/ancestor/counters/curve consistency only.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out).resolve()
    if out.exists():raise FileExistsError(out)
    try:report=verify(args.checkpoint)
    except Exception as exc:
        report=dict(all_passed=False,error=repr(exc),checkpoint=str(Path(args.checkpoint).resolve()),physical_steps=0)
        out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        raise
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(all_passed=report['all_passed'],stage=report['stage'],checks=len(report['checks']),out=str(out))))

if __name__=='__main__':main()
