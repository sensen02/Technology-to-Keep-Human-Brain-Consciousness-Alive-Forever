"""Independent saved evaluation verifier. No episode/Actor/restore or physics imports.
Run only after explicit parent readiness; default invocation refuses execution.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype')
ROUND=PROJECT/'experiments/banc-physical-learning/20261107T195435Z'
FOUND=ROUND/'foundation'
EVAL=Path('/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning/evaluation')
PRIOR=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z/closed_loop')
CORE=PROJECT/'experiments/banc-local-learning/20261107T192931Z/dynamics/module.py'
ENCODER=PROJECT/'experiments/banc-local-learning/20261107T185910Z/encoder/encoder.py'
CORE_SHA='c5106d46a62bedd31faccdbe981764106c23c4dc491890e6e4d21ef531718250'
EPISODE_SHA='f81f2228652fc7bf5f11ebef796f37ddf8c968ce1205202e12506adea6a7474a'
NET_ARRAYS=('v','ge','gi','g_adapt','refractory_until_ms','spike_count','_ring','last_spikes')
ID_FIELDS=('root_ids','cache_indices','allowed_sensory_local','source_sensory_local','excluded_ambiguous_sensory_local','descending_local','ascending_local','mn_local','mn_leg','strict_source_edge_rows','_sensory_mask','_descending_mask')
LEGS=('LF','LH','LM','RF','RH','RM')
W=24.446354755878005
R={'status':'RUNNING','completed_neural_steps':0,'physical_steps':0,'scratch_forward_calls':0,'optimizer_updates':0,'conditions':{},'failures':[]}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def imported(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def exact(label,x,y):
    if not np.array_equal(x,y):
        delta=float(np.max(np.abs(np.asarray(x,dtype=float)-np.asarray(y,dtype=float)))) if np.shape(x)==np.shape(y) and np.asarray(x).dtype.kind in 'biuf' else None
        raise AssertionError(f'{label}: not exact; maxdifference={delta}')

def compare_numbers(label,a,b,tol=1e-10):
    if isinstance(a,dict):
        for k,v in a.items():compare_numbers(label+'.'+k,v,b[k],tol)
    elif isinstance(a,(list,tuple)):compare_numbers(label,np.asarray(a),np.asarray(b),tol)
    elif isinstance(a,(str,bool)) or a is None:
        if a!=b:raise AssertionError(label+': value differs')
    else:
        if np.shape(a)!=np.shape(b) or not np.allclose(a,b,rtol=0,atol=tol,equal_nan=False):raise AssertionError(label+': numeric differs')

def physical_gate(native,assist):
    t=native[:,0];foot=native[:,1:7];nf=native[:,8];total=foot.sum(1)+nf;support=foot.sum(1)/W
    acc=np.convolve(np.gradient(np.gradient(native[:,10],t),t),np.ones(9)/9,mode='same')
    share=np.divide(nf,total,out=np.zeros_like(nf),where=np.abs(total)>1e-9)
    residual=np.abs((W/9801.)*acc-(total-W))/W
    r={'foot_support_mean':float(support.mean()),'foot_support_p10':float(np.percentile(support,10)),'foot_support_min':float(support.min()),'nonfoot_share_mean':float(np.nanmean(share)),'ground_support_mean':float((total/W).mean()),'com_z_min':float(native[:,10].min()),'com_z_mean':float(native[:,10].mean()),'free_fall_fraction':float(np.mean((total/W<.2)&(acc<=-.5*9801.))),'force_balance_residual_p95':float(np.percentile(residual,95)),'force_balance_residual_p50':float(np.percentile(residual,50)),'bearing_legs_median':float(np.median((foot>1e-9).sum(1))),'bearing_legs_min':int((foot>1e-9).sum(1).min()),'per_leg_mean_load':[float(foot[:,i].mean()/W) for i in range(6)],'per_leg_duty':[float((foot[:,i]>1e-9).mean()) for i in range(6)],'samples':len(t)}
    checks={'D1_no_nonfoot_ground_force':bool(native[:,7].max()==0),'D2_no_free_fall':r['free_fall_fraction']==0,'D3_foot_support_not_below_assist':r['foot_support_mean']>=assist['foot_support_mean']-.1,'D4_foot_support_never_collapses':r['foot_support_p10']>=.5,'D4b_nonfoot_is_not_carrying_the_body':r['nonfoot_share_mean']<=.2,'D5_com_height_not_below_assist':r['com_z_min']>=assist['com_z_mean']-.1,'D6_force_balance_consistent':r['force_balance_residual_p95']<=.15,'D7_tripod_redundancy':r['bearing_legs_median']>=3,'D8_still_above_absolute_floor':r['com_z_min']>=.8*1.2072528224791905}
    return {'checks':checks,'passed':bool(all(checks.values())),'release':r}

def score_arrays(native,slip,sc):
    duration=len(native)*.0000125;heading=sc['heading'];delta=sc['end_com']-sc['start_com'];loads=sc['loads5ms'];rates=sc['rates5ms']
    assert heading.shape==(2,) and abs(np.linalg.norm(heading)-1)<1e-12
    completed=[]
    for leg in range(6):
        labels=np.where(loads[:,leg]>.05*W,1,np.where(loads[:,leg]<.005*W,-1,0));runs=[];begin=0
        while begin<len(labels):
            end=begin+1
            while end<len(labels) and labels[end]==labels[begin]:end+=1
            if labels[begin]!=0 and end-begin>=4:runs.append(int(labels[begin]))
            begin=end
        stage=0
        for state in runs:
            if state==(1,-1,1)[min(stage,2)]:stage+=1
            if stage==3:break
        completed.append(stage==3)
    terms={'forward_displacement':float(delta[:2]@heading),'weighted_slip_distance':float(slip[:,7:13].sum()/W*.0005),'lateral_displacement':float(delta[:2]@np.array([-heading[1],heading[0]])),'height_drop_excess':max(0.,float(sc['start_com'][2])-min(float(sc['start_com'][2]),float(native[:,10].min()))-.05),'positive_nonfoot_impulse_overW':float(native[:,7].sum()/W*.0000125),'support_deficit_integral':float(np.maximum(0.,.6-native[:,1:7].sum(1)/W).sum()*.0000125),'rate_squared_time':float(np.mean(rates*rates)*duration),'completed_cycle_legs':sum(completed)}
    reward=terms['forward_displacement']-2*terms['weighted_slip_distance']-.5*abs(terms['lateral_displacement'])-5*terms['height_drop_excess']-5*terms['positive_nonfoot_impulse_overW']-5*terms['support_deficit_integral']-.0005*terms['rate_squared_time']+.03*terms['completed_cycle_legs']
    return terms,reward,completed

def capture(c):
    z={k:getattr(c.net,k).copy() for k in NET_ARRAYS};z.update({k:getattr(c,k).copy() for k in ID_FIELDS})
    z.update(filtered_rates_hz=c.filtered_rates_hz.copy(),_ring_i=np.array(c.net._ring_i,dtype=np.int64),steps=np.array(c.net.steps,dtype=np.int64),t_ms=np.array(c.net.t_ms,dtype=float),neural_module_sha256=np.array(CORE_SHA),parameter_contract_json=np.array(json.dumps({'point':asdict(c.net.p),'adaptation':asdict(c.net.adaptation_params),'dt_ms':c.net.dt,'delay_steps':c.net.delay_steps,'filter_tau_ms':c.filter_tau_ms,'total_conductance_uS':c.total_conductance_uS},sort_keys=True,separators=(',',':'))))
    return z

def run_condition(name,core,encoder,base,canonical,prior_native):
    folder=EVAL/name
    paths=[folder/n for n in ['POLICY.npz','BODY.npz','NATIVE.npz','COMMANDS.npz','ACTOR.npz','INITIAL.npz','NEURAL_INITIAL.npz','NEURAL_FINAL.npz','PROTOCOL.json','RESULT.json','SLIP.npz','SCORE_INPUTS.npz']]
    hashes={str(p):sha(p) for p in paths}
    result=json.loads((folder/'RESULT.json').read_text());protocol=json.loads((folder/'PROTOCOL.json').read_text())
    assert result['valid'] and result['native_steps']==400000 and result['neural_steps']==10000 and result['release_s']==5.
    assert protocol['final_validation'] and protocol['duration_s']==5. and protocol['policy_dt']==.005 and protocol['physics_dt']==.0000125
    assert protocol['run_source_sha256']==EPISODE_SHA
    for source,digest in protocol['source_hashes'].items():assert sha(source)==digest,source
    policy=load(folder/'POLICY.npz');body=load(folder/'BODY.npz');native=load(folder/'NATIVE.npz')['rows'];commands=load(folder/'COMMANDS.npz')['rows'];initial=load(folder/'INITIAL.npz');actor=load(folder/'ACTOR.npz');saved_start=load(folder/'NEURAL_INITIAL.npz');saved_final=load(folder/'NEURAL_FINAL.npz')
    assert len(policy['time_s'])==1000 and len(body['time'])==5001 and native.shape==(400000,20) and commands.shape==(5000,85)
    assert abs(body['time'][0]-1)<1e-8 and not policy['teacher_warmup'].any()
    assert set(saved_start)==set(canonical)
    for k in canonical:exact('canonical/'+k,saved_start[k],canonical[k])
    c=core.LocalCircuit()
    for k in ID_FIELDS:exact('identity/'+k,getattr(c,k),canonical[k])
    # Manual array assignment, independent from foundation runtime.restore_circuit.
    for k in NET_ARRAYS:setattr(c.net,k,saved_start[k].copy())
    c.net._ring_i=int(saved_start['_ring_i']);c.net.steps=int(saved_start['steps']);c.net.t_ms=float(saved_start['t_ms']);c.filtered_rates_hz=saved_start['filtered_rates_hz'].copy()
    assert c.net.steps==2000 and c.net.t_ms==1000
    for k,v in capture(c).items():exact('manual_restore/'+k,v,saved_start[k])
    exact('encoder_roots',encoder.root_ids,c.root_ids);assert np.isin(encoder.local_indices,c.allowed_sensory_local).all()
    for k,v in [('mn_local',c.mn_local),('mn_root_ids',c.root_ids[c.mn_local]),('mn_leg',c.mn_leg),('joint_names',np.asarray(encoder.joint_names))]:exact('actoridentity/'+k,actor[k],v);exact('basisidentity/'+k,base[k],v)
    assert str(actor['basis_sha256'])==sha(FOUND/'BASE.npz')
    theta=actor['theta'];assert theta.shape==(6,4,7) and np.isfinite(theta).all() and np.max(np.abs(theta))<=.25
    weights=np.zeros((323,42))
    for i,leg in enumerate(LEGS):
        ni=np.flatnonzero(c.mn_leg==leg.lower());ji=np.flatnonzero(base['joint_leg']==leg)
        weights[np.ix_(ni,ji)]=np.clip(base['basis'][ni,i*4:i*4+4]@theta[i],-1,1)
    exact('manual_weights',weights,actor['weights']);assert np.max(np.abs(weights))<=1
    assert not weights[c.mn_leg[:,None]!=np.char.lower(base['joint_leg'])[None,:]].any()
    if name=='zero':assert not theta.any() and not weights.any()
    foot0=prior_native[79999,1:7]
    exact('first_restored_contact_vs_saved_handover',policy['local_physical_input'][0,84:90],foot0)
    mins=[];maxs=[];gmax=[];allsp=[];mnsp=[];binmins=[];binmaxs=[];zero=np.zeros(20000)
    for tick in range(1000):
        k=tick*5
        exact('localq',policy['local_physical_input'][tick,:42],body['qpos'][k,initial['qa']]);exact('localv',policy['local_physical_input'][tick,42:84],body['qvel'][k,initial['va']])
        foot=foot0 if tick==0 else native[tick*400-1,1:7]
        exact('localfoot',policy['local_physical_input'][tick,84:90],foot);exact('policytime',policy['time_s'][tick],body['time'][k])
        mn=c.filtered_rates_hz[c.mn_local];exact('MN_rates',policy['mn_rates_hz'][tick],mn);exact('MN_voltage',policy['mn_voltages_mV'][tick],c.net.v[c.mn_local])
        action=30*np.tanh((mn/100)@weights);exact('actor_action',policy['predicted_rate'][tick],action)
        exact('held_commands',commands[k:k+5,1:43],np.broadcast_to(action,(5,42)));exact('commandtimes',commands[k:k+5,0],body['time'][k:k+5])
        current=encoder.encode(policy['local_physical_input'][tick,:42],policy['local_physical_input'][tick,42:84],foot)
        exact('encoder_current',current[encoder.local_indices],policy['sensory_current_nA'][tick]);assert not current[~encoder.allowed_mask].any() and not zero.any()
        for _ in range(10):
            before=c.net.steps
            try:c.advance(current,zero,.5)
            finally:R['completed_neural_steps']+=c.net.steps-before
            mins.append(float(c.net.v.min()));maxs.append(float(c.net.v.max()));gmax.append(float(c.net.g_adapt.max()));allsp.append(int(c.net.last_spikes.sum()));mnsp.append(int(c.net.last_spikes[c.mn_local].sum()))
        binmins.append(mins[-1]);binmaxs.append(maxs[-1])
    final=capture(c);assert set(final)==set(saved_final)
    for k,v in final.items():exact('final/'+k,v,saved_final[k])
    exact('sampled_voltage_summary',result['neural_voltage_range_mV'],[min(binmins),max(binmaxs)])
    assert c.net.steps==12000 and c.net.t_ms==6000 and sum(mnsp)==result['MN_spikes_since_checkpoint']
    assert min(mins)>=-80-1e-10 and max(maxs)<=-50+1e-10
    if name=='zero':assert not policy['predicted_rate'].any() and not commands[:,1:43].any()
    assist_rows=prior_native[(prior_native[:,0]>=.8)&(prior_native[:,0]<1)]
    assist={'foot_support_mean':float(assist_rows[:,1:7].sum(1).mean()/W),'com_z_mean':float(assist_rows[:,10].mean())}
    compare_numbers('assist',assist,protocol['assist'])
    gate=physical_gate(native,assist)
    compare_numbers('physicalgate',gate,result['physical_gate']);assert bool(result['feasible'])==gate['passed']
    sc=load(folder/'SCORE_INPUTS.npz');slip=load(folder/'SLIP.npz')['rows']
    exact('score_loads',sc['loads5ms'],native[np.arange(1,1001)*400-1,1:7]);exact('score_rates',sc['rates5ms'],policy['predicted_rate']);exact('score_startcom',sc['start_com'],body['com'][0]);exact('score_endcom',sc['end_com'],body['com'][-1]);exact('scoreheading',sc['heading'],protocol['initial_heading_xy'])
    terms,reward,cycles=score_arrays(native,slip,sc);compare_numbers('rewardterms',terms,result['reward_terms']);compare_numbers('reward',reward,result['reward']);exact('cycles',cycles,result['cycles']['completed_by_leg'])
    for p,h in hashes.items():assert sha(p)==h
    np.savez_compressed(HERE/(name+'_NEURAL_REPLAY_FINAL.npz'),**final)
    np.savez_compressed(HERE/(name+'_EVERY_STEP.npz'),neural_time_ms=1000+(np.arange(10000)+1)*.5,voltage_min_mV=mins,voltage_max_mV=maxs,g_adapt_max_uS=gmax,all_spike_count_step=allsp,mn_spike_count_step=mnsp)
    report={'status':'PASS','neural_steps':10000,'physical_steps':0,'scratch_forward_calls':0,'all_final_state_exact':True,'all_policy_MN_voltage_rates_encoder_actions_commands_exact':True,'source_initial_contact_exact':True,'voltage_min_every_step_mV':min(mins),'voltage_max_every_step_mV':max(maxs),'stored_5ms_min_mV':min(binmins),'MN_spikes_since_checkpoint':sum(mnsp),'zero_theta':bool(not theta.any()),'zero_weights':bool(not weights.any()),'physical_gate_pass':gate['passed'],'physical_gate_checks':gate['checks'],'reward_recomputed':reward,'completed_cycle_legs':sum(cycles),'input_hashes':hashes,'interpretation':'neural and saved-array scoring replay only; parent verifies physics; physical gate does not establish walking'}
    R['conditions'][name]=report;print(json.dumps({'condition':name,**{k:v for k,v in report.items() if k!='input_hashes'}},indent=2),flush=True)
    return actor,policy,body

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent-artifacts-ready',action='store_true');args=parser.parse_args()
    if not args.parent_artifacts_ready:raise SystemExit('Refuse execution until explicit parent artifacts-ready notice.')
    if (HERE/'VERIFICATION.json').exists():raise RuntimeError('preserve previous verification result')
    assert sha(CORE)==CORE_SHA and sha(ROUND/'learning/episode.py')==EPISODE_SHA
    R['source_hashes']={str(p):sha(p) for p in [CORE,ENCODER,FOUND/'BASE.npz',FOUND/'NEURAL_START.npz',HERE/'PREREGISTRATION.json',ROUND/'learning/episode.py']}
    core=imported('evaluation_fixed_core',CORE);encoder=imported('evaluation_fixed_encoder',ENCODER).SensoryEncoder();base=load(FOUND/'BASE.npz');canonical=load(FOUND/'NEURAL_START.npz');prior_native=load(PRIOR/'NATIVE.npz')['rows']
    first=run_condition('selected',core,encoder,base,canonical,prior_native)
    second=run_condition('zero',core,encoder,base,canonical,prior_native)
    R['selected_equals_zero']={ 'theta':bool(np.array_equal(first[0]['theta'],second[0]['theta'])),'weights':bool(np.array_equal(first[0]['weights'],second[0]['weights'])),'policy_all_arrays':bool(set(first[1])==set(second[1]) and all(np.array_equal(v,second[1][k]) for k,v in first[1].items())),'body_all_arrays':bool(set(first[2])==set(second[2]) and all(np.array_equal(v,second[2][k]) for k,v in first[2].items()))}
    for p,h in R['source_hashes'].items():assert sha(p)==h
    assert R['completed_neural_steps']==20000
    R['status']='PASS_INDEPENDENT_SELECTED_ZERO_REPLAY'

if __name__=='__main__':
    try:main()
    except SystemExit:raise
    except Exception as exc:
        R['status']='FAIL_PRESERVED';R['failures'].append({'error':repr(exc),'traceback':traceback.format_exc()});raise
    finally:
        if R['status']!='RUNNING':(HERE/'VERIFICATION.json').write_text(json.dumps(R,indent=2,allow_nan=False)+'\n')
