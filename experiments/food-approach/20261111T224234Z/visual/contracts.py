"""Exact STIMULUS_AB_288_V1 saved-output reader; stdlib-only until runtime call."""
from pathlib import Path
import hashlib,json,math
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
A_IDS=('A_left_half','A_right_half','A_switch')
B_IDS=('B_noodor','B_equal','B_left_pulse','B_right_pulse','B_left_to_right','B_right_to_left')
IDS=A_IDS+B_IDS
CHECKS=('full288_hidden_chain','actual2880_motor_outputs_exact','actual2880_behavior_outputs_exact','causal_receptor_filter_exact','actual_sensor_geometry_exact','metrics_cycles_recomputed')
MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
BEHAVIOR_SHA='bfd65a3896d259496114695d7f7f3550272cf4c369895a4ddb34f3ddc3352c06'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(ok,message):
    if not ok:raise ValueError(message)
def save_new(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def decision_at(times,now):
    """Latest already-issued query; no future observation or interpolation."""
    from bisect import bisect_right
    return bisect_right(times,now)-1

def events(condition):
    if condition not in IDS:raise ValueError('unregistered condition')
    if condition.startswith('A_'):return []
    if condition=='B_noodor':return []
    result=[('ON_START',.6),('ON_RAMP_END',.8)]
    result+= [('SWAP_START',4.8),('SWAP_RAMP_END',5.0)] if condition in B_IDS[-2:] else [('OFF_START',3.6),('OFF_RAMP_END',3.8)]
    return result

def validate_metadata(condition,result,audit,handover):
    require(condition in IDS,'unregistered condition')
    require(result['kind']=='STIMULUS_AB_288_V1','RESULT schema kind mismatch')
    require(result['condition']==condition,'RESULT condition mismatch')
    require(result['stage']==condition[0],'RESULT stage mismatch')
    require(audit['integrity_all_pass'] is True and bool(audit['checks']) and all(v is True for v in audit['checks'].values()),'incomplete independent audit')
    for key in CHECKS:require(audit['checks'].get(key) is True,'missing new288 audit: '+key)
    require(result['state_layout']==[128,64,64,32] and result['state_total']==288,'new288 layout required')
    require(result['checkpoint_sha256']==MOTOR_SHA and result['behavior_checkpoint_sha256']==BEHAVIOR_SHA,'frozen input identity mismatch')
    require(result['transition']==handover,'RESULT/HANDOVER mismatch')
    require(handover['state_dims']==[128,64,64,32] and handover['motor_state_dims']==[128,64,64],'HANDOVER full288 layout mismatch')
    require(handover['all_exact'] is True and handover['no_reset'] is True,'handover reset/continuity failure')
    require(math.isclose(float(handover['time_s']),5.8,rel_tol=0,abs_tol=1e-8),'handover clock mismatch')
    require(type(result['engineering_pass']) is bool,'engineering flag required; FALSE remains renderable')
    require(result['teacher_calls_after_handover']==0 and result['extra_forces'] is False,'hidden teacher or extra force')
    native=result['native']
    require(native['actual_native_steps']==1232000 and native['post_native_steps']==768000,'incomplete native step counts')


def audited_arrays(condition):
    """Runtime only: no model forward; saved arrays plus source/checkpoint/audit identity."""
    import numpy as np
    prereg=read(ROOT/'PREREG.json');require(prereg['kind']=='STIMULUS_AB_288_V1','wrong experiment prereg')
    require(prereg['A_order']==list(A_IDS) and prereg['B_order']==list(B_IDS),'condition order changed')
    run=ROOT/'trials'/condition
    r=read(run/'RESULT.json');a=read(run/'VERIFY.json');h=read(run/'HANDOVER.json');ident=read(run/'IDENTITY.json')
    validate_metadata(condition,r,a,h)
    require(Path(a['run']).resolve()==run,'audit path mismatch')
    require(a['condition']==condition,'audit condition mismatch')
    require(a['result_sha256']==sha(run/'RESULT.json'),'audit RESULT digest drift')
    require(a['checkpoint_sha256']==r['checkpoint_sha256'] and a['behavior_checkpoint_sha256']==r['behavior_checkpoint_sha256'],'audit policy identity mismatch')
    require(a['state_total']==288 and a['state_layout']==[128,64,64,32],'audit full288 layout mismatch')
    require(type(a['engineering_pass']) is bool and a['engineering_pass']==r['engineering_pass'],'audit original5 flag mismatch')
    require(r['identity_sha256']==ident['identity_sha256'],'actual identity binding mismatch')
    require(sha(r['checkpoint'])==MOTOR_SHA and sha(r['behavior_checkpoint'])==BEHAVIOR_SHA,'actual checkpoint bytes changed')
    paths=[run/n for n in ('RESULT.json','VERIFY.json','IDENTITY.json','HANDOVER.json','trace_1ms.npz','geometry_1ms.npz','neural_decisions.npz','runtime_geometry.npz')]
    paths += [ROOT/'PREREG.json',Path(r['checkpoint']),Path(r['behavior_checkpoint']),HERE/'contracts.py',HERE/'PREREG.json']
    hashes={str(p):sha(p) for p in paths}
    for ledger in (r['source_hashes'],a.get('source_hashes',{})):
        for path,digest in ledger.items():require(sha(path)==digest,'source changed: '+path);hashes[path]=digest
    for key,filename in [('trace','trace_1ms.npz'),('geometry','geometry_1ms.npz'),('neural_decisions','neural_decisions.npz'),('runtime_geometry','runtime_geometry.npz')]:require(sha(run/filename)==r[key]['sha256'],'artifact changed: '+filename)
    def arrays(name):
        with np.load(run/name,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    t=arrays('trace_1ms.npz');g=arrays('geometry_1ms.npz');d=arrays('neural_decisions.npz')
    for group in (t,g,d):require(all(np.isfinite(v).all() for v in group.values()),'nonfinite saved array')
    require(t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48),'full body trace required')
    require(np.max(np.abs(t['time']-np.arange(1,15401)*.001))<1e-8,'1ms trace clock mismatch')
    require(np.array_equal(g['time'],t['time']),'geometry clock mismatch')
    require(np.max(np.abs(d['time']-(1+np.arange(2880)*.005)))<1e-8,'5ms query clock mismatch')
    for stem,size in [('hidden',256),('behavior_hidden',32),('full_hidden',288)]:
        before=d[stem+'_before'];after=d[stem+'_after']
        require(before.shape==after.shape==(2880,size),'wrong '+stem+' shape')
        require(np.array_equal(before[1:],after[:-1]),'broken '+stem+' chain')
    for side in ('before','after'):require(np.array_equal(d['full_hidden_'+side],np.concatenate((d['hidden_'+side],d['behavior_hidden_'+side]),axis=1)),'full288 not motor256+behavior32')
    require(np.max(np.abs(d['behavior_hidden_before'][:960]))==0 and np.max(np.abs(d['behavior_hidden_after'][:960]))==0,'behavior not held during prime')
    if condition in A_IDS:require(np.max(np.abs(d['behavior_hidden_after']))==0,'A probe must not advance behavior')
    for key in ('command','raw_command','receptors','concentration','filter_before','filter_after','pending_before','pending_after'):require(d[key].shape==(2880,2),'wrong '+key+' shape')
    require(d['raw_turn'].shape==(2880,),'raw_turn must scalar per query')
    require(np.array_equal(t['control_source'],np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,3)]),'control source mismatch')
    require(np.array_equal(d['control_source'],np.r_[np.full(960,2),np.full(1920,3)]),'decision source mismatch')
    require(np.array_equal(t['action'],t['actual_requested_action']),'requested/applied actions differ')
    require(np.array_equal(t['action'][1000:],np.repeat(d['actually_applied_action'],5,axis=0)),'action hold differs from actual decisions')
    require(np.array_equal(d['policy_action'][960:],d['actually_applied_action'][960:]),'ANN applied output mismatch')
    for key in ('external_body_force_max','external_generalized_force_max'):require(np.max(np.abs(t[key]))==0,'extra physical forces')
    yaw=np.unwrap(g['thorax_yaw']);require(abs(float(yaw[-1]-yaw[5799])-r['yaw_change_rad'])<1e-12,'actual yaw metric mismatch')
    return dict(run=run,result=r,audit=a,identity=ident,trace=t,geometry=g,decisions=d,yaw=yaw-yaw[5799],source_hashes=hashes)

def summary_row(summary,condition):
    """Pure metadata binding; never promotes the immutable original RESULT."""
    require(summary['source_kind']=='CONCENTRATION_CLAMP_NOT_NAVIGATION' and summary['food_success'] is False,'summary scope mismatch')
    require(summary['status'] in ('B_COMPLETE_FUNCTIONAL_TARGETS_PASS','B_COMPLETE_WITH_FAILED_FUNCTIONAL_TARGETS'),'incomplete B summary')
    require([r['trial'] for r in summary['rows']]==list(B_IDS),'six fixed summary rows required')
    for row in summary['rows']:
        require(type(row['functional_pass']) is bool and bool(row['functional_flags']) and all(type(v) is bool for v in row['functional_flags'].values()),'invalid functional row flags')
        require(row['functional_pass']==all(row['functional_flags'].values()),'summary row decision differs from flags')
        require(isinstance(row['metrics'],dict),'summary metrics required')
    require(type(summary['all_functional_pass']) is bool and summary['all_functional_pass']==all(r['functional_pass'] for r in summary['rows']),'summary aggregate mismatch')
    return next(r for r in summary['rows'] if r['trial']==condition)

def bind_functional(data):
    condition=data['result']['condition']
    if condition in A_IDS:
        data['visual_functional_pass']=data['result']['functional_pass'];data['visual_functional_source']='immutable trial RESULT';return data
    path=ROOT/'B_FUNCTIONAL_SUMMARY.json'
    require(path.is_file(),'B formal visual delivery requires completed B_FUNCTIONAL_SUMMARY.json; original RESULT remains immutable')
    summary=read(path);row=summary_row(summary,condition);ledger=summary['source_hashes']
    for entry in summary['rows']:
        folder=ROOT/'trials'/entry['trial']
        for key,name in [('result','RESULT.json'),('audit','VERIFY.json'),('neural_trace','neural_decisions.npz'),('geometry_trace','geometry_1ms.npz')]:
            expected=folder/name
            require(Path(entry[key]).resolve()==expected,'summary trial artifact path mismatch')
            require(str(expected) in ledger and sha(expected)==ledger[str(expected)],'summary does not bind actual trial artifact: '+str(expected))
        rr=read(folder/'RESULT.json');aa=read(folder/'VERIFY.json')
        require(aa['integrity_all_pass'] is True and all(v is True for v in aa['checks'].values()),'summary references unaudited trial')
        require(aa['condition']==entry['trial'] and Path(aa['run']).resolve()==folder and aa['result_sha256']==ledger[str(folder/'RESULT.json')],'summary references mismatched trial audit')
        require(entry['original_engineering_pass']==rr['engineering_pass'] and entry['strict_endpoint_qualified']==rr['strict_endpoint_qualified'],'summary changed original engineering flags')
    for source,digest in ledger.items():require(Path(source).is_absolute() and sha(source)==digest,'summary source hash changed: '+source)
    data['source_hashes'].update(ledger);data['source_hashes'][str(path)]=sha(path)
    data['visual_functional_pass']=row['functional_pass'];data['visual_functional_flags']=row['functional_flags'];data['visual_functional_metrics']=row['metrics'];data['visual_functional_source']=str(path)
    return data

def verify_unchanged(hashes):
    for path,digest in hashes.items():require(sha(path)==digest,'input changed during visual task: '+path)
