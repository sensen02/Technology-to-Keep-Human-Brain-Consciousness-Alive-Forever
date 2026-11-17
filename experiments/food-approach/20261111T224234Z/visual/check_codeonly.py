"""Small stdlib-only syntax/import/metadata/time checks. Never read trial arrays."""
from pathlib import Path
import ast,copy,json,runpy,sys
from contracts import HERE,CHECKS,MOTOR_SHA,BEHAVIOR_SHA,validate_metadata,summary_row,B_IDS,decision_at,events,save_new,sha

def main():
    paths=[HERE/n for n in ('contracts.py','render.py','plot_sync.py','check_codeonly.py')]
    checks={}
    for path in paths:
        source=path.read_text();tree=ast.parse(source);compile(tree,str(path),'exec')
        checks[path.name+'_syntax']=True
        top=[]
        for n in tree.body:
            if isinstance(n,(ast.Import,ast.ImportFrom)):top.extend([a.name for a in n.names] if isinstance(n,ast.Import) else [n.module])
        checks[path.name+'_top_imports_light']=not any(v and v.split('.')[0] in ('torch','numpy','mujoco','matplotlib','physics','runtime','model','model_io') for v in top)
    h=dict(all_exact=True,no_reset=True,time_s=5.8,state_dims=[128,64,64,32],motor_state_dims=[128,64,64])
    r=dict(kind='STIMULUS_AB_288_V1',condition='B_left_pulse',stage='B',state_layout=[128,64,64,32],state_total=288,checkpoint_sha256=MOTOR_SHA,behavior_checkpoint_sha256=BEHAVIOR_SHA,transition=h,engineering_pass=False,functional_pass=None,teacher_calls_after_handover=0,extra_forces=False,native=dict(actual_native_steps=1232000,post_native_steps=768000))
    a=dict(integrity_all_pass=True,checks={key:True for key in CHECKS})
    validate_metadata('B_left_pulse',r,a,h)
    summary=dict(source_kind='CONCENTRATION_CLAMP_NOT_NAVIGATION',food_success=False,status='B_COMPLETE_WITH_FAILED_FUNCTIONAL_TARGETS',all_functional_pass=False,rows=[dict(trial=c,functional_pass=False,functional_flags={'stable_body':True,'response':False},metrics={}) for c in B_IDS])
    selected=summary_row(summary,'B_left_pulse')
    checks['original5_FALSE_summaryFALSE_separate_immutable_PENDING']=selected['functional_pass'] is False and r['functional_pass'] is None and selected['trial']=='B_left_pulse'
    for name,mutation in (
        ('old256_rejected',lambda rr,aa,hh:rr.update(state_total=256)),
        ('missing288audit_rejected',lambda rr,aa,hh:aa['checks'].pop('full288_hidden_chain')),
        ('failedintegrity_rejected',lambda rr,aa,hh:aa.update(integrity_all_pass=False)),
        ('changed_sensor_rejected',lambda rr,aa,hh:rr.update(behavior_checkpoint_sha256='0'*64)),
        ('reset_rejected',lambda rr,aa,hh:hh.update(no_reset=False)),
        ('incomplete_steps_rejected',lambda rr,aa,hh:rr['native'].update(post_native_steps=1)),
    ):
        rr,aa,hh=copy.deepcopy((r,a,h));mutation(rr,aa,hh)
        try:validate_metadata('B_left_pulse',rr,aa,hh)
        except (ValueError,KeyError):checks[name]=True
        else:checks[name]=False
    checks['causal_overlay_no_future']=decision_at([1.,1.005,1.010],.999)==-1 and decision_at([1.,1.005,1.010],1.004)==0 and decision_at([1.,1.005,1.010],1.005)==1
    checks['events_start_end_explicit']=events('B_left_pulse')==[('ON_START',.6),('ON_RAMP_END',.8),('OFF_START',3.6),('OFF_RAMP_END',3.8)] and events('B_left_to_right')[-2:]==[('SWAP_START',4.8),('SWAP_RAMP_END',5.)] and not events('B_noodor')
    for name in ('render.py','plot_sync.py'):runpy.run_path(str(HERE/name),run_name='_codeonly_inspection')
    heavy=('torch','numpy','mujoco','physics','matplotlib','runtime','model','model_io')
    checks['all_heavy_imports_absent']=all(v not in sys.modules for v in heavy)
    result=dict(status='CODE_ONLY_CHECK_COMPLETE_NO_RUNTIME_OUTPUTS',all_passed=all(checks.values()),checks=checks,source_hashes={str(p):sha(p) for p in paths},heavy_imports={v:v in sys.modules for v in heavy},Core_constructions=0,model_loads=0,forward_calls=0,numpy_array_loads=0,physics_steps=0,render_frames=0,plot_calls=0,trial_result_reads=0)
    save_new(HERE/'CODEONLY_CHECK_B_SUMMARY.json',result);print(json.dumps(result,indent=2));assert all(checks.values())
if __name__=='__main__':main()
