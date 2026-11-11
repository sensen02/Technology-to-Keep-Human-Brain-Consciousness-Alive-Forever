"""Bounded meaningful CODEONLY checks. Never imports model, Core or physics backend."""
from pathlib import Path
import ast
import json
import sys
from contract import HERE, ROOT, OWN_SOURCE_NAMES, PINS, sha, load, pinned, qualification, require, save_new


def check():
    checks={}
    for name in OWN_SOURCE_NAMES:
        path=HERE/name
        require(path.is_file(),'preparation file missing: '+name)
        if path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
        else:load(path)
        checks['syntax_and_exists:'+name]=True
    for name in ('RUN_PLAN.template.json',):load(HERE/name);checks['draft_syntax:'+name]=True
    pinned();checks['frozen_native_recorder_and_methods_hashes']=True
    methods=load(HERE/'METHODS_DRAFT.json');timeline=methods['timeline']
    checks['full_native_count_and_phases']=timeline['settle_steps']+timeline['teacher_steps']+timeline['ANN_steps']==timeline['total_steps']==1232000
    checks['exact_native_and_logical_counts']=timeline['total_steps']*timeline['native_dt_s']==15.4 and timeline['logical_endpoints']==15400 and timeline['neural_decisions']==2880
    checks['bounded_chunk2048_602_last1152']=((1232000+2047)//2048==602 and 1232000-601*2048==1152 and methods['resources']['chunk_rows']==2048)
    def schedule(index):
        ms=index//80
        return ms,(0 if ms<1000 else 2 if ms<5800 else 3),(-1 if ms<1000 else (ms-1000)//5)
    expected={0:(0,0,-1),79999:(999,0,-1),80000:(1000,2,0),463999:(5799,2,959),464000:(5800,3,960),1231999:(15399,3,2879)}
    checks['all_native_source_handover_boundary_indexes']=all(schedule(index)==value for index,value in expected.items())
    checks['logical_state128_plus64']=timeline['state_layout']==[128,64]
    checks['current_record_only_D7_sufficient_proof']='CURRENT actual native record' in methods['D7'] and 'UNKNOWN' in methods['D7']
    checks['D6_signed_absolute_extra_sensitivity_separate']=all(term in methods['D6'] for term in ('signed p95','absolute','Extra-force-accounted','raw/.25/.5/1/2ms'))
    checks['native_signed_and_legacyXY_not_interchanged']='doNOT substitute' in methods['engineering_and_turn']
    checks['video_complete_bins_end15400']=len(list(range(19,15400,20)))==770 and list(range(19,15400,20))[-1]==15399
    execute=ast.parse((HERE/'execute_native.py').read_text());audit=ast.parse((HERE/'verify_native.py').read_text())
    audit_calls=[node for node in ast.walk(audit) if isinstance(node,ast.Call)]
    checks['independent_native_audit_has_no_Core_constructor_or_step_calls']=not any((isinstance(n.func,ast.Name) and n.func.id=='Core') or (isinstance(n.func,ast.Attribute) and n.func.attr in ('mj_step','mj_step1','mj_step2')) for n in audit_calls)
    checks['actual_entry_calls_admission_before_parent_import']=(HERE/'execute_native.py').read_text().index('plan, proof = admitted(')<(HERE/'execute_native.py').read_text().index('parent = parent_module()')
    checks['requires_explicit_selected_three_case_proof']=all(term in (HERE/'contract.py').read_text() for term in ("s['selected_all_three_pass'] is True","{'neutral','left','right'}","'REGISTERED_SELECTED_LEFT_NATIVE'"))
    # A nonexistent/unselected candidate must stop before any Torch/physics import.
    rejected=False
    try:qualification(ROOT/'FINAL_SUMMARY.json',ROOT/'NOT_SELECTED/candidate.pt','0'*64)
    except (ValueError,FileNotFoundError,TypeError,KeyError):rejected=True
    checks['missing_selected_candidate_fails_closed_before_model_import']=rejected
    checks['no_model_or_physics_backend_imported']=all(name not in sys.modules for name in ('torch','mujoco','physics','student_wave','flygym'))
    report=dict(all_passed=all(checks.values()),checks=checks,code_only=True,actual_native_ready=False,
                selected_candidate='NOT_ESTABLISHED_BY_PREPARATION',Core_constructions=0,physics_steps=0,optimizer_updates=0,
                preparation_source_hashes={str(HERE/name):sha(HERE/name) for name in OWN_SOURCE_NAMES},
                frozen_method_hashes={str(p):h for p,h in PINS.items()},
                limitation='AST/schedule/fail-closed and exact frozen-source checks only. No current selected model, compiled runtime, native data, D1-D8 result or video was executed/validated by CODEONLY preparation.')
    require(report['all_passed'],'native CODEONLY checks failed '+json.dumps(checks))
    return report


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out');args=parser.parse_args()
    report=check()
    if args.out:save_new(args.out,report)
    print(json.dumps(dict(all_passed=report['all_passed'],checks=len(report['checks']),actual_native_ready=False,physics_steps=0)))
if __name__=='__main__':main()
