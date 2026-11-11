"""Prepare offline, or explicitly run the five final5000 conditions in fixed stages."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse
import ast
import copy
import json
import os
import subprocess
import sys
import time
import numpy as np
import eval5000 as e
import verify_actual5000 as v

HERE=e.HERE
PYTHON=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python')

def dump(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def prepare():
    prereg=e.check_sources();checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise ValueError('offline check failed '+name)
    for name in e.SOURCE_NAMES[:3]:ast.parse((HERE/name).read_text(),filename=str(HERE/name))
    check('syntax_all_sources',True)
    original=ast.parse((e.CONT/'eval3000.py').read_text());derived=ast.parse(e.derived_source())
    functions=lambda tree:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    of,df=functions(original),functions(derived)
    for name in ('jsonable','pin','condition_score','checked_load','source_paths','neutral_reference','parser'):
        check('original_function_AST_unchanged_'+name,ast.dump(of[name])==ast.dump(df[name]))
    assignments={n.targets[0].id:n for n in ast.walk(of['evaluate']) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('command','ref')}
    class Normalize(ast.NodeTransformer):
        def visit_Assign(self,node):
            node=self.generic_visit(node)
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in assignments:return copy.deepcopy(assignments[node.targets[0].id])
            return node
        def visit_Constant(self,node):
            if node.value==5000:return ast.copy_location(ast.Constant(3000),node)
            replacements={'fixed5000 corrective-fit matched1s smoothstep actual evaluation':'fixed3000 continued fast neural motor actual evaluation'}
            if isinstance(node.value,str) and node.value in replacements:node.value=replacements[node.value]
            return node
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='dict':node.keywords=[k for k in node.keywords if k.arg!='continuation_metadata']
            return node
    check('original_control_AST_unchanged_except_declared_adapters',ast.dump(Normalize().visit(copy.deepcopy(df['evaluate'])))==ast.dump(of['evaluate']))
    ast.parse(v.derived_source());check('derived_saved_verifier_syntax',True)
    check('fixed_checkpoint_only',e.CHECKPOINT==HERE/'TRAIN/checkpoint_005000.pt')
    check('fixed_five_conditions',list(e.COMMANDS)==['neutral','left','right','speed085','speed090'] and prereg['conditions']==e.COMMANDS and prereg['checkpoint_step']==5000)
    teacher_root=e.ROOT.parent/'20261110T191449Z'
    teacher_paths={'left':teacher_root/'teacher/left_full/SAMPLES_5MS.npz','right':teacher_root/'teacher/right_full/SAMPLES_5MS.npz','speed085':teacher_root/'deceleration_moderate/straight_085/SAMPLES_5MS.npz','speed090':teacher_root/'deceleration_moderate/straight_090/SAMPLES_5MS.npz'}
    for condition in e.COMMANDS:
        commands=np.stack([e.command_at(condition,k) for k in range(2880)])
        check(condition+'_prime_and_first_ANN_zero',np.array_equal(commands[:961],np.zeros((961,2))))
        if condition=='neutral':check('neutral_all_zero',np.array_equal(commands,np.zeros((2880,2))));continue
        check(condition+'_exact_old_ramp_all2880',np.array_equal(commands,np.stack([e.old_ramp_cached.command_at(condition,k) for k in range(2880)])))
        with np.load(teacher_paths[condition],allow_pickle=False) as z:actual=z['command2'].copy()
        check(condition+'_training_teacher_ramp_matches',actual.shape==(2880,2) and np.max(np.abs(actual-commands))<=1e-14)
    namespace={'np':np};exec(compile(ast.Module(body=[copy.deepcopy(of['condition_score'])],type_ignores=[]),'offline_score','exec'),namespace)
    score=namespace['condition_score'];auto={'nonfoot_force_max_native':0.,'all_gates':True,'forward_signed_native_per_s':.095};cycles={'all_legs_completed':True}
    check('eligible_left_signed_threshold',score('left',auto,cycles,.01)['eligible'] and not score('left',auto,cycles,.009)['eligible'])
    check('eligible_right_signed_threshold',score('right',auto,cycles,-.01)['eligible'] and not score('right',auto,cycles,.01)['eligible'])
    check('eligible_speed_matched_neutral_threshold',score('speed085',auto,cycles,0.,.1)['eligible'] and not score('speed085',dict(auto,forward_signed_native_per_s=.09501),cycles,0.,.1)['eligible'])
    for change in (dict(auto,all_gates=False),dict(auto,nonfoot_force_max_native=1e-20)):
        check('gate_failure_'+str(len(checks)),not score('neutral',change,cycles,0.)['eligible'])
    check('all_six_required',not score('neutral',auto,{'all_legs_completed':False},0.)['eligible'])
    check('no_physics_or_model_imported',all(n not in sys.modules for n in ('torch','mujoco','physics','student_wave','fast_motor')))
    return dict(status='READY_CODE_ONLY_NO_PHYSICS',checks=checks,all_passed=all(checks.values()),source_hashes={str(p):e.sha256(p) for p in e.source_paths()},physics_integration_steps=0,model_constructions=0,training_updates=0,launch_guard='--training-complete --sha256 FINAL5000_SHA',checkpoint=str(e.CHECKPOINT),stages=[['neutral'],['left','right'],['speed085','speed090']],max_processes=2,cpu_threads_per_process=1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--training-complete',action='store_true');p.add_argument('--sha256');a=p.parse_args()
    ready=prepare()
    if a.prepare_only:
        if a.training_complete or a.sha256:raise ValueError('prepare-only cannot combine launch inputs')
        print(json.dumps(ready,indent=2));return
    if not a.training_complete or not a.sha256:raise ValueError('actual execution requires later parent authorization --training-complete --sha256 SHA')
    if len(a.sha256)!=64 or any(c not in '0123456789abcdef' for c in a.sha256) or e.sha256(e.CHECKPOINT)!=a.sha256:raise ValueError('final5000 SHA mismatch')
    summary=HERE/'FINAL5000_SUMMARY.json'
    expected=[summary,*[HERE/('EVAL_'+c) for c in e.COMMANDS],*[HERE/('VERIFY_'+c+'.json') for c in e.COMMANDS],*[HERE/(prefix+c+'.log') for c in e.COMMANDS for prefix in ('EVAL_','VERIFY_')]]
    if any(path.exists() for path in expected):raise FileExistsError('fresh outputs required; partial failures preserved, no overwrite or automatic retry')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    started=time.time();results={};status='RUNNING'
    def persist():
        complete=len(results)==5
        dump(summary,dict(status=status,checkpoint=str(e.CHECKPOINT),checkpoint_sha256=a.sha256,conditions=results,all_conditions_eligible=bool(complete and all(r.get('engineering_pass',False) for r in results.values())),saved_trajectory_verification=bool(complete and all(r.get('integrity_all_pass',False) for r in results.values())),readiness=ready,elapsed_s=time.time()-started,food_success=False))
    def command(label,argv):
        with (HERE/(label+'.log')).open('x') as f:run=subprocess.run([str(PYTHON),'-B',*map(str,argv)],cwd=str(HERE),env=env,stdout=f,stderr=subprocess.STDOUT)
        return dict(returncode=run.returncode,argv=list(map(str,argv)),log=str(HERE/(label+'.log')))
    def case(c):
        record=dict(engineering_pass=False,integrity_all_pass=False)
        try:
            out=HERE/('EVAL_'+c);audit=HERE/('VERIFY_'+c+'.json')
            argv=[HERE/'eval5000.py','--checkpoint',e.CHECKPOINT,'--sha256',a.sha256,'--condition',c,'--out',out,'--training-complete']
            if c!='neutral':argv+=['--neutral-reference',HERE/'EVAL_neutral/RESULT.json','--neutral-audit',HERE/'VERIFY_neutral.json']
            record['execution']=command('EVAL_'+c,argv)
            if record['execution']['returncode']!=0:return record
            record['verification']=command('VERIFY_'+c,[HERE/'verify_actual5000.py','--run',out,'--out',audit])
            r=json.loads((out/'RESULT.json').read_text());record.update(result=str(out/'RESULT.json'),result_sha256=e.sha256(out/'RESULT.json'),criteria=r['criteria'],autonomous=r['autonomous'],yaw_change_rad=r['yaw_change_rad'],engineering_pass=bool(r['engineering_pass']))
            if record['verification']['returncode']==0:
                report=json.loads(audit.read_text());record.update(integrity_all_pass=bool(report['integrity_all_pass']),audit=str(audit),audit_sha256=e.sha256(audit))
        except BaseException as exc:record.update(error=repr(exc),integrity_all_pass=False)
        return record
    persist()
    try:
        results['neutral']=case('neutral');persist()
        if not results['neutral']['engineering_pass'] or not results['neutral']['integrity_all_pass']:
            status='STOPPED_NEUTRAL_NOT_QUALIFIED';persist();raise SystemExit(1)
        for stage in (('left','right'),('speed085','speed090')):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures={pool.submit(case,c):c for c in stage}
                for future in as_completed(futures):
                    c=futures[future];results[c]=future.result();persist()
                    print(json.dumps(dict(condition=c,engineering_pass=results[c]['engineering_pass'],integrity_all_pass=results[c]['integrity_all_pass'])),flush=True)
        for path,digest in ready['source_hashes'].items():
            if e.sha256(path)!=digest:raise ValueError('source changed during execution '+path)
        if e.sha256(e.CHECKPOINT)!=a.sha256:raise ValueError('checkpoint changed during execution')
        status='COMPLETE' if all(r['integrity_all_pass'] for r in results.values()) else 'COMPLETE_WITH_AUDIT_OR_EXECUTION_FAILURE';persist()
        if status!='COMPLETE':raise SystemExit(1)
    except BaseException as exc:
        if status=='RUNNING':status='INTERRUPTED_OR_FAILED'
        persist();raise

if __name__=='__main__':main()
