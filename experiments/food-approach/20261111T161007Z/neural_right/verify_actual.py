"""Independent saved endpoints/actual neural replay; zero integration/optimizer steps."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import evaluate_motor as ev
HERE=ev.HERE
PRIMARY=HERE.parent.parent/'20261110T235830Z/evaluation/verify_actual.py'
def build():
 src=PRIMARY.read_text()
 src=ev.replace_once(src,"score(r['condition'],auto,cycles,delta)==r['criteria']","score(r['condition'],auto,cycles,delta,r['native'])==r['criteria']")
 old="""    if r.get('native') is not None:
        check('native_recorder_complete',r['native']['native_samples']==1232000 and r['native']['pending_rows']==0)
        for chunk in r['native']['chunks']:check('native_chunk_'+Path(chunk['path']).name,sha256(chunk['path'])==chunk['sha256'])"""
 new="""    native=r['native']
    check('every_native_step_recorder_complete',native['actual_native_steps']==1232000 and native['post_native_steps']==768000)
    check('native_extra_zero',native['extra_force_max']==0.)
    check('native_peak_nonnegative',native['nonfoot_peak_native']>=native['post_nonfoot_peak_native']>=0.)
    check('strict_native_flag_exact',r['criteria']['strict_native_nonfoot_zero']==(native['post_nonfoot_peak_native']==0.))
    baseline=json.loads((ORIGINAL/'IDENTITY.json').read_text())
    check('unchanged_actual_physics',ident['physics']==baseline['physics'])
    check('unchanged_finite_actuator',ident['actuator']==baseline['actuator'])
    check('unchanged_all_compiled_arrays',ident['compiled_arrays']==baseline['compiled_arrays'])
    check('no_adhesion',ident['adhesion']['enabled'] is False)
    check('native12_5us',ident['physics']['timestep']==.0000125)
    check('49qpos48qvel',ident['physics']['nq']==49 and ident['physics']['nv']==48)
    check('42finite_PD',len(ident['actuator']['kp'])==42 and np.isfinite(ident['actuator']['torque_limit']).all() and np.min(ident['actuator']['torque_limit'])>0)
    check('new_RIGHT_ONLY_format',payload['format_version']=='RIGHT_ONLY_V1' and payload['curve'][-1]['step']==2000)
    check('new_yaw006_identity','yaw/.06' in payload['identity']['command_domain'])
    import ast
    rollout_ast=ast.parse((HERE.parent.parent/'20261110T235830Z/evaluation/evaluate_motor.py').read_text())
    rollout=next(n for n in rollout_ast.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    direct_writes=[ast.unparse(n) for n in ast.walk(rollout) if isinstance(n,(ast.Assign,ast.AugAssign,ast.AnnAssign)) and any(x in ast.unparse(n).split('=')[0] for x in ('.qpos','.qvel'))]
    resets=[n for n in ast.walk(rollout) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('reset','reset_from')]
    check('rollout_source_root_writes_zero',len(direct_writes)==0 and len(resets)==0)
    physics_ast=ast.parse((BASE/'physics.py').read_text());core=next(n for n in physics_ast.body if isinstance(n,ast.ClassDef) and n.name=='Core');tick=next(n for n in core.body if isinstance(n,ast.FunctionDef) and n.name=='tick_rate')
    tick_writes=[n for n in ast.walk(tick) if isinstance(n,(ast.Assign,ast.AugAssign,ast.AnnAssign)) and any(x in ast.unparse(n).split('=')[0] for x in ('.qpos','.qvel'))]
    check('actual_native_tick_source_root_writes_zero',len(tick_writes)==0)"""
 src=ev.replace_once(src,old,new)
 # The import binds only this new evaluator; frozen source is separately hashed in readiness.
 namespace={'__name__':'_right_only_endpoint_audit','__file__':str(PRIMARY)};exec(compile(src,str(PRIMARY)+'::RIGHT_ONLY_explicit_audit','exec'),namespace)
 return namespace['verify']
def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=list(ev.COMMANDS),required=True);a=p.parse_args();out=HERE/('VERIFY_'+a.condition+'.json')
 if out.exists():raise ValueError('no overwrite independent audit')
 r=build()(HERE/('EVAL_'+a.condition));r['source_hashes'][str(PRIMARY)]=ev.sha256(PRIMARY)
 out.write_text(json.dumps(ev.jsonable(r),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(integrity_all_pass=r['integrity_all_pass'],engineering_pass=r['engineering_pass'],checks=len(r['checks']),out=str(out))),flush=True)
if __name__=='__main__':main()
