"""Independent saved-array LEFT audit derived from immutable original full audit."""
from pathlib import Path
import argparse,json
import evaluate_motor as ev
HERE=ev.HERE
PRIMARY=ev.io.ROOT/'20261110T235830Z/evaluation/verify_actual.py'

def adapted_source():
 source=PRIMARY.read_text()
 source=ev.replace_once(source,"score(r['condition'],auto,cycles,delta)==r['criteria']","score(r['condition'],auto,cycles,delta,r['native'])==r['criteria']")
 old="""    if r.get('native') is not None:
        check('native_recorder_complete',r['native']['native_samples']==1232000 and r['native']['pending_rows']==0)
        for chunk in r['native']['chunks']:check('native_chunk_'+Path(chunk['path']).name,sha256(chunk['path'])==chunk['sha256'])"""
 new="""    native=r['native']
    check('every_native_step_complete',native['actual_native_steps']==1232000 and native['post_native_steps']==768000)
    check('native_extra_zero',native['extra_force_max']==0.)
    check('native_peak_nonnegative',native['nonfoot_peak_native']>=native['post_nonfoot_peak_native']>=0.)
    check('strict_native_flag_exact',r['criteria']['strict_native_nonfoot_zero']==(native['post_nonfoot_peak_native']==0. and native['post_native_steps']==768000 and native['actual_native_steps']==1232000 and native['extra_force_max']==0.))
    baseline=json.loads((ORIGINAL/'IDENTITY.json').read_text())
    check('unchanged_actual_physics',ident['physics']==baseline['physics'])
    check('unchanged_finite_actuator',ident['actuator']==baseline['actuator'])
    check('unchanged_all_compiled_arrays',ident['compiled_arrays']==baseline['compiled_arrays'])
    check('no_adhesion',ident['adhesion']['enabled'] is False)
    check('native12_5us',ident['physics']['timestep']==.0000125)
    check('49qpos48qvel',ident['physics']['nq']==49 and ident['physics']['nv']==48)
    check('42finite_PD',len(ident['actuator']['kp'])==42 and np.isfinite(ident['actuator']['torque_limit']).all() and np.min(ident['actuator']['torque_limit'])>0)
    check('strict_LEFT_HEAD_format',payload['format_version']=='strict_LEFT_HEAD_V1' and payload['curve'][-1]['step']==4000)
    check('LEFT_coupled_identity',payload['identity']['gate']=='relu(command[:,1:2])' and payload['identity']['residual_bound']==6. and payload['identity']['encoder_frozen'] and payload['identity']['gru_frozen'])
    import ast
    from evaluate_motor import PRIMARY as rollout_source
    rollout_ast=ast.parse(rollout_source.read_text())
    rollout=next(n for n in rollout_ast.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    direct_writes=[ast.unparse(n) for n in ast.walk(rollout) if isinstance(n,(ast.Assign,ast.AugAssign,ast.AnnAssign)) and any(x in ast.unparse(n).split('=')[0] for x in ('.qpos','.qvel'))]
    resets=[n for n in ast.walk(rollout) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('reset','reset_from')]
    check('rollout_source_root_writes_zero',len(direct_writes)==0 and len(resets)==0)
    physics_ast=ast.parse((BASE/'physics.py').read_text());core=next(n for n in physics_ast.body if isinstance(n,ast.ClassDef) and n.name=='Core');tick=next(n for n in core.body if isinstance(n,ast.FunctionDef) and n.name=='tick_rate')
    tick_writes=[n for n in ast.walk(tick) if isinstance(n,(ast.Assign,ast.AugAssign,ast.AnnAssign)) and any(x in ast.unparse(n).split('=')[0] for x in ('.qpos','.qvel'))]
    check('actual_native_tick_source_root_writes_zero',len(tick_writes)==0)"""
 return ev.replace_once(source,old,new)

def build():
 namespace={'__file__':str(PRIMARY),'__name__':'_strict_left_head_saved_actual_audit'}
 exec(compile(adapted_source(),str(PRIMARY)+'::STRICT_LEFT_HEAD_AUDIT','exec'),namespace)
 return namespace['verify']

def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=list(ev.COMMANDS),required=True);p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.parent_resource_released:raise ValueError('parent resource release required for saved production replay')
 out=HERE/('VERIFY_'+args.condition+'.json');run=HERE/('EVAL_'+args.condition)
 if out.exists():raise FileExistsError(out)
 try:
  result=build()(run);result['source_hashes'][str(PRIMARY)]=ev.sha256(PRIMARY);result['result_sha256']=ev.sha256(run/'RESULT.json')
 except BaseException as exc:
  out.write_text(json.dumps(dict(integrity_all_pass=False,error=repr(exc),physics_integration_steps=0),indent=2)+'\n');raise
 out.write_text(json.dumps(ev.jsonable(result),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(integrity_all_pass=result['integrity_all_pass'],engineering_pass=result['engineering_pass'],checks=len(result['checks']),out=str(out))),flush=True)
if __name__=='__main__':main()
