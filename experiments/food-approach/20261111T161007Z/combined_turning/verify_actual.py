"""Future independent composed-neural256 saved actual audit; formalFAIL remains formalFAIL."""
from pathlib import Path
import argparse,json
import evaluate_motor as ev
HERE=ev.HERE
PRIMARY=HERE.parent.parent/'20261110T235830Z/evaluation/verify_actual.py'

def replace_once(src,old,new):
 if src.count(old)!=1:raise ValueError('immutable saved-audit adaptation drift '+old)
 return src.replace(old,new)
def adapted_source():
 src=PRIMARY.read_text()
 for old,new in [("full2880_dualhidden_decisions","full2880_threehidden_decisions"),("full192_hidden_chain","full256_hidden_chain"),("actual2880_checkpoint_bothhidden_exact","actual2880_checkpoint_threehidden_exact")]:src=replace_once(src,old,new)
 if src.count('(2880,192)')!=2:raise ValueError('expected exactly two old hidden-schema dimensions')
 src=src.replace('(2880,192)','(2880,256)');src=replace_once(src,'np.zeros(192)','np.zeros(256)');src=replace_once(src,"hand['state_dims']==[128,64]","hand['state_dims']==[128,64,64]")
 src=replace_once(src,"score(r['condition'],auto,cycles,delta)==r['criteria']","score(r['condition'],auto,cycles,delta,r['native'])==r['criteria']")
 old="""    if r.get('native') is not None:
        check('native_recorder_complete',r['native']['native_samples']==1232000 and r['native']['pending_rows']==0)
        for chunk in r['native']['chunks']:check('native_chunk_'+Path(chunk['path']).name,sha256(chunk['path'])==chunk['sha256'])"""
 new="""    native=r['native']
    check('native_all1232000_and_ANN768000',native['actual_native_steps']==1232000 and native['post_native_steps']==768000)
    check('native_extra_zero',native['extra_force_max']==0.)
    check('native_NF_criterion_exact',r['criteria']['strict_native_nonfoot_zero']==(native['post_nonfoot_peak_native']==0.))
    baseline=json.loads((ORIGINAL/'IDENTITY.json').read_text())
    check('original_compiled_arrays_unchanged',ident['compiled_arrays']==baseline['compiled_arrays'])
    check('original_physics_unchanged',ident['physics']==baseline['physics'])
    check('original42finitePD_unchanged',ident['actuator']==baseline['actuator'])
    check('noadhesion',ident['adhesion']['enabled'] is False)
    check('native12_5us',ident['physics']['timestep']==.0000125)
    check('combined_new_constant_format',payload['format_version']=='COMBINED_TURN_V1' and payload['optimizer'] is None and payload['training_performed'] is False and 'curve' not in payload and 'training_updates' not in payload)
    check('all256_layout_bound',r['state_layout']==[128,64,64] and r['state_total']==256 and ident['config']['state_layout']==[128,64,64] and model.identity()['state_dims']==[128,64,64])
    check('all_composed_weights_frozen',all(not p.requires_grad for p in model.parameters()))
    from evaluate_motor import demonstration
    check('demonstration_scored_separately',r['demonstration_stable_motion']==demonstration(r['condition'],auto,cycles,delta,native))
    check('strict_engineering_NOT_relaxed',r['engineering_qualified']==r['engineering_pass']==r['criteria']['eligible'])
    import ast
    from evaluate_motor import PRIMARY as ROLLOUT_SOURCE
    tree=ast.parse(ROLLOUT_SOURCE.read_text());rollout=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    writes=[n for n in ast.walk(rollout) if isinstance(n,(ast.Assign,ast.AugAssign,ast.AnnAssign)) and any(key in ast.unparse(n).split('=')[0] for key in ('.qpos','.qvel'))]
    resets=[n for n in ast.walk(rollout) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('reset','reset_from')]
    check('actual_rollout_rootwrites_and_resets_zero',len(writes)==0 and len(resets)==0)"""
 return replace_once(src,old,new)

def build():
 ns={'__file__':str(PRIMARY),'__name__':'_composed_fullsaved_actual_audit256'};exec(compile(adapted_source(),str(PRIMARY)+'::COMBINED_TURN_V1_explicit256_audit','exec'),ns);return ns['verify']
def main():
 a=argparse.ArgumentParser();a.add_argument('--condition',choices=list(ev.COMMANDS),required=True);args=a.parse_args();out=HERE/('VERIFY_'+args.condition+'.json')
 if out.exists():raise ValueError('no overwrite composed actual audit')
 r=build()(HERE/('EVAL_'+args.condition));result=json.loads((HERE/('EVAL_'+args.condition)/'RESULT.json').read_text());r.update(demonstration_stable_motion=result['demonstration_stable_motion'],engineering_qualified=result['engineering_qualified'],state_layout=[128,64,64],state_total=256,result_sha256=ev.sha256(HERE/('EVAL_'+args.condition)/'RESULT.json'));r['source_hashes'][str(PRIMARY)]=ev.sha256(PRIMARY);out.write_text(json.dumps(ev.jsonable(r),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(integrity_all_pass=r['integrity_all_pass'],engineering_pass=r['engineering_pass'],demonstration_stable_motion=r['demonstration_stable_motion'],checks=len(r['checks']),state_total=256)),flush=True)
if __name__=='__main__':main()
