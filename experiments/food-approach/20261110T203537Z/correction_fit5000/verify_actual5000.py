"""Independent full saved-trajectory/checkpoint audit; zero physics integration."""
from pathlib import Path
import argparse
import json
import types
import eval5000 as evaluation

HERE=evaluation.HERE

def derived_source():
    source=(evaluation.CONT/'verify_actual3000.py').read_text()
    pairs=[('from eval3000 import HERE,ROOT,BASE,ORIGINAL,COMMANDS,pin,jsonable,condition_score,checked_load',''),
           ("expected=np.r_[np.zeros((960,2)),np.tile(COMMANDS[result['condition']],(1920,1))]","expected=np.stack([command_at(result['condition'],k) for k in range(2880)])"),
           ("check('fixed3000_checkpoint',payload['curve'][-1]['step']==3000)","check('fixed5000_checkpoint',payload['curve'][-1]['step']==5000)"),
           ("('verify_actual3000.py','eval3000.py','EVAL_PREREG.json')","('verify_actual5000.py','eval5000.py','pipeline5000.py','EVAL_PREREG.json')")]
    for old,new in pairs:source=evaluation.replace_once(source,old,new)
    return source

def verify(run):
    run=Path(run).resolve();r=json.loads((run/'RESULT.json').read_text())
    if r['condition'] not in evaluation.COMMANDS or run!=HERE/('EVAL_'+r['condition']):raise ValueError('only five fixed outputs')
    evaluator=evaluation.derived_module();mod=types.ModuleType('_correction5000_saved_audit')
    mod.__dict__.update(__name__=mod.__name__,__file__=str(HERE/'verify_actual5000.py'),HERE=HERE,ROOT=evaluation.ROOT,BASE=evaluation.BASE,ORIGINAL=evaluator.ORIGINAL,COMMANDS=evaluation.COMMANDS,pin=evaluator.pin,jsonable=evaluator.jsonable,condition_score=evaluator.condition_score,checked_load=evaluation.checked_load,command_at=evaluation.command_at)
    exec(compile(derived_source(),str(HERE/'verify_actual5000.py')+'::derived','exec'),mod.__dict__)
    report=mod.verify(run)
    if r['condition']!='neutral':
        ident=json.loads((run/'IDENTITY.json').read_text());_,identity,fm=evaluator.pin()
        args=argparse.Namespace(checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256'],neutral_audit=r['neutral_reference']['audit'])
        fresh=evaluation.neutral_reference(r['neutral_reference']['path'],args,identity,fm,ident)
        if fresh!=r['neutral_reference']:raise ValueError('new neutral association drift')
        report['checks']['all_nonneutral_new_neutral_same_sha_source_identity']=True
    elif r['neutral_reference'] is not None:raise ValueError('neutral may not reuse a reference')
    report['saved_trajectory_verification']=True
    report['food_qualification']=False
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out).resolve()
    if not Path(a.run).is_absolute() or not Path(a.out).is_absolute() or not out.is_relative_to(HERE) or out.exists():raise ValueError('fresh absolute audit paths within correction_fit5000 required')
    try:report=verify(a.run)
    except BaseException as exc:
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(dict(integrity_all_pass=False,saved_trajectory_verification=False,error=repr(exc),physics_integration_steps=0),indent=2)+'\n');tmp.replace(out);raise
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    print(json.dumps(dict(integrity_all_pass=True,engineering_pass=report['engineering_pass'],checks=len(report['checks']),saved_trajectory_verification=True)))

if __name__=='__main__':main()
