"""Independent saved ramp diagnostic audit, derived from immutable3000 verifier.
No Core/model physics construction, no integration. Load each saved archive once.
"""
from pathlib import Path
import argparse
import json
import types
import eval_ramp as ramp

HERE=ramp.HERE


def derived_source():
    source=(ramp.CONT/'verify_actual3000.py').read_text()
    pairs=[
      ('from eval3000 import HERE,ROOT,BASE,ORIGINAL,COMMANDS,pin,jsonable,condition_score,checked_load',''),
      ("expected=np.r_[np.zeros((960,2)),np.tile(COMMANDS[result['condition']],(1920,1))]","expected=np.stack([command_at(result['condition'],k) for k in range(2880)])"),
      ("nr['source_hashes']==result['source_hashes'] and ni['compiled_arrays']==ident['compiled_arrays']","nr['source_hashes']==ref['original_source_hashes'] and ni['compiled_arrays']==ident['compiled_arrays']"),
      ("('verify_actual3000.py','eval3000.py','EVAL_PREREG.json')","('verify_ramp.py','eval_ramp.py','EVAL_PREREG.json')"),
    ]
    for old,new in pairs:
        if source.count(old)!=1:raise ValueError('original verifier shape drift '+old)
        source=source.replace(old,new)
    return source


def verify(run):
    evaluator=ramp.derived_module()
    mod=types.ModuleType('_food_ramp3000_saved_verifier')
    mod.__dict__.update(__name__=mod.__name__,__file__=str(HERE/'verify_ramp.py'),HERE=HERE,ROOT=ramp.ROOT,BASE=ramp.BASE,ORIGINAL=evaluator.ORIGINAL,COMMANDS=ramp.COMMANDS,
                        pin=evaluator.pin,jsonable=evaluator.jsonable,condition_score=evaluator.condition_score,checked_load=ramp.checked_load,command_at=ramp.command_at)
    exec(compile(derived_source(),str(HERE/'verify_ramp.py')+'::derived_verify_actual3000','exec'),mod.__dict__)
    report=mod.verify(run)
    # Validate the explicit new-source association for turns as well as speeds.
    r=json.loads((Path(run)/'RESULT.json').read_text());ident=json.loads((Path(run)/'IDENTITY.json').read_text())
    _,identity,fm=evaluator.pin()
    args=argparse.Namespace(checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256'],neutral_audit=r['neutral_reference']['audit'])
    fresh=ramp.neutral_reference(r['neutral_reference']['path'],args,identity,fm,ident)
    if fresh!=r['neutral_reference']:raise ValueError('matched neutral association changed')
    report['checks']['all_conditions_neutral_association_explicit']=True
    report['diagnostic_only']=True;report['food_qualification']=False
    report['scope']='same3000 checkpoint smooth1s command onset diagnostic, fullwindow original gates; NOT deployment/food qualification'
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out).resolve()
    if not Path(a.run).is_absolute() or not Path(a.out).is_absolute() or not out.is_relative_to(HERE) or out.exists():raise ValueError('fresh absolute ramp audit paths required')
    try:report=verify(a.run)
    except BaseException as exc:
        out.write_text(json.dumps(dict(integrity_all_pass=False,error=repr(exc),physics_integration_steps=0),indent=2)+'\n');raise
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(integrity_all_pass=True,engineering_pass=report['engineering_pass'],checks=len(report['checks']),diagnostic_only=True)))

if __name__=='__main__':main()
