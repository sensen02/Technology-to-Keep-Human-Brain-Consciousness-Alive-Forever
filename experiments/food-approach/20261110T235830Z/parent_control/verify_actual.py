"""Independent saved actual control audit binding only actual parent500 model API."""
from pathlib import Path
import json
import types
import evaluate_parent as ep

HERE=ep.HERE

def module():
    source=(ep.EVALUATION/'verify_actual.py').read_text()
    if source.count('from evaluate_motor import')!=2:raise ValueError('frozen verifier import sites changed')
    source=source.replace('from evaluate_motor import','from evaluate_parent import')
    marker="for n in ('verify_actual.py','evaluate_motor.py','EVAL_PREREG.json')"
    if source.count(marker)!=1:raise ValueError('frozen verifier source metadata changed')
    source=source.replace(marker,"for n in ('verify_actual.py','evaluate_parent.py','PREREG.json')")
    m=types.ModuleType('_parent_control_saved_audit');m.__dict__.update(__file__=str(HERE/'verify_actual.py'),__name__=m.__name__)
    exec(compile(source,str(HERE/'verify_actual.py')+'::derived','exec'),m.__dict__)
    original=m.verify
    def verify(run):
        report=original(run)
        r=json.loads((Path(run)/'RESULT.json').read_text())
        sw,identity,io=ep.pin()
        assoc=ep.neutral_association(identity)
        if r['parent_control_neutral']!=assoc:raise ValueError('parent500 neutral association changed')
        if report['checkpoint_sha256']!=ep.SHA:raise ValueError('audit parent500 binding mismatch')
        report['checks']['parent_neutral_association']=True
        report['source_hashes']={str(p):ep.sha256(p) for p in ep.source_paths()}
        report['scope']='actual fixed parent500 matched-ramp saved checkpoint/192state/full endpoint audit; no optimization, no Core integration or food qualification'
        return report
    m.verify=verify
    return m

if __name__=='__main__':module().main()
