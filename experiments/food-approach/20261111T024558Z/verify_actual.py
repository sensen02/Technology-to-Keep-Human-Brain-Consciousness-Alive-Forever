"""Independent frozen saved-array auditor with explicit calibrated loader binding."""
from pathlib import Path
import importlib.util,types,json,argparse
import evaluate_motor as current
HERE=Path(__file__).resolve().parent
source=(current.PRIMARY/'verify_actual.py').read_text()
source=source.replace('from evaluate_motor import HERE,BASE,ORIGINAL,COMMANDS,pin,checked_load,flatten,command_at,score,jsonable,sha256','from evaluate_motor import HERE,BASE,ORIGINAL,COMMANDS,pin,checked_load,flatten,command_at,score,jsonable,sha256,verify_parent_right')
source=source.replace("    if r.get('native') is not None:","    if r['condition']=='right':\n        check('right_parent_arrays_exact',verify_parent_right(t,d,g,ident,identity)['all_exact'])\n    if r.get('native') is not None:")
m=types.ModuleType('_calibration_saved_array_audit');m.__dict__.update(__file__=str(HERE/'verify_actual.py'),__name__=m.__name__)
exec(compile(source,str(HERE/'verify_actual.py')+'::frozen-auditor','exec'),m.__dict__)
def verify(run):return m.verify(run)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 r=verify(a.run)
 with Path(a.out).open('x') as f:json.dump(current.jsonable(r),f,indent=2);f.write('\n')
 print(json.dumps(dict(integrity_all_pass=r['integrity_all_pass'],engineering_pass=r['engineering_pass'],checks=len(r['checks']))),flush=True)
