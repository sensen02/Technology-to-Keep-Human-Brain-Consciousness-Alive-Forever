"""Full independent actual audit using SAME new half test input formula."""
from pathlib import Path
import types,json,argparse
import evaluate_motor as current
HERE=Path(__file__).resolve().parent
if current.sha256(current.PRIMARY/'verify_actual.py')!='1293cf1db7b61a90eeb13500a18eb53796b4ff8891a38502dbfac3f94e5e7245':raise ValueError('frozen audit changed')
source=(current.PRIMARY/'verify_actual.py').read_text()
m=types.ModuleType('_half_actual_saved_audit');m.__dict__.update(__file__=str(HERE/'verify_actual.py'),__name__=m.__name__)
exec(compile(source,str(HERE/'verify_actual.py')+'::unchanged-audit-body','exec'),m.__dict__)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();r=m.verify(a.run)
 with Path(a.out).open('x') as f:json.dump(current.jsonable(r),f,indent=2);f.write('\n')
 print(json.dumps(dict(integrity_all_pass=r['integrity_all_pass'],engineering_pass=r['engineering_pass'],checks=len(r['checks']))),flush=True)
