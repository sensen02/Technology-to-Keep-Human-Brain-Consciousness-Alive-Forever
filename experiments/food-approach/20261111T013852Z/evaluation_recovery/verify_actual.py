"""Recovery-path independent audit, same frozen scoring/model/whole-array logic."""
from pathlib import Path
import sys,types
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import evaluate_motor as current
source=current.predecessor_source('verify_actual.py').replace('2000','10000')
module=types.ModuleType('_recovery_saved_actual_audit');module.__dict__.update(__file__=str(HERE/'verify_actual.py'),__name__=module.__name__)
exec(compile(source,str(HERE/'verify_actual.py')+'::frozen-predecessor','exec'),module.__dict__)
verify=module.verify
main=module.main
if __name__=='__main__':main()
