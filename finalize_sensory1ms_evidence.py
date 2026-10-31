"""Final explicit artifact ledger, no simulation."""
from pathlib import Path
import json,hashlib
O=Path('/home/sensen/Desktop/engine修复验收/s4-sensory1ms-command5ms/20261027T225000Z');R=Path(__file__).parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 pr=json.loads((O/'PREREGISTRATION.json').read_text());assert all(sha(Path(p))==h for p,h in pr['source_hashes'].items())
 code=['run_sensory1ms_single.py','test_sensory1ms_preflight.py','verify_sensory1ms_single.py','analyze_sensory1ms_command5ms.py','finalize_sensory1ms_evidence.py']
 for n in code:(O/n).write_bytes((R/n).read_bytes())
 desktop=['S4_SENSORY1MS_COMMAND5MS.md','S4_SENSORY1MS_COMMAND5MS.json','S4_SENSORY1MS_COMMAND5MS.png','S4_SENSORY1MS_COMMAND5MS_EARLY_CHAIN.png']
 for n in desktop:
  p=Path('/home/sensen/Desktop')/n;assert p.is_file();(O/n).write_bytes(p.read_bytes())
 change=dict(only_change='sensory5ms to1ms;command5ms retained',old_file='/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z/PREREGISTRATION.json',new_file=str(O/'PREREGISTRATION.json'),old_sha256=sha(Path('/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z/PREREGISTRATION.json')),new_sha256=sha(O/'PREREGISTRATION.json'),reason='Bounded intervention authorized sensory-only; static model/initial/pools/threshold/gain/filters unchanged',sources_unchanged=True)
 (O/'CHANGE_LEDGER.json').write_text(json.dumps(change,indent=2)+'\n')
 files=[O/n for n in code+desktop+['PREREGISTRATION.json','TEST.log','INITIAL_GATE.json','MAPPING.json','DERIVED_EPISODE.py','INSTRUMENTED_CONSUME.py','EXECUTION.json','SOURCE_UNCHANGED.json','VERIFICATION.json','ANALYSIS.json','ACCOUNTING.json','CHANGE_LEDGER.json']]+[O/'normal'/n for n in ['BODY.npz','INTERVALS.npz','TICKS.npz','SENSORY.npz','SENSORY_REFRESH.json','RECORDS.json','CONTACTS.json']]
 (O/'MANIFEST_SHA256.json').write_text(json.dumps({str(p):sha(p) for p in files},indent=2)+'\n');print('Sealed and verified',len(files),'explicit artifacts')
if __name__=='__main__':main()
