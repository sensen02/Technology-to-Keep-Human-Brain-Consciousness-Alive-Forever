"""Freeze explicit outputs and exact hash ledger, no simulation."""
from pathlib import Path
import json,hashlib
from run_one_second_motor_pair import dump,sha
R=Path('/run/media/sensen/Data2/cell_wound_prototype');O=Path('/home/sensen/Desktop/engine修复验收/s4-command1ms/20261027T222000Z')
files=['run_command1ms_single.py','test_command1ms_preflight.py','test_command1ms_legacy_reference.py','verify_command1ms_single.py','analyze_command1ms_single.py','plot_command1ms_early_chain.py','finalize_command1ms_evidence.py','COMMAND_CADENCE_REVIEW.zh-CN.md']
for name in files:
 p=R/name
 if p.exists():(O/name).write_bytes(p.read_bytes())
for name in ['S4_COMMAND1MS.md','S4_COMMAND1MS.json','S4_COMMAND1MS.png','S4_COMMAND1MS_EARLY_CHAIN.png']:
 p=Path('/home/sensen/Desktop')/name
 if p.exists():(O/name).write_bytes(p.read_bytes())
pr=json.loads((O/'PREREGISTRATION.json').read_text());assert all(sha(Path(p))==h for p,h in pr['source_hashes'].items());assert all(sha(Path(p))==h for p,h in pr['history_hashes'].items())
dump(O/'CHANGE_LEDGER.json',dict(parameter_change=dict(old_file='/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z/PREREGISTRATION.json',old_sha256=sha(Path('/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z/PREREGISTRATION.json')),new_file=str(O/'PREREGISTRATION.json'),new_sha256=sha(O/'PREREGISTRATION.json'),reason='Only completed motor readout/write hold5ms to1ms; original static INITIAL/tier/pools/neural/sensory/body clocks and normalization retained'),sources_unchanged=True,history_unchanged=True))
paths=[O/p for p in ['PREREGISTRATION.json','TEST.log','TEST_FAILURE.log','INITIAL_GATE.json','MAPPING.json','INSTRUMENTED_CONSUME.py','EXECUTION.json','SOURCE_UNCHANGED.json','VERIFICATION.json','ANALYSIS.json','ACCOUNTING.json','CHANGE_LEDGER.json']]
paths += [O/'normal'/p for p in ['INTERVALS.npz','TICKS.npz','BODY.npz','CONTACTS.json','RECORDS.json','SENSORY_OBSERVATIONS.json']]
paths += [O/p for p in files if (O/p).exists()]
paths += [O/p for p in ['S4_COMMAND1MS.md','S4_COMMAND1MS.json','S4_COMMAND1MS.png','S4_COMMAND1MS_EARLY_CHAIN.png'] if (O/p).exists()]
dump(O/'MANIFEST_SHA256.json',{str(p):sha(p) for p in paths});print('Verified all preregistered sources/history unchanged; explicit artifact hashes',len(paths))
