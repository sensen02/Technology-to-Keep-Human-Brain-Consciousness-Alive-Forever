"""One-shot source artifact generator for the documented canonical-path amendment.
Produces a concrete independently inspectable entrypoint; never runs simulation.
Original files are read-only and all generated paths must be absent.
"""
from pathlib import Path
import hashlib,json,ast
C=Path(__file__).resolve().parent
old=C/'experiment.py';new=C/'experiment_v2.py';supplement=C/'AMENDMENT_01_SUPPLEMENT.json'
assert not new.exists() and not supplement.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old)=='e65ce23014288f12e5196136ec8eab12a56102fa97d3492894a46b29b4499377'
s=old.read_text();changes=[]
def replace(oldtext,newtext,count):
    global s
    assert s.count(oldtext)==count,(oldtext,s.count(oldtext),count)
    s=s.replace(oldtext,newtext);changes.append(dict(old=oldtext,new=newtext,count=count))
replace('.is_relative_to(O)','.is_relative_to(O.resolve())',4)
replace("C/'verify.py',C/'report.py']","C/'verify.py',C/'report.py',C/'experiment.py',C/'amendment_01.py',C/'AMENDMENT_01.json',C/'build_v2.py',C/'AMENDMENT_01_SUPPLEMENT.json']",1)
replace("a['parts'][k]>=b['parts'][k]-1e-9 for k in ['foot_support_mean','com_z_min','first_nonfoot_s']","(a['parts'][k] if a['parts'][k] is not None else a['duration_s']) >= (b['parts'][k] if b['parts'][k] is not None else b['duration_s'])-1e-9 for k in ['foot_support_mean','com_z_min','first_nonfoot_s']",1)
replace("guard='At1.3 J<=zero+1e-9 and footmean/minCOM/firstnonfoot>=zero-1e-9.',","guard='At1.3 J<=zero+1e-9 and footmean/minCOM/firstnonfoot>=zero-1e-9. No contact first_nonfoot=None is right-censored at duration_s.',amendment=js(C/'AMENDMENT_01_SUPPLEMENT.json'),",1)
replace("dump(O/'CLEANUP_LEDGER.json',dict(importance='重要辅助',deleted=[]))","dump(O/'CLEANUP_LEDGER.json',dict(importance='重要辅助',deleted=js(C/'AMENDMENT_01.json')['cleanup_executed']))\n    dump(O/'AMENDMENT_01.json',js(C/'AMENDMENT_01.json'))\n    dump(O/'AMENDMENT_01_SUPPLEMENT.json',js(C/'AMENDMENT_01_SUPPLEMENT.json'))",1)
ast.parse(s)
assert '.is_relative_to(O)' not in s and s.count('.is_relative_to(O.resolve())')==4
# Canonical path contract for every helper, without creating or deleting anything.
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator')/C.name
assert not O.exists()
for path in [O/'PREREGISTRATION.json',O/'DATA/check.npz',O/'COVERAGE.json',O/'CONFIGS/model.npz']:
    assert path.resolve().is_relative_to(O.resolve())
assert not O.parent.resolve().is_relative_to(O.resolve())
new.write_text(s)
record=dict(importance='核心必需',amendment='01 concrete execution supplement',original=str(old),original_sha256=sha(old),executed_source=str(new),executed_source_sha256=sha(new),generator=str(Path(__file__)),generator_sha256=sha(Path(__file__)),changes=changes,static_ast_pass=True,canonical_four_sites_pass=True,previous_launcher_failure=dict(job='bash-32',exit_code=1,traceback='amendment_01.py line10 assert text.count(.is_relative_to(O))==3: AssertionError',cause='Count assertion correctly stopped before execution: actual4 sites including cleanup, not3.',physical_trials=0,model_fits=0,registration_written=False),total_pre_registration_failed_starts=2,physical_trials_before_registration=0,prior_amendment=str(C/'AMENDMENT_01.json'),clarification='This concrete source supersedes the unexecuted dynamic launcher. Original source, failed launcher and amendment record remain unchanged. Both failed starts occurred before any PREREGISTRATION/BEFORE was written. First successful registration freezes all originals and the actual v2 entrypoint; no reanchoring.',limits='Same finite data/fit/dev/final plan; only documented canonical path bug and no-contact right-censor edge fixed before first registration.')
supplement.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print('STATIC_PREFLIGHT_OK',new,sha(new))
