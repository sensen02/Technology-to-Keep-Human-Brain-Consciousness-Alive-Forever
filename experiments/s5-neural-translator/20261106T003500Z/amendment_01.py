"""Auditable one-time execution amendment; original experiment.py is never edited.
No driver is imported. The explicitly transformed source runs in its own namespace.
"""
from pathlib import Path
import hashlib,json
C=Path(__file__).resolve().parent
source=C/'experiment.py';record=json.loads((C/'AMENDMENT_01.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==record['original_source_sha256']
text=source.read_text()
assert text.count('.is_relative_to(O)')==3
text=text.replace('.is_relative_to(O)','.is_relative_to(O.resolve())')
old="C/'verify.py',C/'report.py']"
assert text.count(old)==1
text=text.replace(old,"C/'verify.py',C/'report.py',C/'amendment_01.py',C/'AMENDMENT_01.json']")
old="a['parts'][k]>=b['parts'][k]-1e-9 for k in ['foot_support_mean','com_z_min','first_nonfoot_s']"
new="(a['parts'][k] if a['parts'][k] is not None else a['duration_s']) >= (b['parts'][k] if b['parts'][k] is not None else b['duration_s'])-1e-9 for k in ['foot_support_mean','com_z_min','first_nonfoot_s']"
assert text.count(old)==1;text=text.replace(old,new)
old="guard='At1.3 J<=zero+1e-9 and footmean/minCOM/firstnonfoot>=zero-1e-9.',"
new="guard='At1.3 J<=zero+1e-9 and footmean/minCOM/firstnonfoot>=zero-1e-9. No contact first_nonfoot=None is right-censored at duration_s.',amendment=js(C/'AMENDMENT_01.json'),"
assert text.count(old)==1;text=text.replace(old,new)
old="dump(O/'CLEANUP_LEDGER.json',dict(importance='重要辅助',deleted=[]))"
new="dump(O/'CLEANUP_LEDGER.json',dict(importance='重要辅助',deleted=js(C/'AMENDMENT_01.json')['cleanup_executed']))\n    dump(O/'AMENDMENT_01.json',js(C/'AMENDMENT_01.json'))"
assert text.count(old)==1;text=text.replace(old,new)
# Original source plus this explicit transform completely specify the executed program.
namespace={'__file__':str(source),'__name__':'__main__'}
exec(compile(text,str(source)+'[AMENDMENT_01]','exec'),namespace)
