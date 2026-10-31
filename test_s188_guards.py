"""Engine-free guards demanded in S188: row/dof misalignment must raise, duplicate tags must fail before writing."""
import json
from pathlib import Path
from s179_core import build_row_dof_map, validate_identity, IdentityError
def assert_unique_tags(records, context=''):
    tags=[r.get('tag') for r in records]
    dup=sorted({t for t in tags if t is not None and tags.count(t)>1})
    if dup: raise AssertionError(f'{context}: duplicate tags before write: {dup[:5]} ({len(dup)} total)')
names=['x']*6+[f'{leg}{i}' for leg in ('LF','LH','LM','RF','RH','RM') for i in range(7)]
rm=build_row_dof_map(names)
out=[]
try:
    validate_identity(rm,rows=[13],dofs=[13],context='deliberate'); out.append('FAIL_misalignment_not_raised')
except IdentityError: out.append('PASS_misalignment_raised')
validate_identity(rm,rows=[13],dofs=[19],context='ok'); out.append('PASS_aligned_ok')
recs=[{'tag':'a'},{'tag':'a'}]
try:
    assert_unique_tags(recs,'test'); out.append('FAIL_dup_not_raised')
except AssertionError: out.append('PASS_dup_raised')
assert_unique_tags([{'tag':'a'},{'tag':'b'}],'ok'); out.append('PASS_unique_ok')
print(json.dumps(out))
assert all(x.startswith('PASS') for x in out), out
