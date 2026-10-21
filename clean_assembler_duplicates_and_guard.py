"""Clean the accidental duplicate `hit4=` blocks in the live assembler and fix the guard that kept re-adding them.

`if 'n_gate_passing' in t` was always true because the phrase also appears inside the count_definitions TEXT, so every
pass of the finalisation script re-inserted the same keyword argument (Python rejects the duplicate at parse time).
The guard now tests for the actual code usage `n_gate_passing=int(sum(`.
"""

import hashlib
import pathlib
import time

LIVE = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype/run_equilibrium_reissue_finalize.py')
FIN = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype/finalize_authoritative_assembler_rerun.py')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

lines = LIVE.read_text().split('\n')
idx = [i for i, l in enumerate(lines) if l.strip().startswith('hit4=sum(')]
print('hit4= blocks at lines:', [i + 1 for i in idx])
for i in reversed(idx[1:]):
    block = lines[i:i + 4]
    assert block[0].strip().startswith('hit4=sum(') and block[1].strip().startswith("if h['max_abs_qacc_mm_s2'] <= 1e-3"), block
    del lines[i:i + 4]
LIVE.write_text('\n'.join(lines))
after = sha(LIVE)
print('assembler sha after cleaning:', after, '| mtime:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(LIVE.stat().st_mtime)))
import py_compile, tempfile, os
py_compile.compile(str(LIVE), cfile=os.path.join(tempfile.mkdtemp(), 'x.pyc'), doraise=True)
print('assembler compiles OK; remaining hit4 blocks:',
      sum(1 for l in LIVE.read_text().split('\n') if l.strip().startswith('hit4=sum(')))

ft = FIN.read_text()
assert "if 'n_gate_passing' in t:" in ft
ft = ft.replace("if 'n_gate_passing' in t:", "if 'n_gate_passing=int(sum(' in t:", 1)
FIN.write_text(ft)
print('guard fixed (now tests the code usage, not the prose that mentions the old name)')
