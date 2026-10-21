"""Rebuild the live assembler cleanly from the batch snapshot + the documented F2 change.

The live file had been damaged by my own deduplication pass (overlapping 3-line blocks removed 4 extra lines of the README
substitution dict, so the assembler raised KeyError('n05')). The clean authoritative revision is therefore rebuilt as
snapshot bytes + the four F2 replacements, all of which are written out here explicitly.
"""

import hashlib
import pathlib
import py_compile
import os
import tempfile
import time
import zipfile

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
BATCH = pathlib.Path('/home/sensen/Desktop/engine修复验收/S2-equilibrium-A-corrected-reissued-20261015T161259')
LIVE = ROOT / 'run_equilibrium_reissue_finalize.py'
DECLARED_OLD = '5c4212768b89629f46623578fb9003e7cb67383e98077539ba5d63a3525aa2d0'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

with zipfile.ZipFile(BATCH / 'source_snapshot.zip') as z:
    snap = z.read('run_equilibrium_reissue_finalize.py')
assert hashlib.sha256(snap).hexdigest() == DECLARED_OLD
t = snap.decode('utf-8')
print('snapshot revision restored in memory:', hashlib.sha256(snap).hexdigest())

REPL = [
    ("""    hits = runs_block['A_ms240']['per_equilibrium_hits'] or []
    report['per_equilibrium_summary'] = dict(""",
     """    hits = runs_block['A_ms240']['per_equilibrium_hits'] or []
    _gate_item = lambda h: dict(qacc=bool(h['max_abs_qacc_mm_s2'] <= DECLARED_GATE['max_abs_qacc_mm_s2']),
                                six_tarsi=bool(h['six_tarsi_in_contact']),
                                zero_nonfoot=bool(h['n_nonfoot_contacts'] == 0),
                                diamond=bool(h['max_u_diamond'] is not None and h['max_u_diamond'] <= 1.0 + 1e-9))
    _all_four = lambda h: all(_gate_item(h).values())
    report['per_equilibrium_summary'] = dict("""),
    ("""        n_gate_passing=int(sum(1 for h in hits if h['max_abs_qacc_mm_s2'] <= DECLARED_GATE['max_abs_qacc_mm_s2'])),
        n_search_tolerance_passing=int(sum(1 for h in hits if h['max_abs_qacc_mm_s2'] <= SEARCH_TOL_MM_S2)),""",
     """        n_all_four_gate_items_passing=int(sum(1 for h in hits if _all_four(h))),
        n_qacc_item_passing=int(sum(1 for h in hits if h['max_abs_qacc_mm_s2'] <= DECLARED_GATE['max_abs_qacc_mm_s2'])),
        n_search_tolerance_passing=int(sum(1 for h in hits if h['max_abs_qacc_mm_s2'] <= SEARCH_TOL_MM_S2)),
        count_definitions=('three different counts, each defined: (1) n_search_tolerance_passing = max|qacc| <= 1.0 mm/s^2; '
                           '(2) n_qacc_item_passing = the qacc item of the declared gate, max|qacc| <= 1e-3 mm/s^2; '
                           '(3) n_all_four_gate_items_passing = all four declared gate items. The old field name '
                           'n_gate_passing counted only item (2) and was removed on 2026-10-15.'),"""),
    ("""                      declared_gate_passed=bool(h['max_abs_qacc_mm_s2'] <= DECLARED_GATE['max_abs_qacc_mm_s2']
                                                and h['six_tarsi_in_contact'] and h['n_nonfoot_contacts'] == 0
                                                and h['max_u_diamond'] is not None and h['max_u_diamond'] <= 1.0 + 1e-9))""",
     """                      gate_items=_gate_item(h), all_four_gate_items_passed=_all_four(h),
                      declared_gate_passed=_all_four(h))"""),
    ("""  其中（<= 1e-3 mm/s^2）`%(hit3)s`。""",
     """  其中 qacc 项（<= 1e-3 mm/s^2）`%(hit3)s` 个；**四项门限全过 %(hit4)s 个**（起点 16 / 117 / 131 / 212）。
  起点 207 的 `max|qacc|` 虽过 qacc 项，但 `six_tarsi_in_contact = false` —— 它的 51 个跗节接触只覆盖 **5 条腿**（缺 LF），
  所以"六足跗节接触"这一项不过；起点 5 的 qacc 项未过。**"过门限"一律指四项全过。**"""),
    ("""        hit3=runs_block['A_ms240']['n_starts_below_1e_minus_3_mm_s2'],""",
     """        hit3=runs_block['A_ms240']['n_starts_below_1e_minus_3_mm_s2'],
        hit4=sum(1 for h in (runs_block['A_ms240']['per_equilibrium_hits'] or [])
                 if h['max_abs_qacc_mm_s2'] <= 1e-3 and h['six_tarsi_in_contact'] and h['n_nonfoot_contacts'] == 0
                 and h['max_u_diamond'] is not None and h['max_u_diamond'] <= 1 + 1e-9),"""),
]
for old, new in REPL:
    assert t.count(old) == 1, 'anchor not unique: %r' % old[:60]
    t = t.replace(old, new, 1)
LIVE.write_text(t)
h = sha(LIVE)
py_compile.compile(str(LIVE), cfile=os.path.join(tempfile.mkdtemp(), 'x.pyc'), doraise=True)
print('clean authoritative revision written:', h,
      '| mtime:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(LIVE.stat().st_mtime)))
print('compiles OK | hit4 blocks:', sum(1 for l in LIVE.read_text().split('\n') if l.strip().startswith('hit4=sum(')))
print('README dict keys present:', all(k in LIVE.read_text() for k in ('n05=', 'n10=', 'n20=', 'hit4=', 'peq=', 'void=')))
