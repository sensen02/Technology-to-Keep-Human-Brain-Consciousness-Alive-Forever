from __future__ import annotations
import hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S302=Path('/home/sensen/Desktop/engine修复验收/s302-tier-gain-sweep/20261026T200813Z')
BASE=S302/'DERIVED_RUNNER.py'; OUT=Path(sys.argv[1]); SUBS=[]
def sub(o,n,t): SUBS.append(dict(tag=t,old=o,new=n))
sub("ARMS = ('w_5e-4', 'w_5e-5', 'w_5e-6', 'w_5e-7', 'w_5e-8')",
    "ARMS = ('w_5e-4', 'w_5e-5', 'w_2.5e-5', 'w_1.5e-5', 'w_1e-5', 'w_7.5e-6')", 'U1_arm_set')
sub("WMAP = {'w_5e-4': 5e-4, 'w_5e-5': 5e-5, 'w_5e-6': 5e-6, 'w_5e-7': 5e-7, 'w_5e-8': 5e-8}",
    "WMAP = {'w_5e-4': 5e-4, 'w_5e-5': 5e-5, 'w_2.5e-5': 2.5e-5, 'w_1.5e-5': 1.5e-5, 'w_1e-5': 1e-5, 'w_7.5e-6': 7.5e-6}", 'U2_weight_map')
sub("BUDGET = dict(model_load=1, banc_load=5, tier_build=5, select_touch_tier=10, edges_build=5,\n              network_init=5, live_step=20000, neural_tick=4000, live_fwd_position=5, live_forward=0,\n              diagnostic_forward=20005)",
    "BUDGET = dict(model_load=1, banc_load=6, tier_build=6, select_touch_tier=12, edges_build=6,\n              network_init=6, live_step=24000, neural_tick=4800, live_fwd_position=6, live_forward=0,\n              diagnostic_forward=24006)", 'U3_budget_six_arms')
sub("S302_PREREG_FROZEN", "S303_PREREG_FROZEN", 'U4_marker')
sub("""        for label in ARMS:
            run(label)
""", """        for label in ARMS:
            run(label)
            if label == ARMS[1]:
                _ref = Path('/home/sensen/Desktop/engine修复验收/s302-tier-gain-sweep/20261026T200813Z/w_5e-5')
                _b = np.load(_ref / 'SPIKES.npz')
                _cur = np.asarray(_SPIKES.get(label), dtype=np.uint8)
                _same = bool(_cur.size > 0 and np.array_equal(np.packbits(_cur, axis=1), _b['packed']))
                _ra = np.asarray(np.load(out / label / 'NEURAL.npz')['rates_hz'], float)
                _rb = np.asarray(np.load(_ref / 'NEURAL.npz')['rates_hz'], float)
                _d = float(np.abs(_ra - _rb).max())
                _rep = dict(arm=label, spikes_bitwise_equal=bool(_same), rates_max_abs_diff=_d,
                            passed=bool(_same and _d <= 1e-9),
                            note='S295 recorded that constraint warm-start persists across runs, so a sub-1e-9 rate difference is accepted and attributed rather than treated as a bug')
                dump(out / 'ARM_REPRO.json', _rep)
                if not _rep['passed']:
                    dump(out / 'STOP.json', dict(reason='w_5e-5 arm reproducibility gate failed', calls=calls))
                    return 2
""", 'U5_arm_repro_gate')
def main():
    der=BASE.read_text(encoding='utf-8')
    for s in SUBS:
        n=der.count(s['old']); assert n==1, f"anchor {s['tag']} x{n}"; der=der.replace(s['old'],s['new'])
    p=OUT/'DERIVED_RUNNER.py'; p.write_text(der,encoding='utf-8')
    json.dump(dict(base=str(BASE),base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),derived=str(p),derived_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                   substitutions=[dict(tag=s['tag'],old=s['old'],new=s['new'],occurrences=1) for s in SUBS]),open(OUT/'SUBSTITUTIONS.json','w'),indent=2)
    env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
    r=subprocess.run([sys.executable,str(p),'--output',str(OUT)],capture_output=True,text=True,cwd=str(ROOT),env=env)
    (OUT/'RUN_STDOUT.txt').write_text(r.stdout+'\n---STDERR---\n'+r.stderr,encoding='utf-8')
    print(json.dumps(dict(rc=r.returncode,err=(r.stderr or '')[-800:]))); return r.returncode
raise SystemExit(main())
