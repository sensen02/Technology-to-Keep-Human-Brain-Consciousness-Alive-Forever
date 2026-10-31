"""S302: derive the five weight_scale arms from the S300 runner by asserted substitutions, then run."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = Path('/home/sensen/Desktop/engine修复验收/s300-leginfo-workpoint-repaired/20261026T195730Z/DERIVED_RUNNER.py')
OUT = Path(sys.argv[1])
SUBS=[]
def sub(old,new,tag): SUBS.append(dict(tag=tag,old=old,new=new))
sub("ARMS = ('s_1.00', 's_0.50', 's_0.25', 's_0.10', 's_0.05', 's_0.02')",
    "ARMS = ('w_5e-4', 'w_5e-5', 'w_5e-6', 'w_5e-7', 'w_5e-8')", 'T1_arm_names')
sub("SMAP = {'s_1.00': 1.0, 's_0.50': 0.5, 's_0.25': 0.25, 's_0.10': 0.1, 's_0.05': 0.05, 's_0.02': 0.02}",
    "SMAP = {a: 1.0 for a in ARMS}\nWMAP = {'w_5e-4': 5e-4, 'w_5e-5': 5e-5, 'w_5e-6': 5e-6, 'w_5e-7': 5e-7, 'w_5e-8': 5e-8}", 'T2_sensory_gain_one_and_weight_map')
sub("BUDGET = dict(model_load=1, tier_build=1, network_init=6, live_step=24000,\n              neural_tick=4800, live_fwd_position=6, live_forward=0,\n              diagnostic_forward=24006)",
    "BUDGET = dict(model_load=1, banc_load=5, tier_build=5, select_touch_tier=10, edges_build=5,\n              network_init=5, live_step=20000, neural_tick=4000, live_fwd_position=5, live_forward=0,\n              diagnostic_forward=20005)", 'T3_budget_no_graded_branch')
sub("        built = build_neural_tier(TierConfig(sign_fill='all_empty').validate())\n        calls['tier_build'] += 1",
    "        _orig_build_neural_tier = build_neural_tier\n        def build_neural_tier(cfg=None, data_path=None):\n            calls['banc_load'] += 1\n            calls['edges_build'] += 1\n            calls['select_touch_tier'] += int(getattr(cfg, 'growth_iterations', 2))\n            return _orig_build_neural_tier(cfg, data_path)\n        built = build_neural_tier(TierConfig(sign_fill='all_empty').validate())\n        calls['tier_build'] += 1", 'T4_counted_build_wrapper')
sub("            tier = NeuralTier(built, TierConfig(sign_fill='all_empty', background_seed=0).validate(), OwnershipLedger())",
    """            AB = built if label == ARMS[0] else build_neural_tier(TierConfig(sign_fill='all_empty', weight_scale_uS_per_synapse=WMAP[label]).validate())
            if label != ARMS[0]:
                calls['tier_build'] += 1
                assert np.array_equal(np.asarray(AB['tier']['root_ids']), np.asarray(roots)), 'per-arm tier root ids differ'
            _ed = AB['edges']
            _w = dict(arm=label, weight=WMAP[label], fingerprint=AB['fingerprint'],
                      roots_sha256=hashlib.sha256(np.ascontiguousarray(np.asarray(AB['tier']['root_ids'])).tobytes()).hexdigest())
            for _n in ('We', 'Wi'):
                _m = _ed[_n] if isinstance(_ed, dict) else getattr(_ed, _n.lower(), None)
                if _m is not None:
                    _w[_n + '_data_sha256'] = hashlib.sha256(np.ascontiguousarray(_m.data).tobytes()).hexdigest()
                    _w[_n + '_indices_sha256'] = hashlib.sha256(np.ascontiguousarray(_m.indices).tobytes()).hexdigest()
                    _w[_n + '_indptr_sha256'] = hashlib.sha256(np.ascontiguousarray(_m.indptr).tobytes()).hexdigest()
                    _w[_n + '_data_sum'] = float(np.asarray(_m.data, float).sum())
                    np.savez_compressed(out / ('EDGES_' + label + '_' + _n + '.npz'), data=np.asarray(_m.data), indices=np.asarray(_m.indices), indptr=np.asarray(_m.indptr), shape=np.asarray(_m.shape))
            _STRUCT[label] = _w
            tier = NeuralTier(AB, TierConfig(sign_fill='all_empty', background_seed=0).validate(), OwnershipLedger())""", 'T5_per_arm_weight_build_and_structure')
sub("gate = compare_archive(out / 's_1.00', Path('/home/sensen/Desktop/engine修复验收/s289-decoder-range-support/20261026T173400Z/decoder_original'))",
    "gate = compare_archive(out / 'w_5e-4', Path('/home/sensen/Desktop/engine修复验收/s289-decoder-range-support/20261026T173400Z/decoder_original'))\n                gate2 = compare_archive(out / 'w_5e-4', Path('/home/sensen/Desktop/engine修复验收/s300-leginfo-workpoint-repaired/20261026T195730Z/s_1.00'))\n                dump(out / 'GATE_VS_S300.json', gate2)\n                gate = dict(passed=bool(gate['passed'] and gate2['passed']), checks=gate.get('checks', []) + gate2.get('checks', []), reference_s289=gate, reference_s300=gate2)", 'T6_gate_vs_s289_and_s300')
sub("            if label == 's_1.00':", "            if label == ARMS[0]:", 'T7_gate_first_arm')
sub("S300_PREREG_FROZEN", "S302_PREREG_FROZEN", 'T8_frozen_marker')
def main():
    src=BASE.read_text(encoding='utf-8'); der=src
    for s in SUBS:
        n=der.count(s['old']); assert n==1, f"anchor {s['tag']} appears {n} times"; der=der.replace(s['old'],s['new'])
    assert der.count("def main(")==1
    der=der.replace("def main(","_STRUCT = {}\n\n\ndef main(",1)
    der=der.replace("        assert calls == BUDGET",
        """        dump(out / 'ARM_STRUCTURE.json', _STRUCT)
        assert calls == BUDGET""",1)
    p=OUT/'DERIVED_RUNNER.py'; p.write_text(der,encoding='utf-8')
    json.dump(dict(base=str(BASE),base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),derived=str(p),derived_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                   substitutions=[dict(tag=s['tag'],old=s['old'],new=s['new'],occurrences=1) for s in SUBS],
                   note='no edge/selection/fingerprint code added; only a counting wrapper and the weight argument'),
              open(OUT/'SUBSTITUTIONS.json','w'),indent=2)
    env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
    r=subprocess.run([sys.executable,str(p),'--output',str(OUT)],capture_output=True,text=True,cwd=str(ROOT),env=env)
    (OUT/'RUN_STDOUT.txt').write_text(r.stdout+'\n---STDERR---\n'+r.stderr,encoding='utf-8')
    print(json.dumps(dict(rc=r.returncode,err=(r.stderr or '')[-1200:])))
    return r.returncode
raise SystemExit(main())
