"""S300: derive the six-arm runner from S289 by asserted text substitutions, then run it."""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = ROOT / 'run_s289_decoder_range_support.py'
OUT = Path(sys.argv[1])

ABS = """    MODEL = mj.MjModel.from_xml_path(str(FROZEN))
    calls['model_load'] += 1
"""
SUBS = []
def sub(old, new, tag):
    SUBS.append(dict(tag=tag, old=old, new=new))

sub("        def run(label):\n",
    "        MODEL = mj.MjModel.from_xml_path(str(FROZEN))\n        calls['model_load'] += 1\n\n        def run(label):\n", 'S1a_load_model_once_outside_loop')
sub("            m = mj.MjModel.from_xml_path(str(FROZEN))\n            calls['model_load'] += 1\n",
    "            m = MODEL\n", 'S1b_use_shared_model')
sub("            sensory_gain = SENSORY_GAINS[label]\n", "            sensory_gain = SMAP[label]\n", 'S2_sensory_scale_entry')
sub("ARMS = ('decoder_original', 'decoder_range_x2')",
    "ARMS = ('s_1.00', 's_0.50', 's_0.25', 's_0.10', 's_0.05', 's_0.02')", 'S3a_arm_order')
sub("SENSORY_GAINS = {arm: 1. for arm in ARMS}",
    "SMAP = {'s_1.00': 1.0, 's_0.50': 0.5, 's_0.25': 0.25, 's_0.10': 0.1, 's_0.05': 0.05, 's_0.02': 0.02}\nSENSORY_GAINS = {arm: SMAP[arm] for arm in ARMS}", 'S3b_scale_map')
sub("DECODER_DENOMINATORS = {'decoder_original': 450., 'decoder_range_x2': 900.}",
    "DECODER_DENOMINATORS = {arm: 450. for arm in ARMS}", 'S3c_denominator_450_only')
sub("BUDGET = dict(model_load=2, tier_build=1, network_init=2, live_step=8000,\n              neural_tick=1600, live_fwd_position=2, live_forward=0,\n              diagnostic_forward=8002)",
    "BUDGET = dict(model_load=1, tier_build=1, network_init=6, live_step=24000,\n              neural_tick=4800, live_fwd_position=6, live_forward=0,\n              diagnostic_forward=24006)", 'S3d_budget_six_arms')
sub("""        run('decoder_original')
        gate = compare_archive(out / 'decoder_original', REFERENCE / 'gain_1')
        dump(out / 'GATE_DECODER_ORIGINAL_EXACT.json', gate)
        if not gate['passed']:
            dump(out / 'STOP.json', dict(reason='decoder_original exact reproduction failed; decoder_range_x2 NOT executed', candidate_executed=False, calls=calls))
            return 2
        run('decoder_range_x2')
""",
"""        for label in ARMS:
            run(label)
            if label == 's_1.00':
                gate = compare_archive(out / 's_1.00', REFERENCE / 'decoder_original')
                dump(out / 'GATE_ORIGINAL_EXACT.json', gate)
                if not gate['passed']:
                    dump(out / 'STOP.json', dict(reason='s_1.00 exact reproduction failed; the remaining five arms are unreachable by construction', candidate_executed=False, calls=calls))
                    return 2
""", 'S4_gate_inside_loop_after_first_arm')
sub("            traj.append(snapshot())\n            for i in range(400):\n",
    "            allspikes = []\n            traj.append(snapshot())\n            for i in range(400):\n", 'S5a_capture_init')
sub("                sp = tier.advance(2, submitted)",
    "                sp = tier.advance(2, submitted)\n                allspikes.extend([np.asarray(x, dtype=bool).copy() for x in sp])", 'S5b_capture_all_neurons')
sub("            dump(out / 'CALL_LEDGER.json', calls)\n\n        for label in ARMS:",
    """            _SPIKES[dest.name] = np.asarray(allspikes, dtype=np.uint8)
            dump(out / 'CALL_LEDGER.json', calls)

        for label in ARMS:""", 'S5c_keep_spikes_per_arm')
sub("    if not out.is_absolute() or out.exists():\n        raise ValueError('fresh absolute --output required')",
    "    if not out.is_absolute():\n        raise ValueError('absolute --output required')\n    if out.exists() and not (out / 'S300_PREREG_FROZEN').is_file():\n        raise ValueError('fresh output required')",
    'S6_allow_preregistered_directory')
sub("    dump(out / 'PREREG.json', prereg)", "    dump(out / 'PREREG_RUNTIME.json', prereg)", 'S7_keep_frozen_prereg')
sub("    out.mkdir(parents=True, exist_ok=False)", "    out.mkdir(parents=True, exist_ok=True)", 'S8_reuse_preregistered_directory')
sub("gate = compare_archive(out / 's_1.00', REFERENCE / 'decoder_original')",
    "gate = compare_archive(out / 's_1.00', Path('/home/sensen/Desktop/engine修复验收/s289-decoder-range-support/20261026T173400Z/decoder_original'))",
    'S9_gate_reference_is_the_s289_arm')
sub("        assert calls == BUDGET",
    """        for _name, _sp in _SPIKES.items():
            _dest = out / _name
            np.savez_compressed(_dest / 'SPIKES.npz', packed=np.packbits(_sp, axis=1), shape=np.array(_sp.shape),
                                spike_counts=_sp.sum(0).astype(np.int32))
            _rates = np.load(_dest / 'NEURAL.npz')['rates_hz']
            _pool = json.loads(POOLS.read_text())
            _ids = sorted({int(x) for r in _pool['measured']['resolved'] for x in r['pool_root_ids']})
            np.savez_compressed(_dest / 'POOL_NEURONS.npz', pool_root_ids=np.array(_ids, dtype=np.uint64),
                                local_index=np.array([int(np.flatnonzero(roots == i)[0]) if np.any(roots == i) else -1 for i in _ids], dtype=np.int64))
            _groups, _stats = channel_index_map(int(len(roots)), roots)
            dump(_dest / 'SENSORY_CHANNEL_INDEX.json', dict(channels=sum(len(v) for v in _groups.values()),
                                                             groups={f'{leg}:{ch}': [int(x) for x in idx] for leg, chs in _groups.items() for ch, idx in chs.items()},
                                                             stats=_stats))
            dump(_dest / 'SPIKE_SUMMARY.json', dict(shape=list(_sp.shape), duty=float(_sp.mean()), total_spikes=int(_sp.sum()),
                                                    per_neuron_min=int(_sp.sum(0).min()), per_neuron_max=int(_sp.sum(0).max())))
        assert calls == BUDGET""", 'S5d_append_spike_assets')

def main():
    src = BASE.read_text(encoding='utf-8')
    derived = src
    for s in SUBS:
        n = derived.count(s['old'])
        assert n == 1, f"anchor for {s['tag']} appears {n} times"
        derived = derived.replace(s['old'], s['new'])
    assert derived.count("def main(") == 1
    derived = derived.replace("def main(", "_SPIKES = {}\n\n\ndef main(", 1)
    assert "_SPIKES = {}" in derived
    path = OUT / 'DERIVED_RUNNER.py'
    path.write_text(derived, encoding='utf-8')
    json.dump(dict(base=str(BASE), base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
                   derived=str(path), derived_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   substitutions=[dict(tag=s['tag'], old=s['old'], new=s['new'], occurrences=1) for s in SUBS],
                   note='every anchor asserted to appear exactly once; no other edits'),
              open(OUT / 'SUBSTITUTIONS.json', 'w'), indent=2)
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1')
    r = subprocess.run([sys.executable, str(path), '--output', str(OUT)], capture_output=True, text=True, cwd=str(ROOT), env=env)
    (OUT / 'RUN_STDOUT.txt').write_text(r.stdout + '\n---STDERR---\n' + r.stderr, encoding='utf-8')
    print(json.dumps(dict(returncode=r.returncode, tail=(r.stdout or '')[-500:], err=(r.stderr or '')[-1500:])))
    return r.returncode

raise SystemExit(main())
