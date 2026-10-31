"""P1/S2: auditable 10 ms extension of S112, with S111 reference evidence.

Does not modify the original runner or archived evidence. The explicit literal
patches below derive a separate in-memory entry point from its hash-bound source.
Expanded source is archived before any model/network construction.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import run_com_contact_probe as base

ROOT = Path(__file__).resolve().parent
HISTORY = Path('/home/sensen/Desktop/engine修复验收/s111-pool-synergy-response/20261024T054520Z')

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output
    if not out.is_absolute() or out.exists():
        raise ValueError('fresh absolute output required')
    source = Path(base.__file__).read_text()
    patches = [
        ('PRE, TICKS, SUBSTEPS = 80, 6, 5', 'PRE, TICKS, SUBSTEPS = 80, 20, 5'),
        ('EXPECTED = dict(model_load=2, forward=62, step=60, neural_ticks=172)', 'EXPECTED = dict(model_load=2, forward=202, step=200, neural_ticks=200)'),
        ('physical_duration_ms=3,', 'physical_duration_ms=10,'),
        ('physical_steps_per_arm=30, physical_samples_per_arm=31,', 'physical_steps_per_arm=100, physical_samples_per_arm=101,'),
        ('early_window_s=[0, .003]', 'early_window_s=[0, .01]'),
        ("early_aggregation='trapezoid dx=.0001 divided by .003'", "early_aggregation='trapezoid dx=.0001 divided by .01'"),
        ("(31,49)", "(101,49)"), ('(6,17099)', '(20,17099)'),
        ('(31,90,48)', '(101,90,48)'), ('(31,m.nJmom)', '(101,m.nJmom)'),
        ('np.arange(31)*.0001', 'np.arange(101)*.0001'),
        ('net.steps == 86 and dec.interval == 85', 'net.steps == 100 and dec.interval == 99'),
        ('dec.elapsed_s-.043', 'dec.elapsed_s-.05'), ('net.t_ms/1000-.043', 'net.t_ms/1000-.05'),
        ('dx=.0001)/.003', 'dx=.0001)/.01'),
        ('"S112/P1/S2-S4: matched baseline/pool15 COM/contact diagnostic, 3 ms.', '"S111-extension/P1/S2: baseline/pool15 COM/contact diagnostic, 10 ms.'),
    ]
    for old,new in patches:
        if old not in source:
            raise ValueError('source patch no longer matches: '+old)
        source = source.replace(old,new)
    # Additional raw per-contact leg identities, including non-tarsal leg contacts.
    anchor = "rows = [sample()]; currents, spikes, features, commands, schedule = [], [], [], [], []"
    replacement = """original_sample = sample
            def sample():
                row = original_sample()
                legs = []
                for pair in row['contact_body_ids']:
                    members = []
                    for bid in pair:
                        name = (mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(bid)) or '').lower()
                        members.append(next((i for i, leg in enumerate(LEGS) if name.startswith(leg)), -1))
                    legs.append(members)
                row['contact_leg_ids'] = np.asarray(legs, dtype=np.int64).reshape(-1,2)
                return row
            rows = [sample()]; currents, spikes, features, commands, schedule = [], [], [], [], []"""
    if source.count(anchor)!=1:
        raise ValueError('sample instrumentation anchor changed')
    source=source.replace(anchor,replacement)
    oldhash=json.loads((HISTORY/'INPUT_HASHES.json').read_text())
    changed={name:dict(historical=h,current=digest(name)) for name,h in oldhash.items() if digest(name)!=h}
    assert len(changed)==4, 'review source drift before executing'
    # Main refuses existing outputs; capture wrapper must let it create the directory.
    namespace=dict(__name__='s111_extension_derived', __file__=str(Path(__file__).resolve()))
    exec(compile(source,str(Path(__file__).resolve())+'::derived','exec'),namespace)
    original_json=namespace['atomic_json']
    def archived_json(path, obj):
        if Path(path).name=='PREREG.json':
            obj.update(experiment='S111 10ms COM/contact extension; not S283',
                historical_reference=str(HISTORY), source_drift=changed,
                base_runner=str(Path(base.__file__).resolve()), base_runner_sha256=digest(base.__file__),
                derived_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                leg_order=['lf','lm','lh','rf','rm','rh'],
                historical_reproduction='must compare shared arrays to S111; drift is NOT hidden',
                extended_impulse_scope_s=[0,.01], extended_impulse_absolute_max_native=1e-4,
                extended_impulse_relative_max=.01, tolerance_relaxation=False,
                expected_command=sys.argv, absolute_force_units='NOT_ESTABLISHED')
            original_json(path,obj)
            (out/'DERIVED_RUNNER.py').write_text(source)
            original_json(out/'SOURCE_PATCHES.json',dict(literal_patches=patches,
                extra_instrumentation='contact_leg_ids body-name prefixes, read-only',
                source_drift=changed))
            return
        original_json(path,obj)
    namespace['atomic_json']=archived_json
    namespace['main']()

if __name__=='__main__':
    main()
