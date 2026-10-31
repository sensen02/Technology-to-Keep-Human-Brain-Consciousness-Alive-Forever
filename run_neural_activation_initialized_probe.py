"""S141/P1: one new neural-history-derived activation initial condition.
Import inert; parent executes once after independent source review. No old file edits.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import types
import csv
import numpy as np

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT/'run_extended_initialization_probe.py'
RECORDER = ROOT/'run_constraint_row_recording.py'
S121 = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z')
OUTPUT = Path('/home/sensen/Desktop/engine修复验收/s141-neural-activation-initialization/20261024T200305Z')
README = ROOT/'S141_NEURAL_ACTIVATION_INITIALIZATION_README.md'
FORMAL = Path('/home/sensen/Desktop/engine修复验收/s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json')
ARM = 'init80ms'
EXPECTED = dict(model_load=1, forward=122, step=120, neural_ticks=166)
FROZEN = {str(ORIGINAL): '6385a0842dfb8042f4c97cebcb71a2a68f1bd435ef5fc980d0b852bea7d13eea',
          str(RECORDER): '55afefe38079eba5e00e2bd37c2b249cc21a052cee9bb3079b93a82c2631d4ee'}
ANCHOR = "            assert d.time == 0 and np.array_equal(d.qpos,q) and not any(x.any() for x in (d.qvel,d.act,d.ctrl,d.xfrc_applied,d.qfrc_applied))\n"
INSERTION = "            _initialize_release(mj,m,d,dec,release_ctrl,warmu,dest,reserve)\n"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def check_hashes(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected: raise ValueError('input changed: '+str(path))


def load_private(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_sources():
    check_hashes(FROZEN)
    inputs = json.loads((S121/'INPUT_HASHES.json').read_text())
    check_hashes(inputs)
    complete = json.loads((S121/'COMPLETE.json').read_text())
    result = json.loads((S121/ARM/'RESULT.json').read_text())
    check_hashes(complete['source_chain_hashes']); check_hashes(result['hashes'])
    assert sha(S121/ARM/'TRAJECTORY.npz') == result['trajectory_sha256']
    assert result['calls'] == dict(model_load=1,forward=121,step=120,neural_ticks=166)
    assert sha(FORMAL) == '582a44503fc8dd5db23fc093b90af5c16eb5d928c4bd7a43eee4d6cf1f62855d'
    files = [Path(__file__).resolve(), ORIGINAL, RECORDER, README, FORMAL, *map(Path,inputs)]
    files += sorted(p for p in S121.rglob('*') if p.is_file())
    commitments = {str(p):sha(p) for p in dict.fromkeys(files)}
    with np.load(S121/ARM/'TRAJECTORY.npz',allow_pickle=False) as z:
        baseline = {k:z[k].copy() for k in z.files}
    return commitments, baseline


def patched_source(source):
    assert source.count(ANCHOR) == 1, 'release hook anchor not unique'
    assert source.count('            rows = [sample()]') == 1
    assert source.index(ANCHOR) < source.index('            rows = [sample()]')
    return source.replace(ANCHOR, ANCHOR+INSERTION)


def configure(original, validated):
    commitments, baseline = validated
    check_hashes(commitments)
    assert original.PRE == 160 and original.TICKS == 6
    assert original.TIMESTEPS == {ARM:.000025} and original.SUBSTEPS == {ARM:20}
    with np.load(original.WEIGHTS,allow_pickle=False) as z:
        assert np.array_equal(z['weights'],baseline['weights'])
    original.EXPECTED = EXPECTED.copy()
    original.EXPECTED_PER_ARM = {ARM:EXPECTED.copy()}
    original.OUTPUT_PARENT = OUTPUT
    recorder = load_private(RECORDER,'_s141_passive_s121')
    save_json, save_npz = original.atomic_json, original.atomic_npz
    original_capture = original.com_contact_record
    records, metadata, warm_saved, event = [], {}, {}, {}
    source = ORIGINAL.read_text()
    compiled = patched_source(source)
    intervention = dict(task='S141/P1/S2-S4', paired_s121=str(S121),
        formal_s119=str(FORMAL),formal_s119_sha256=sha(FORMAL),training_constraints=json.loads(FORMAL.read_text()),
        initialization_condition='NEW: act initialized ONCE from actual final PRE160 neural decoded command',
        neural_history_only=True, external_algorithm_controller=False, pose_fix=False,
        initial_ctrl='actual warmu[-1] / release_decoded at neural source .080s; normal tick0 replaces it',
        physical_initial='original qpos, zero qvel/external loads; activation set once by compiled actuator_actadr; ctrl from final PRE command',
        initialization='PRE160 held actual zero-act initial observation; genuine net.step; no body steps; one declared release activation write',
        act_initialization_count=1, additional_release_forward=1, candidate_eligibility='NONE_NEW_INITIALIZATION_CONDITION',
        strict_s119_comparison_valid=False, training_acceptance='NOT_APPLICABLE_NEW_INITIALIZATION_CONDITION',
        selection_rule='NONE; diagnostic actual-vs-S121 only, no S119 candidate admission',
        physical_samples_per_arm={ARM:121}, expected_calls=EXPECTED,
        expected_calls_per_arm={ARM:EXPECTED},
        call_counts='122 explicit forward: initial sensory + release recomputation +120 after-step; RK4 internal evaluations excluded',
        private_source_insertion=dict(anchor=ANCHOR,insertion=INSERTION,
            original_sha256=sha(ORIGINAL),compiled_source_sha256=hashlib.sha256(compiled.encode()).hexdigest()),
        baseline_is_historical=True, single_config=True, all_weights_identical=True,
        deployment=False,selection=False,promotion=False,checkpoint_replacement=False,
        standing_or_walking_claim=False,physical_units='NOT_ESTABLISHED',behavior_claim='NONE')

    def initialize(mj,m,d,dec,release_ctrl,warmu,dest,reserve):
        assert not event, 'activation initialization called more than once'
        assert d.time == 0 and dec.interval == 159 and abs(dec.elapsed_s-.08) < 1e-12
        assert m.nu == m.na == 90
        adr = np.asarray(m.actuator_actadr).copy(); num = np.asarray(m.actuator_actnum).copy()
        assert np.array_equal(num,np.ones(90,dtype=num.dtype))
        assert np.array_equal(np.sort(adr),np.arange(90)), 'must be a bijection, one activation per muscle'
        command = np.asarray(warmu[-1]).copy()
        assert command.shape == (90,) and np.isfinite(command).all()
        assert np.array_equal(command,release_ctrl)
        assert not any(x.any() for x in (d.act,d.ctrl,d.qvel,d.qfrc_applied,d.xfrc_applied))
        qbefore = d.qpos.copy()
        before_act = d.act.copy(); before_ctrl = d.ctrl.copy()
        original.require_no_callbacks(original.callback_snapshot(mj))
        d.ctrl[:] = command
        d.act[adr] = command  # ONLY activation initialization write; physics evolves it thereafter.
        reserve(forward=1); mj.mj_forward(m,d)
        assert d.time == 0 and np.array_equal(d.qpos,qbefore) and not d.qvel.any()
        assert np.array_equal(d.act[adr],command) and np.array_equal(d.ctrl,command)
        assert not d.xfrc_applied.any() and not d.qfrc_applied.any()
        original.require_no_callbacks(original.callback_snapshot(mj))
        event.update(source_s=.08,physical_s=0.,dec_interval=159,act_write_count=1,
            compiled_actadr=adr.tolist(),compiled_actnum=num.tolist(),ctrl_source='actual warmu[-1] == release_decoded',
            extra_forward_count=1,record_initial_sample='after initialized act/ctrl and release forward; never force historical zero sample')
        actual = dict(warm_saved)
        actual.update(body_act=d.act.copy(),body_ctrl=d.ctrl.copy(),
            pre_release_zero_act=before_act,pre_release_zero_ctrl=before_ctrl,
            BODY_ACT_BEFORE_INITIALIZATION=before_act, BODY_CTRL_BEFORE_INITIALIZATION=before_ctrl,
            BODY_QPOS_BEFORE_INITIALIZATION=qbefore, BODY_QVEL_BEFORE_INITIALIZATION=np.zeros_like(d.qvel),
            actuator_actadr=adr,actuator_actnum=num,activation_initialization_command=command)
        save_npz(dest/'INITIALIZATION.npz',**actual)
        save_json(dest/'ACT_INITIALIZATION.json',event)
        with (dest/'PHASE_ACT_INIT.csv').open('w',newline='') as f:
            w = csv.writer(f); w.writerow(['phase','physical_s','source_s','actuator_index','activation_address','activation_count','before_act','after_act','actual_PRE160_ctrl','after_ctrl'])
            for i in range(90): w.writerow(['release_act_init',0,.08,i,int(adr[i]),int(num[i]),float(before_act[adr[i]]),float(d.act[adr[i]]),float(command[i]),float(d.ctrl[i])])

    def capture(mj,m,d,floor_ids):
        result = original_capture(mj,m,d,floor_ids)
        assert event, 'sample cannot precede release activation initialization'
        records.append(recorder.constraint_record(mj,m,d,len(records)))
        if not metadata:
            metadata.update(recorder.constraint_metadata(mj,m))
            metadata['sampling'] = 'S141 initial sample AFTER release activation initialization and additional forward; later original after-step samples; pure constraint copies'
            metadata['clocks'] = 'sample0 physical_tick=-1, source .08 release command already installed; tick0 then consumes .080..0805 normally'
        return result

    def json_hook(path,value):
        path = Path(path)
        if path.name == 'INPUT_HASHES.json': value = dict(value,**commitments)
        elif path.name == 'PREREG.json':
            value = dict(value,**intervention)
            value['source_hashes'] = dict(value['source_hashes'],**commitments)
            save_json(path.parent/'S141_PREREG.json',value)
            (path.parent/'PRIVATE_RUNNER_SOURCE.py').write_text(compiled)
        elif path.name == 'COMPLETE.json':
            assert len(records) == 121 and event['act_write_count'] == 1
            assert {k:value['ledger'][k] for k in EXPECTED} == EXPECTED
            check_hashes(commitments)
            value = dict(value,**intervention)
            value['source_chain_hashes'] = dict(value['source_chain_hashes'],**commitments)
            value['constraint_samples'] = len(records)
        return save_json(path,value)

    def npz_hook(path,**arrays):
        path = Path(path)
        if 'weights' in arrays: assert np.array_equal(arrays['weights'],baseline['weights'])
        if path.name == 'INITIALIZATION.npz':
            assert not warm_saved and not event
            warm_saved.update({k:np.asarray(v).copy() for k,v in arrays.items()})
            return save_npz(path.parent/'PRE_RELEASE_ZERO_INITIALIZATION.npz',**arrays)
        if path.name == 'TRAJECTORY.npz':
            assert event and len(records) == 121
            assert np.array_equal(arrays['time'],baseline['time'])
            assert np.array_equal(arrays['act'][0][np.asarray(event['compiled_actadr'])],warm_saved['release_decoded'])
            assert np.array_equal(arrays['ctrl'][0],warm_saved['release_decoded'])
            save_npz(path.parent/'CONSTRAINT_ROWS.npz',**recorder.pack_records(records))
            save_json(path.parent/'CONSTRAINT_ROW_METADATA.json',metadata)
            keys = ['load_ratio','limit_excess','root_angle_rad','nonfoot_depth','all_six_gates']
            with (path.parent/'ACTUAL_VS_S121.csv').open('w',newline='') as f:
                w = csv.writer(f); w.writerow(['sample','time_s']+[s+'_'+k for k in keys for s in ('S121','S141','delta')])
                for i,t in enumerate(arrays['time']):
                    row = [i,float(t)]
                    for k in keys:
                        a,b = float(baseline[k][i]),float(arrays[k][i]); row += [a,b,b-a]
                    w.writerow(row)
            save_json(path.parent/'BASELINE_COMPARISON.json',dict(
                candidate_eligibility='NONE_NEW_INITIALIZATION_CONDITION',strict_s119_comparison_valid=False,
                historical_s121=str(S121),baseline_trajectory_sha256=sha(S121/ARM/'TRAJECTORY.npz'),
                per_sample_csv='ACTUAL_VS_S121.csv',initial_activation_changed=True,
                metrics={k:dict(s121_min=float(np.min(baseline[k])),actual_min=float(np.min(arrays[k])),
                    s121_max=float(np.max(baseline[k])),actual_max=float(np.max(arrays[k]))) for k in keys},
                no_standing_or_walking_claim=True))
        return save_npz(path,**arrays)

    original._initialize_release = initialize
    original.com_contact_record = capture
    original.atomic_json = json_hook; original.atomic_npz = npz_hook
    return records, commitments


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True,type=Path)
    out = p.parse_args().output
    if out != OUTPUT or not out.is_absolute() or out.exists(): raise ValueError('exact fresh OUTPUT only; no resume')
    validated = validate_sources()
    source = ORIGINAL.read_text(); compiled = patched_source(source)
    original = types.ModuleType('_s141_private_original'); original.__file__ = str(ORIGINAL)
    exec(compile(compiled,str(ORIGINAL),'exec'),original.__dict__)
    configure(original,validated)
    original.main()


if __name__ == '__main__': main()
