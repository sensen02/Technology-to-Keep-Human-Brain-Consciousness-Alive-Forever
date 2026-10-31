"""S162/P1: ONE S154-matched run; only sensory refresh stride 10 -> 1.
Import/prepare are inert. No extra ablations, parameter changes or controller.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, hashlib, json, sys, types
ROOT = Path(__file__).resolve().parent
S154 = ROOT/'run_activation_rate_probe.py'
REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
OUTPUT_PARENT = Path('/home/sensen/Desktop/engine修复验收/s162-fast-sensory')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inert(path, name):
    m=types.ModuleType(name); m.__file__=str(path)
    exec(compile(path.read_text(),str(path),'exec'),m.__dict__); return m

HELPERS = '''
import dataclasses, shutil

def s162_freeze_bound(out, identities):
    folder=out/'BOUND_INPUT_FREEZE'; folder.mkdir()
    records=[]
    for index,(name,digest) in enumerate(identities.items()):
        original=Path(name); frozen=folder/(f'{index:04d}_'+original.name)
        shutil.copyfile(original,frozen)
        assert sha(frozen)==digest==sha(original)
        records.append(dict(source=name,sha256=digest,frozen=str(frozen)))
    atomic_json(out/'BOUND_INPUT_FREEZE.json',dict(records=records,
        excluded_new_parent_tools=True, before_actual_model_load=True))
'''

OBSERVATION = '''                    assert abs(d.time-tick*.0005) < 1e-12
                    observed = observe_quantities(m,d,sensory_ref,joints,geom_leg)
                    current, encoding_record = enc.encode(observed)
                    refresh_times.append(float(d.time))
                    atomic_json(dest/f'OBSERVATION_{tick:02d}.json',raw_observation_record(observed)[0])
                    atomic_json(dest/f'ENCODING_{tick:02d}.json',dict(
                        schema='S162 raw OBSERVATION unchanged; encoder record retained separately',
                        tick=tick, physical_sample_time_s=float(d.time),
                        body_last_step_index=tick*20, current_nA=current.tolist(),
                        features=encoding_record, raw_file=f'OBSERVATION_{tick:02d}.json'))
                    sensory_schedule.append(dict(tick=tick, physical_sample_time_s=float(d.time),
                        body_last_step_index=tick*20, held_from_tick=None,
                        held_current_between_ticks=False, refreshed_every_tick=True,
                        source_start_s=(PRE+tick)*.0005,source_end_s=(PRE+tick+1)*.0005,
                        order='current integrated body -> observe -> encode -> neural -> decode -> immediate ctrl -> 20 RK4 body substeps'))'''

PARAMETERS = '''            atomic_json(dest/'ENCODER_PARAMETERS.json',dict(
                schema='S162 exact configured encoder and observation references',
                config=dataclasses.asdict(enc.cfg), n_neurons=enc.n,
                groups={leg:{ch:idx.tolist() for ch,idx in channels.items()} for leg,channels in enc.groups.items()},
                reference_angles=None if enc.reference_angles is None else enc.reference_angles.tolist(),
                sensory_reference_qpos=sensory_ref.tolist(), JOINTS=joints,
                geom_leg={str(k):v for k,v in geom_leg.items()},
                encoder_source=str(ROOT/'engine/embodied/sensory_channels.py'),
                encoder_source_sha256=sha(ROOT/'engine/embodied/sensory_channels.py'),
                observation_source=str(ROOT/'run_neural_muscle_closure.py'),
                observation_source_sha256=sha(ROOT/'run_neural_muscle_closure.py'),
                units='native; physical units NOT_ESTABLISHED'))
'''

def prepare():
    s154=inert(S154,'_s162_s154')
    runner,binding,baseline,_=s154.prepare(s154.SELECTION)
    replace=inert(ROOT/'run_half_gain_physical_probe.py','_s162_replace').replace_exact
    prior=runner.private_source
    extra_bound=[Path(__file__).resolve(),S154,REFERENCE/'SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py',
                 REFERENCE/'init80ms/MODEL_PROVENANCE.json',REFERENCE/'init80ms/NATIVE_MODEL_ARRAYS.npz',
                 REFERENCE/'init80ms/INITIALIZATION.npz',REFERENCE/'init80ms/TRAJECTORY.npz']
    bound_names=list(dict.fromkeys([*binding['hashes'],*[str(p) for p in extra_bound]]))
    assert not any(Path(n).name in ('verify_fast_sensory_independent.py','analyze_fast_sensory_comparison.py') for n in bound_names)
    old_observation="""                    observed = observe_quantities(m,d,sensory_ref,joints,geom_leg)
                    current, _ = enc.encode(observed)
                    refresh_times.append(float(d.time))
                    atomic_json(dest/f'OBSERVATION_{tick:02d}.json',raw_observation_record(observed)[0])"""
    def transform(text):
        source,changes=prior(text)
        source,extra=replace(source,[
            ('    identities = {str(s): sha(s) for s in sources}',
             '    sources = list(dict.fromkeys([*sources, *map(Path, '+repr(bound_names)+')]))\n    identities = {str(s): sha(s) for s in sources}'),
            ("    atomic_json(out/'INPUT_HASHES.json', identities)","    atomic_json(out/'INPUT_HASHES.json', identities)\n    s162_freeze_bound(out, identities)"),
            ('            refresh_times = []','            refresh_times = []; sensory_schedule = []'),
            ('                if tick % 10 == 0:','                if tick % 1 == 0:'),
            (old_observation,OBSERVATION),
            ("            initial_observed = observe_quantities(m,d,sensory_ref,joints,geom_leg)",PARAMETERS+"            initial_observed = observe_quantities(m,d,sensory_ref,joints,geom_leg)"),
            ("            atomic_json(dest/'SCHEDULE.json',schedule)","            atomic_json(dest/'SCHEDULE.json',schedule)\n            atomic_json(dest/'SENSORY_SCHEDULE.json',sensory_schedule)"),
            ('                sensory_5ms_refresh_reached=any(t >= .005-1e-12 for t in refresh_times),',
             "                sensory_5ms_refresh_reached=any(t >= .005-1e-12 for t in refresh_times),\n                experiment='S162_FAST_SENSORY', sensory_refresh_period_s=.0005,\n                actual_postrelease_refresh_count=sum(t > 1e-12 for t in refresh_times),\n                legacy_sensory_5ms_field='actual sample time >=5ms, NOT whether any refresh occurred',"),
            ("sensory_rule='live time0 and5ms only if reached; otherwise time0 only NOT closed-loop verification'", "sensory_rule='S162 live current integrated body every .5ms BEFORE neural: time0 then first postrelease .5ms; no hold across ticks', reference_S154="+repr(str(REFERENCE))+", sole_new_intervention='refresh stride10 ->1; S154 model parameters unchanged', hypothesis='before .5ms bitexact S154; later raw/current/spikes/decode may differ but no forced difference; clipping or unmapped quantities can yield raw differences without current differences'"),
        ])
        source=HELPERS+'\n'+source
        ast.parse(source); compile(source,str(runner.ORIGINAL),'exec')
        assert 'if tick % 10' not in source and 'd.ctrl[:] = 0.5 * u' in source
        return source,changes+extra
    runner.private_source=transform
    source,changes=transform(runner.ORIGINAL.read_text())
    # Recompile recorder main metadata only; acquisition stays in private_source.
    tree=ast.parse((ROOT/'run_free_physical_prewarm_probe.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    metadata_main=ast.get_source_segment((ROOT/'run_free_physical_prewarm_probe.py').read_text(),node)
    metadata_main,_=replace(metadata_main,[
        ("protocol='S142 ONE bounded free physical diagnostic; no assistance or activation assignment'", "protocol='S162 ONE S154-matched fast sensory diagnostic', EXPERIMENT='FAST_SENSORY', reference_S154="+repr(str(REFERENCE))+", sensory_refresh_period_s=.0005, sole_new_intervention='sensory stride10 ->1; model unchanged versus S154', preanalysis_hypothesis='before .5ms bitexact S154; later raw/current/spikes/decode may differ, no forced difference', activation_tau_s=.001, deactivation_tau_s=.0004, interface_gain=.5"),
        ("sensory='live observe_quantities at physical0 and5ms only if reached; no replay; hold current between'", "sensory='S162 live observe_quantities every tick before neural, current integrated state at tick*.0005; time0 then .5ms'"),
        ("feedback_acceptance='if stop before5ms, no refreshed physical feedback evidence; not accepted'", "feedback_acceptance='actual postrelease observations recorded; not evidence of autonomous support success'"),
    ])
    exec(compile(metadata_main,str(ROOT/'run_free_physical_prewarm_probe.py'),'exec'),runner.__dict__)
    return runner,source,changes

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='auto'); p.add_argument('--prepare-only',action='store_true'); a=p.parse_args()
    runner,source,changes=prepare()
    if a.prepare_only:
        print(json.dumps(dict(status='INERT_COMPILED',private_sha256=hashlib.sha256(source.encode()).hexdigest(),changes=len(changes),model_loads=0))); return
    out=OUTPUT_PARENT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') if a.output=='auto' else Path(a.output)
    if not out.is_absolute() or out.exists(): raise ValueError('fresh absolute output only; no resume')
    old=sys.argv
    try: sys.argv=[str(Path(__file__).resolve()),'--output',str(out)]; runner.main()
    finally: sys.argv=old
if __name__=='__main__': main()
