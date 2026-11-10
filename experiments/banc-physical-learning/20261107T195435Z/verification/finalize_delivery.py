"""Finalize verified selected/zero replay report and scoped Desktop mirror."""
from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent
DEST=Path('/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/verification')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    report=json.loads((HERE/'VERIFICATION.json').read_text())
    if report['status']!='PASS_INDEPENDENT_SELECTED_ZERO_REPLAY':raise RuntimeError('cannot publish successful verification before PASS')
    s,z=report['conditions']['selected'],report['conditions']['zero']
    text=f'''# Independent final selected and zero verification

Both final neural replays passed. The selected policy is the **zero incumbent**, not a newly learned nonzero walking controller. Selection produced no improvement; support-gate passage must not be presented as learned walking.

Verification workspace: `{HERE}`.

Desktop delivery: `{DEST}`.

## Replay scope and result

The verifier imports only the frozen192931Z conductance core and frozen185910Z physical encoder. It does not import the episode runner, foundation actor/restore helpers, scoring function or a physics engine. Complete NEURAL_INITIAL state is checked against the canonical foundation snapshot, then assigned manually: full voltage, conductances, g_adapt, refractory, counts, ring/index, lastspikes, filter, clock and identities. Start is2000steps/1000ms; each saved release is1000policybins×5ms, ending12000steps/6000ms.

Every start-of-bin323MN rate/voltage, reconstructed556sensory current, same-leg basis×theta clipped weight, actor output and five held1ms command rows matched exactly. Inputs were checked against BODY joint coordinates/velocities and NATIVE footloads; firstcontact matched prior192931Z NATIVE row79999 exactly, so no scratchforward was needed. DN input remained exactlyzero. All final state fields and source identities/configuration matched exactly.

| Condition | Completed replay steps | MN spikes since checkpoint | Every-step minimum mV | Original support gate | Recomputed reward | Completed leg cycles |
|---|---:|---:|---:|---|---:|---:|
| selected | {s['neural_steps']} | {s['MN_spikes_since_checkpoint']} | {s['voltage_min_every_step_mV']:.12g} | {s['physical_gate_pass']} | {s['reward_recomputed']:.17g} | {s['completed_cycle_legs']} |
| zero | {z['neural_steps']} | {z['MN_spikes_since_checkpoint']} | {z['voltage_min_every_step_mV']:.12g} | {z['physical_gate_pass']} | {z['reward_recomputed']:.17g} | {z['completed_cycle_legs']} |

Voltage was recorded every0.5ms rather than only at saved5ms endpoints; every-step bounds passed the declared engineering range and derived−80mV lower bound. Full perstep min/max/gAmax/spikecounts and replayed finalstates are saved in this verification directory. The conductance/receptor/encoder mechanism remains illustrative and uncalibrated.

## Equality and saved scoring

Selected/zero equality results:

```json
{json.dumps(report['selected_equals_zero'],indent=2)}
```

`CONDITION_EQUALITY.json` additionally confirms field-by-field equality for all10 source NPZ artifacts (actor, physical initial, neural initial/final, policy, body, native, commands, slip and score inputs), plus both independently replayed finalstates and every-step histories. This comparison executed zero additional neural or physical steps.

The reported physical D gates, rewardterms and stance-air-stance cycle counts were independently recomputed from saved native arrays with the original criteria. This checks reporting arithmetic; it does not independently integrate physical dynamics. Parent separately handles physical replay and rendering. Passing the support gates or observing ongoing neuronal spikes does not establish coordinated locomotion or a necessary neural contribution to support when actoroutput iszero.

## Accounting and provenance

Completed neural replay steps: **{report['completed_neural_steps']}** (10000 selected +10000 zero). Physical steps: **{report['physical_steps']}**. Scratchforward calls: **{report['scratch_forward_calls']}**. Optimizer updates: **{report['optimizer_updates']}**. No fitting, candidate selection, new brain training or surgical intervention.

`PREREGISTRATION.json` and the checker were prepared before the final artifacts existed; no artifact polling or replay occurred before the parent's explicit readiness notice. Execution used existing `/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B` with `verify_evaluations.py --parent-artifacts-ready`. Input/source hashes are in `VERIFICATION.json`; all were rechecked unchanged. The final mirror manifest covers every owned artifact and byte-verifies both locations. Original models, learning/foundation files and tracked project files remain untouched by this task.
'''
    (HERE/'REPORT.md').write_text(text)
    sources=dict(report['source_hashes'])
    for condition in report['conditions'].values():
        sources.update(condition['input_hashes'])
        protocol_path=next(Path(p) for p in condition['input_hashes'] if p.endswith('/PROTOCOL.json'))
        sources.update(json.loads(protocol_path.read_text())['source_hashes'])
    assert all(sha(Path(p))==h for p,h in sources.items())
    execution={'status':'PASS','completed_neural_steps':20000,'physical_steps':0,'scratch_forward_calls':0,'optimizer_updates':0,'unique_input_source_hashes_verified':len(sources),'hashes':sources,'checker_sha256':sha(HERE/'verify_evaluations.py'),'equality_checker_sha256':sha(HERE/'compare_conditions.py'),'replay_job':'bash-221 exit0','all_condition_arrays_identical':True,'selected_policy':'unchangedzero incumbent;no learnednonzero walking','no_failures':True}
    (HERE/'EXECUTION_RECORD.json').write_text(json.dumps(execution,indent=2)+'\n')
    files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='DELIVERY_MANIFEST.json'}
    manifest={'workspace':str(HERE),'desktop':str(DEST),'files':files,'neural_steps':report['completed_neural_steps'],'physical_steps':0,'scratch_forward_calls':report['scratch_forward_calls'],'optimizer_updates':0}
    (HERE/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    DEST.mkdir(parents=True,exist_ok=True)
    for p in HERE.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    assert all(sha(HERE/n)==sha(DEST/n) for n in list(files)+['DELIVERY_MANIFEST.json'])
    print(json.dumps({'files_byte_verified':len(files)+1,'total_bytes':sum(v['bytes'] for v in files.values()),'completed_neural_steps':report['completed_neural_steps'],'desktop':str(DEST),'report_sha256':sha(HERE/'REPORT.md'),'verification_sha256':sha(HERE/'VERIFICATION.json')},indent=2))
if __name__=='__main__':main()
