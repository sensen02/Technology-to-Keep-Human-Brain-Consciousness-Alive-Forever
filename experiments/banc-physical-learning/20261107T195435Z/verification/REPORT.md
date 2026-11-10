# Independent final selected and zero verification

Both final neural replays passed. The selected policy is the **zero incumbent**, not a newly learned nonzero walking controller. Selection produced no improvement; support-gate passage must not be presented as learned walking.

Verification workspace: `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/verification`.

Desktop delivery: `/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/verification`.

## Replay scope and result

The verifier imports only the frozen192931Z conductance core and frozen185910Z physical encoder. It does not import the episode runner, foundation actor/restore helpers, scoring function or a physics engine. Complete NEURAL_INITIAL state is checked against the canonical foundation snapshot, then assigned manually: full voltage, conductances, g_adapt, refractory, counts, ring/index, lastspikes, filter, clock and identities. Start is2000steps/1000ms; each saved release is1000policybins×5ms, ending12000steps/6000ms.

Every start-of-bin323MN rate/voltage, reconstructed556sensory current, same-leg basis×theta clipped weight, actor output and five held1ms command rows matched exactly. Inputs were checked against BODY joint coordinates/velocities and NATIVE footloads; firstcontact matched prior192931Z NATIVE row79999 exactly, so no scratchforward was needed. DN input remained exactlyzero. All final state fields and source identities/configuration matched exactly.

| Condition | Completed replay steps | MN spikes since checkpoint | Every-step minimum mV | Original support gate | Recomputed reward | Completed leg cycles |
|---|---:|---:|---:|---|---:|---:|
| selected | 10000 | 26225 | -76.6114764878 | True | -0.12593044586684476 | 0 |
| zero | 10000 | 26225 | -76.6114764878 | True | -0.12593044586684476 | 0 |

Voltage was recorded every0.5ms rather than only at saved5ms endpoints; every-step bounds passed the declared engineering range and derived−80mV lower bound. Full perstep min/max/gAmax/spikecounts and replayed finalstates are saved in this verification directory. The conductance/receptor/encoder mechanism remains illustrative and uncalibrated.

## Equality and saved scoring

Selected/zero equality results:

```json
{
  "theta": true,
  "weights": true,
  "policy_all_arrays": true,
  "body_all_arrays": true
}
```

`CONDITION_EQUALITY.json` additionally confirms field-by-field equality for all10 source NPZ artifacts (actor, physical initial, neural initial/final, policy, body, native, commands, slip and score inputs), plus both independently replayed finalstates and every-step histories. This comparison executed zero additional neural or physical steps.

The reported physical D gates, rewardterms and stance-air-stance cycle counts were independently recomputed from saved native arrays with the original criteria. This checks reporting arithmetic; it does not independently integrate physical dynamics. Parent separately handles physical replay and rendering. Passing the support gates or observing ongoing neuronal spikes does not establish coordinated locomotion or a necessary neural contribution to support when actoroutput iszero.

## Accounting and provenance

Completed neural replay steps: **20000** (10000 selected +10000 zero). Physical steps: **0**. Scratchforward calls: **0**. Optimizer updates: **0**. No fitting, candidate selection, new brain training or surgical intervention.

`PREREGISTRATION.json` and the checker were prepared before the final artifacts existed; no artifact polling or replay occurred before the parent's explicit readiness notice. Execution used existing `/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B` with `verify_evaluations.py --parent-artifacts-ready`. Input/source hashes are in `VERIFICATION.json`; all were rechecked unchanged. The final mirror manifest covers every owned artifact and byte-verifies both locations. Original models, learning/foundation files and tracked project files remain untouched by this task.
