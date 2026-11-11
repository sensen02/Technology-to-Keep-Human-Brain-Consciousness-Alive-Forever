# Conditional turning/deceleration teacher diagnostic

This is a bounded algorithm-teacher demonstration for the first food-odor approach round. It does not train an actor, load the verified actor checkpoint, implement a food field, or establish neural autonomy. The verified walking baseline at `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/` was imported by absolute path and asserted at runtime.

`PREREG.json` fixed four configurations before native execution: +0.008/-0.008 requested yaw at full speed and +0.004/-0.004 at half speed. Each run used 1 s zero settle, 4.8 s unchanged neutral teacher priming, then 9.6 s smoothly ramped conditional teacher with no reset. The body-frame twist is implemented in `turning_teacher.py`: it changes foot targets through the measured target reference and finite IK, while the root remains free for MuJoCo dynamics. `Core(adhesion=False)` retained the verified 12.5 µs physics, gravity, friction, 42 finite PD actuators, limits and contact geometry. No root trajectory writes or extra forces were used.

`bash-178` completed with exit code 0. `verify_results.py` independently passed all saved-evidence checks: all four traces are complete 15.4 s runs (15,400 one-ms endpoints; 2,880 five-ms decisions), finite, command-bounded, zero applied body/generalized external force, exact neutral-prefix parity, unchanged actor observation target speed `.075`, matching saved hashes, and exact recomputation of the original gates and slip. `REFERENCE_CHECKS.json` also passed neutral bitexact parity, mirror symmetry, and velocity derivative checks.

| condition | actual signed yaw (rad) | actual mean yaw rate | COM displacement (native x,y,z) | slip max | forward signed speed | original five gates | nonfoot |
|---|---:|---:|---|---:|---:|---|---:|
| left_full (+.008) | +0.0612837993 | +0.0063837291 | (0.68755338, -0.00347117, -0.00018072) | 0.4747733814 | 0.0716197355 | PASS | 0 |
| right_full (-.008) | -0.0785123015 | -0.0081783647 | (0.69253934, +0.06653822, -0.00007243) | 0.4482804924 | 0.0721391605 | PASS | 0 |
| left_half (+.004) | +0.0310407290 | +0.0032334093 | (0.35927488, -0.01540584, -0.00729558) | 0.4977455399 | 0.0374204963 | FAIL forward only | 0 |
| right_half (-.004) | -0.0423243131 | -0.0044087826 | (0.36131121, +0.03264853, -0.00711003) | 0.4741920121 | 0.0376326354 | FAIL forward only | 0 |

All four runs had requested-sign agreement, no early stop, foot support mean about 1.00002–1.00005, COM minimum 1.25596–1.25871, and strict nonfoot force zero. The two half-speed runs honestly fail the unchanged signed forward-speed gate of 0.05 because their achieved forward speed is about 0.0374; they are still useful bounded deceleration/turning demonstrations. These are 1 ms endpoint engineering summaries, not a native D1–D8 audit and not a claim of full walking acceptance.

Training API for the parent:

```python
from turning_teacher import pin_baseline, make_teacher
pin_baseline()
from physics import Core
c = Core(adhesion=False)
teacher = make_teacher(c, speed_scale=1.0, yaw_rate=0.008, prime_s=4.8)
# after the parent's 1 s physical settle, t is elapsed teacher time:
a42, diag = teacher.action(c, t)
```

The actual training arrays are in each condition directory as `SAMPLES_5MS.npz`: `X238` (fixed `.075` target-speed observation), `Y42`, `command2` normalized as `[speed/.075 - 1, requested_yaw/.008]`, physical command diagnostics, provenance phase and actual time. The first 960 rows are neutral priming and the next 1,920 rows are conditional. Full native-integrated evidence is in `TRACE_1MS.npz`, with signed thorax yaw in `YAW_1MS.npz` and per-condition `RESULT.json`.
