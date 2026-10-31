# S142: ONE bounded free full-body physical diagnostic (P1/S2–S4)

## Execution boundary

Parent executes ONCE only after inert preflight. No model loading, forward, step or neural tick has been performed by the implementing agent. No search, second arm, retry, assistance, activation initialization, pose correction or checkpoint replacement is authorized by this source.

Source: `/run/media/sensen/Data2/cell_wound_prototype/run_free_physical_prewarm_probe.py`.

Suggested parent command (fresh real UTC output is chosen automatically):

```bash
cd /run/media/sensen/Data2/cell_wound_prototype
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 ./venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/run_free_physical_prewarm_probe.py
```

Alternatively pass `--output /home/sensen/Desktop/engine修复验收/s142-free-physical-prewarm/<real-current-UTC>`; it must be absolute and not exist. Default parent is `/home/sensen/Desktop/engine修复验收/s142-free-physical-prewarm`. No resume or overwrite.

## Exact design and hard-stop distinction

Uses a private compiled derivative of `/run/media/sensen/Data2/cell_wound_prototype/run_extended_initialization_probe.py`. Original file is never edited. Original setup, full body, source W, initial pose/contact, cold network, PRE160 initialization and zero qvel/act/ctrl assertion remain intact. There is **no S141 activation/ctrl assignment**. PRE160 is 80 ms computational neural-only initialization; it is **not physical prewarm**.

Changes: TICKS6→20 and maximum expected counters to one load/401 explicit forwards/400 physical steps/180 neural ticks; replace fixed-length acquisition/reporting tail with bounded loop, actual-length recording and stop reporting. `SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py`, `EXACT_CHANGES.json`, copies of the wrapper/original/recorder and hashes are archived before model load. Existing original reference integrity checks still run; a failed preflight does not trigger an alternative trial.

Original actual runtime guards are callback absence, zero external applied loads, finite samples and no MuJoCo warnings. Original six benchmark gates were **posthoc**, NOT original early stops. S142 preserves actual guards and explicitly ADDS conservative termination at the first sampled failure: height ratio <.85; root relative angle >.35 rad; fewer than3 contact legs; load ratio <.8; nonfoot depth >1e−12; limit excess >1e−6. Check release sample and every25 µs body sample. No thresholds relaxed and no force-continuation to10ms. The original callback/zero-load assertions remain immediately before each physical step. Exceptions produce FAILED and an emergency state; never auto-retry.

Physics max10 ms, native RK4 step25 µs,20 steps per0.5ms neural tick. Live actual body observation at physical0 and5ms **only if reached**, real encoding/net.step, no replay. Partial final body tick is retained; schedule separately records planned end, actual end and neural source interval. The neural clock can lead physical stop by the remaining fraction of its0.5ms operator-split interval.

Expected S121 baseline first conservative stop is load failure around1.925ms. If so, the5ms refreshed sensory observation is **not reached**; do not report post-refresh feedback, completed10ms prewarm, autonomous stability, walking or accepted physical prewarm. Source always writes physical_prewarm_accepted=false. Actual result, not expectation, determines duration/reason.

## Evidence and units

Under `init80ms`: actual-length `TRAJECTORY.npz`, `CONSTRAINT_ROWS.npz`, constraint/COM/moment metadata, initialization/neural states/connectivity, raw actual observations, `SCHEDULE.json`, `STOP.json`, `STOP_FULL_STATE.npz`, `RESULT.json`. Full existing COM/body/contact recording is retained at every sample; complete constraint row type/id/J/force and contact maps use dynamic sample counts instead of S121 pack_records' fixed121 assertion. Original pure-copy per-record recorder is reused without invoking its old configure/main.

Records qpos/qvel/qacc, act/ctrl, all90 force/length/velocity and transmission moments, all48 generalized force terms, COM/world external and contact forces/moments, native packed mass qM, warmstart and native model arrays. Root vz is explicitly distinct from COM vz. Signed powers: passive·qvel, actuator·qvel, all-constraint·qvel, actual contact-row J.Tforce·qvel, remaining constraint·qvel, and individual actuator_force×actuator_velocity. Whole-body external/contact impulses use trapezoid integration on actual times through stop. Signed powers are **not work or energy**; parent may integrate offline later.

`qfrc_passive` is joint/tendon/fluid aggregate and NOT passive muscle. Native actuator force includes model active/passive muscle terms but this recorder does not separate them. `d.energy` is copied without enabling energy computation; it is NOT a validated energy or elastic potential record. No stored contact/joint/muscle elastic potential is established; no elastic-energy claim. Units remain **native, physical units NOT_ESTABLISHED**. Original fixed121 closure report is deliberately not falsely applied to an early-stop/max401 trajectory: parent must verify saved actual arrays offline. COMPLETE means diagnostic recording completed, not physical or behavioral success.

## File scope and preflight

Only the new source and this README are created. No original source, tracker, frozen historical directory, model, weights or neural/physical parameters are edited. Preflight consists only of AST parsing, private-source transformation checks and inert wrapper import; no actual model/network execution. Once READY is sent, implementing agent freezes these files and stops.
