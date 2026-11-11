# Restricted range diagnostic: four fixed profiles

This directory alone receives all new outputs. Frozen parent root: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/`.

`PREREG.json` was written before managed pipeline launch. Exactly halfleft, halfright, left_to_right and right_to_left run once each. The previously executed same-checkpoint neutral actual trajectory is reused as control; no redundant neutral simulation occurs. Prior fullrange prerequisite is explicitly waived only for diagnostic mapping after observed full±1 slip failures. No deployment or food qualification is inferred.

`run_one.py` calls unchanged prepared `residual_profiles/evaluate_residual_profile.py:prepare()` and adds another explicit pin hook. That hook changes only captured new metadata/source paths and recursive NumPy scalar JSON conversion. Actual original evaluator, physical inputs, finitePD, source sequence, actor calculations and scoring remain unchanged. `pipeline.py` launches each physical subprocess and collects its exit, preserves logs and invokes saved verification once after four complete runs.

The autonomous window remains9.6s after1ssettle+4.8s actual originalteacher neutralprime. Native timestep12.5us, actual target integration1ms, policy5ms. Original five whole-window engineering gates, all six sensed cycles and existing signed-yaw criteria remain unchanged. Halfprofiles require whole signed>=.005rad, switching requires each4.8s signed>=.002rad including transient. Test command schedule is neural input only, never script-generated joint motion.

`verify_saved.py` reads each archive ONCE into bounded arrays, independently recomputes unchanged gates/cycle events/yaw and replays all2880 actual observations through the checkpoint with explicit dual192hidden state. It checks15400 endpoints, exact source/action alignment, whole5800physical prefix versus previously verified neutral, hidden chains, command schedule, hashes, force zeros, captured new prereg and compiledruntime identity. No physical integration or optimization is used during verification.

Outputs perprofile: `IDENTITY.json`, `trace_1ms.npz`, `neural_decisions.npz`, `geometry_1ms.npz`, `RESULT.json`. Final `VERIFICATION.json` and `ALL4_RESULT.json` report all results including failures. These are1ms endpoint engineering diagnostics of actual12.5us physics, not full native D1–D8 certification, biological validation, training or food approach.
