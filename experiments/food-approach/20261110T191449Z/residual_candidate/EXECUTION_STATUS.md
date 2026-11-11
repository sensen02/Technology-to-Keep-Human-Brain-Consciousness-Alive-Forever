# Authorized fixed500 residual candidate execution

Parent agent session-80ccae55-a957-4374-b48e-7ba52c0ab34c authorized run after measured conditional finetune500 neutral slip0.54195384 and left slip0.59382492 failed original0.5. Left yaw+0.05975475 showed correct turn but not joint acceptance. No body limitation inferred.

Frozen architecture and original offline checks unchanged. New sources cached_training.py, check_cache.py, verify_saved.py, evaluate_candidate.py, run_fixed500.py plus TRAIN_PREREG.json and RUN_AUTHORIZATION.json are confined here. CACHE_CHECKS.json11/11 passed full2880 production/cache means/bothstates and fullBPTT gradient parity, no optimization in this check. A preliminary identity check failed when evaluator was edited during initial offline cache test; final cache test was rerun after evaluator completed, all current source hashes verified before actual optimization.

Managed child job bash-196 runs fixed500 CPU1 Adam.001 adapteronly, followed by only500 neutral/left/right realphysics. Log /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate/TRAIN_FIXED500.log. Parent notified, child collects. No behavior training, no checkpoint shopping or extension.

API for later behavior integration: evaluate_candidate.load_checked(checkpoint,sha256) -> model,payload with strict500 and source/data checks; model.step(obsB238,commandB2,(baseB128,residualB64)) -> meanB42,(baseB128,residualB64); deterministic action=tanh(mean). Physical runner stores concatenated192 for audit but model accepts explicit tuple. Keep both states continuous across handover.

Status at writing: running; final acceptance not established. FIXED500_RESULTS.json will record all3 candidate gates when actual runs finish. Trajectory fit does not certify turning.
