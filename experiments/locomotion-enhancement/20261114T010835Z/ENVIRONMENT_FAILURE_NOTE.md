# Execution failure record

The original target-feedback pipeline was started once with `/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python`. Its preflight was separately verified with the qualified body environment and passed 72 checks, but the training child failed before any numerical work because that execution environment lacked Torch:

`ModuleNotFoundError: No module named 'torch'`

Failure occurred in `train.py` during `cpu1()` before model loading, array loading, optimizer creation, forward, physical/Core execution, or training. The original `JOBS.json`, `PIPELINE.log`, and stage log are retained and the original pipeline is not retried or overwritten. A separate recovery namespace will rerun the already-qualified finite candidate using the existing qualified environment:

`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`

This is an environment recovery, not a scientific/model change or additional training budget.
