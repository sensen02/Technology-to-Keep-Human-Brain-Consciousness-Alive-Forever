# Bounded static audit and handling

A read-only independent subagent inspected /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration/PREREG.json, api.py, run.py, verify.py and bounded relevant baseline sections. No physics or imports were executed by the auditor, and it changed no files.

Normal timing and gates found correct:1000 zero ticks+4800 baseline ticks+19200 modified ticks;4800 decisions960+3840; modified begins5.8s, first endpoint5.801, last25.0. The unchanged baseline last-first metric duration is19.199s while the actual physical modified window is19.2s. Original world-X forward.05 and original net-XY slip denominator retained; halfspeed forward.025 separately; strictNF0/COM/support/slip/all6 cycles as declared. Explicit baseline lineage and teacher path, smoothstep schedule, yaw boundary and cycle replay align.

Three concrete limitations and handling:

1. Frozen api.py original sample loader checks current RESULT success but hashes only samples. Preferred separate /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration/curriculum.py authenticates both RESULT and SAMPLES hashes and requires independently recomputed halfspeed success. Parent must use this preferred loader. api.py remains frozen.
2. Frozen run.py lacks finally persistence for arbitrary teacher/physics/nonfinite exceptions, and pre-prefix early stop cannot emit normal RESULT because modified boundary values are unavailable. If triggered, retain existing artifacts/log and report incomplete evidence without retry. This does not affect successful normal completion; final report must explicitly record whether triggered.
3. Frozen verify.py initial alignment checks one action tick per decision. Separate /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/deceleration/verify_alignment.py checks the entire modified+prefix action trace against repeat(Y42,5), previous actual action in X238, all4800 sample rows/full-body shapes, and original halfspeed-left first15.4s bitexact correspondence. No original or currently running source edits are required.

This audit is engineering review, not a physical gate result. Final numerical verification is separate.
