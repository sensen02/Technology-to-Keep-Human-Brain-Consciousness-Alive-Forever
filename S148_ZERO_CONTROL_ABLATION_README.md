# S148/P1 zero-control interface ablation — bounded diagnostic, not a candidate

Runner: `/run/media/sensen/Data2/cell_wound_prototype/run_zero_control_ablation_probe.py`.

Only the parent executes ONE actual run, after READY. This implementation agent performed only inert source/AST/filesystem validation, no MuJoCo import/model/forward/step/neural execution.

Parent command:

```sh
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/run_zero_control_ablation_probe.py --selection /home/sensen/Desktop/engine修复验收/s145-secant-contact-preload/20261024T203438782961Z --output auto
```

Fresh real-UTC output beneath `/home/sensen/Desktop/engine修复验收/s148-zero-control-ablation`. No resume, no overwrite. All original files and tracker remain untouched by this implementation.

Same S146 independently proven root-z initialization and original sensory reference, actual new-pose initial observation, genuine PRE160 neural/FIR rerun, same weights/model/contact/passive/position/velocity muscle terms and zero initial qvel/act/ctrl/external loads. No post-initial act assignment or pose correction. The sole interface intervention replaces physical `d.ctrl[:] = u` with `d.ctrl[:] = np.zeros_like(u)`. Tick `decoded` remains raw genuine neural output; tick `applied` is zero; sampled native ctrl/act are asserted zero. Passive muscle force need not be zero.

Maximum ten 25us steps = .00025s, eleven samples, at most one post-init genuine .5ms neural tick. Native dt/substeps remain 25us/20; only the physical inner-loop length is bounded to ten. Actual executed counts remain ledger-derived, including earlier failure. A partial physical tick is NOT relabeled: its neural/FIR interval ends at source80.5ms while physical may end at .25ms or earlier. PRE160 is computational initialization, not physical prewarm.

Original six support gates become conservative first-failure stops exactly as S142; original root-height reference is unshifted, never rebaselined to S146 root height. Load below .8 stops immediately with the failing sample retained. Original hard guards, every-forward native capture, all48 dynamics, actual constraint rows, signed powers, contact data, native model arrays and stop full-state capture are inherited intact. S142 wrapper metadata is privately overridden to .25ms/10/11/1 and [0,.00025], not 10ms or four ticks.

Before model loading the inherited SOURCE_FREEZE archive records exact count-checked transforms, private runner source hash, source identities, and S145 proof/artifact/runner copies including the S148 runner. PREREG declares EXPERIMENT=ZERO_CTRL_ABLATION, candidate=false, S119_same_condition_candidate=false, physical_prewarm_accepted=false; it binds S146 full-state trajectory, initialization, model provenance chain and private runner identities. S146 comparison files are provenance/context, not replay inputs. Independent parent verification must establish same initial full state and neural initialization, raw decoded versus zero applied, all actual counts and original gates.

Interpretation: source records zero act by design, not an activation fix or successful behavior. No training/checkpoint change, no autonomous walking/standing/prewarm claim; physical units remain NOT_ESTABLISHED.
