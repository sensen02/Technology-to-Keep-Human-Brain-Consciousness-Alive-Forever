# Hair plate literature report (condensed from the research subagent, verbatim numbers)

**Status: MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT.** This file records the published numbers the
project now cites. Full text of the subagent's report is in the session transcript; the numbers,
ranges and URLs below are copied from it, including its own caveats about what it digitised itself.

## Bottom line

The project's declared `reference_angle_rad = 0.30` is **not supported by any published number**.
The only fly-specific quantitative hair-plate study (Pratt et al. 2026) measures the coxa hair plate
**CxHP8** over a response span of **~149° = 2.60 rad** of coxa rotation, and the largest single-plate
dynamic range in the classic literature (cockroach scapal hair plate, Okada & Toh 2001) is
**~38–45° = 0.66–0.79 rad**. 0.30 rad is 2.2–8.7x smaller than anything published.

Two further published statements bear directly on the project's channel construction:

1. **Every published hair-plate tuning curve is for ONE joint** (fly: coxa rotation/adduction;
   cockroach: antennal scape; locust: thorax-coxa or femur-tibia). **No published receptor integrates
   multiple joints**, so the project's "mean angular deviation over six hinges" is not a hair-plate
   quantity. The two BANC `body_part_sensory` values for hair-plate cells are `coxa,<leg>` and
   `<leg>,trochanter`, i.e. the dataset agrees the attachment is the coxa or the trochanter.
2. **The published posture-reference signal of the fly leg is the FeCO *claw* population**, tonically
   active as a deviation from a femur-tibia angle of ~80 deg (Dallmann et al. 2025) / ~90 deg static
   offset (Mamiya et al. 2018) -- **not** a hair plate. The fly's tuned hair plate behaves as a
   **limit detector at the extreme of the range**.

## Numbers

| Receptor (paper) | Joint | Stimulus | Range | Threshold | Gain | Source |
|---|---|---|---|---|---|---|
| CxHP8 coxa hair plate, Drosophila (Pratt 2026) | thorax-coxa | coxa rotation (anterior<->posterior) | rise ~80 deg -> peak ~180-200 deg, decline to 229 deg; **span ~149 deg = 2.601 rad** | ~80-90 deg (Ca flat below) | no spike-rate gain published | Nat Commun 17:2664, doi:10.1038/s41467-026-69333-z |
| CxHP8 (same) | thorax-coxa | adduction | -30..+47 deg, rise from ~10 deg | ~10 deg | none published | same |
| CxHP8 (same) | thorax-coxa | flexion | 0-54 deg, weak | ~5-10 deg | none published | same |
| CxHP8 joint ranges (same, Supp S1) | thorax-coxa | joint definition | rotation [30, 229] deg; adduction [-43, 47] deg; flexion [0, 54] deg | - | - | same |
| Cockroach lateral scapal hair plate (Okada & Toh 2001) | antennal scape | hair deflection, ramp-and-hold | hold 15/23/34/46/60 deg; tonic span ~34->60 deg = **0.59-0.79 rad** | **tonic response needs >=34 deg (0.593 rad)**; 15/23 deg phasic only | ~2.45-2.76 Hz/deg (agent's arithmetic on re-plotted points, not stated in the paper) | J Exp Biol 204:4301, doi:10.1242/jeb.204.24.4301; numbers via Biol Cybern 120:6, doi:10.1007/s00422-025-01032-2 |
| Hair-field model unit (van der Veen 2026) | insect leg joints | joint angle -> hair deflection -> rate | assumes hair fields sample the FULL joint working range; 90 deg illustrative | - | linear by construction | doi:10.1007/s00422-025-01032-2 |
| FeCO claw neurons (Mamiya 2018) | femur-tibia | tibia angle, ramp-and-hold | **18-180 deg (162 deg = 2.827 rad) tested**; natural walking excursion **60-80 deg (1.05-1.40 rad)** | static offset ~90 deg | no spike gain (calcium only) | Neuron 100:636, doi:10.1016/j.neuron.2018.09.009 |
| FeCO claw neurons (Dallmann 2025) | femur-tibia | joint angle as a deviation | "encode the position of the femur-tibia joint as a **deviation from a joint angle of ~80 deg**" | - | population activity rises non-linearly with flexion or extension | Nature 647:445, doi:10.1038/s41586-025-09554-2 |
| FeCO hook neurons (Mamiya 2018) | femur-tibia | ramp velocity | **100-800 deg/s tested** (= 1.745-13.963 rad/s); up to 1440 deg/s for some stimuli | never tonic during hold | no spike gain published; calcium slope flat across 100-800 deg/s | Neuron 100:636, doi:10.1016/j.neuron.2018.09.009 |
| FeCO club neurons (Mamiya 2018) | femur-tibia | vibration | **0.9 and 0.054 um, 100-2000 Hz**, peaks 400 Hz (0.9 um) / 800 Hz (0.054 um) | responds at the smallest tested amplitude | no gain published; spatial map of frequency | same |
| Fly leg hair-plate census (Pratt 2026) | all legs | - | **214 hair-plate neurons in 42 plates across six legs** | - | - | Nat Commun 17:2664 |
| Per-leg hair-plate counts (FBbt, Kuan 2020) | prothoracic leg | - | CoHP3 3, CoHP4 4, CoHP8 8 (coxa = 15); TrHP1 1, TrHP5 5, TrHP6 6, TrHP7 7 (trochanter = 19); TiHP3 3 (tibia) | - | - | FBbt:00058079/00058080/00058081, 00052583/00058085/00058086/00058087, 00058077 |
| Axon diameters (Kuan 2020) | - | - | CoHP8 1030+-90 nm, CoHP4 1140+-240 nm, CoHP3 1380+-20 nm; TrCS large axons 850-1050 nm; motor neurons 1-2 um; chordotonal and bristle axons narrower | - | - | Nat Neurosci 23:1637, doi:10.1038/s41593-020-0704-9 |

## What this does NOT support

- The 0.30 rad normaliser: no published range, threshold or gain corresponds to it.
- Treating a hair plate as a multi-joint integrator.
- Keying `hair_plate_neuron` to the femur-tibia joint: Drosophila coxal hair plates encode **coxa**
  rotation and adduction, and the dataset's own attachment field says coxa or trochanter.
- Claiming a hair plate supplies the posture reference; the published posture signal is the FeCO claw
  population's tonic deviation from ~80 deg.
- Any conversion from calcium imaging to a spike rate: **no published spike gain exists for any
  Drosophila hair plate**, and the only quantitative hair-plate dose-response in the literature is in
  degrees of **hair deflection**, with the hair-angle-to-joint-angle transfer function unmeasured.
- Some of the numbers above were digitised by the agent from published figures and are marked as such;
  they are provisional unless the source data are obtained.
