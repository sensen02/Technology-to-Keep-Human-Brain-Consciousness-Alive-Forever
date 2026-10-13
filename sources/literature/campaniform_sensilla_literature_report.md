# Leg campaniform sensilla: what they transduce, over what range, and in what units

Literature research report for the BANC `campaniform_sensillum_neuron` input channel.
All DOIs below were resolved via Crossref. Numbers are quoted only from papers whose **full text** was opened; where only an abstract was reachable, that is stated explicitly.

---

## 1. Summary table

`n.r.` = not reported as a number in the accessible text. `/` = not applicable to that entry.

| Cell group / species | Physical stimulus | Range with units | Threshold | Rate-vs-stimulus relation | Source |
|---|---|---|---|---|---|
| **Blow fly** *Calliphora vicina*, hind-leg **tibial group** (AS3/AS4, 2 sensilla, caps ⟂ tibial long axis) | Applied force to tibia with joint movement resisted (ramp-and-hold, half-sine, exponential) | Tonic firing increases **linearly up to ≈0.3 mN**; mean body weight of the animal = **0.29 mN** | Firing consistently reflects amplitude **above 0.05 mN** | Tonic (amplitude) *and* rate-sensitive: `r² = 0.93`, n = 191 tests, N = 3 animals. Strong inhibition by small transient force *decreases*; hysteresis | [Zill et al. 2025, *J Neurophysiol* 133:1749–1760](https://doi.org/10.1152/jn.00044.2025) |
| **Stick insect** *Carausius morosus*, **tibial group 6B** (distal, round caps ~1.0 mm distal to FT joint) | Torque/force at the femoro-tibial joint in the plane of joint movement; also to tibial muscle contractions | Tested peak forces **0.57–2.8 mN**; ramp-and-hold offsets **0, 2.9, 5.4 mN** | no numeric threshold published; large 6B units adapt to **<10 Hz** during hold at low offset | Firing ∝ **+dF/dt**; regression slopes 8.1 (large) and 8.2 (all 6B) AP/s per unit dF/dt; **r² = 0.98** (n pooled); per-amplitude slopes 24.3 / 18.3 / 12.3 / 11.2 / 9.9 | [Zill et al. 2021, *J Neurophysiol* 126:227–248](https://doi.org/10.1152/jn.00120.2021); [Zill et al. 2024, *J Neurophysiol* 131:198–215](https://doi.org/10.1152/jn.00414.2023); [Harris et al. 2022, *J Neurophysiol* 128:790–807](https://doi.org/10.1152/jn.00285.2022) |
| **Stick insect** *C. morosus*, **tibial group 6A** (proximal, oval caps ~0.3 mm distal to FT joint) | Same stimuli; fires to **force decreases** | Same stimulus range as 6B | "higher threshold" for the *opposite* direction (stated qualitatively, no number) | Discharge ∝ **−dF/dt**; discharge falls as offset load rises (slope −5.95) | [Zill et al. 2011, *J Comp Physiol A* 197:851–867](https://doi.org/10.1007/s00359-011-0647-4); [Harris et al. 2022](https://doi.org/10.1152/jn.00285.2022) |
| **Stick insect** *C. morosus*, **trochanteral groups 3 & 4** (dorsal, coxo-trochanteral joint) | Dorsal load in the plane of CTr joint movement; also to **depressor muscle contractions** and forces at the muscle insertion | Loads emulating walking; torques derived from freely walking animals | Lowest torque / torque-rate at which G3/G4 activity was *confirmed*: **3.4 µNm** and **10.5 µNm/s** (trochanter–femur lever 11.7 mm) | Encodes **both amplitude and rate** of loading *and* of muscle force | [Zill et al. 2012, *J Neurophysiol* 108:1453–1472](https://doi.org/10.1152/jn.00274.2012); [Dallmann et al. 2017, *Proc R Soc B* 284:20171755](https://doi.org/10.1098/rspb.2017.1755) |
| **Cockroach** *Periplaneta americana*, **proximal tibial group** | Force applied to tibia (ramp-and-hold) | Tested mean amplitude **0.85 ± 0.42 mN**; hold offsets **0.92, 1.84, 2.45 mN** | n.r. as a number | Tonic: hold-phase firing closely reflects offset magnitude; strong phasic response during ramp | [Harris et al. 2022, *J Neurophysiol* 128:790–807](https://doi.org/10.1152/jn.00285.2022) |
| **Cockroach** *P. americana*, **distal tibial group** | Same | Same | Effectively silenced by even small sustained loads: slope **−13.7** | Fires phasically to **force decreases** | [Harris et al. 2022](https://doi.org/10.1152/jn.00285.2022) |
| **Cockroach**, **trochanteral CS** | Cuticular strain from resisted leg-muscle contractions | — | — | Larger caps recruited as force increases → **amplitude-based range fractionation** | [Zill et al. 2017, *Arthropod Struct Dev* 46:564–578](https://doi.org/10.1016/j.asd.2017.05.004); mechanism attributed to Vincent et al. in [Saltin et al. 2025](https://doi.org/10.1098/rsif.2024.0559) |
| **Cockroach**, **femoral CS** (7–11 caps in a row, 200–500 µm distal to TrF joint) | Strain; cap long axis nearly parallel to TrF joint (mean −0.45° ± 6.8°) | Cap long-axis length **11.9 ± 4.9 µm** (n = 20, N = 3) | n.r. | Multiunit firing reflects amplitude of sustained force | [Zill et al. 2017](https://doi.org/10.1016/j.asd.2017.05.004) |
| **Drosophila**, **femoral CS field** (11 caps at **2.5 % of femur length** from the TrF joint) | **FE model output, not measurement.** Simulated GRF from forward walking | Displacement at the NompC transduction site **stays below the 7 Å thermal-noise limit**; femur bends **<2°**. Needed amplification to cross 7 Å: **×1.84** (proximal–distal axis), **×1.57** (dorsal–ventral), **×3.05** (left–right) | **7 Å** displacement of the NompC channel (thermal noise floor, from ref. 40 of that paper) | Model conclusion: GRF alone is unlikely to activate this field; strain patterns vary only coarsely with force direction | [Saltin et al. 2025, *J R Soc Interface* 22:20240559](https://doi.org/10.1098/rsif.2024.0559) |
| **Drosophila**, proximal leg (robotic dynamically-scaled leg) | Strain gauges at TrF/TrG positions | Recorded in **microstrain** and **microstrain/s**, plotted vs. step phase; **no numeric values given in the text** | n.r. | Peak strain at anterior extreme position; rate shows a single positive peak at AEP | [Dinges et al. 2023, *Living Machines / LNCS* — NSF PAR 10464276](https://par.nsf.gov/servlets/purl/10464276) |

---

## Q1. What leg CS transduce, and which effect they measure

**They transduce cuticular strain, not force directly, and they measure the local strain rate and its history rather than a static load.**

CS are campaniform caps set into the cuticle, with the sensory dendrite attached to the underside of the cap; external forces that are *resisted by the muscles* bend the exoskeleton and displace the cap, which compresses the neuron ([Saltin et al. 2025, *J R Soc Interface*](https://doi.org/10.1098/rsif.2024.0559), Introduction). So the chain is **ground reaction force / muscle force → cuticular strain → cap displacement → spike**. Crucially, the CS are not measuring the force you apply to the foot; they measure the strain where they sit.

**Which effect.** Across every system where this was tested directly, CS are **phasic-tonic**, and the tonic (amplitude) component is generally weaker than the phasic (rate) one:

- Stick insect tibial 6B: all sensilla show "directional, **phasico-tonic** responses", and "all receptors accurately encode **the rate of change** of force increments and decrements" (abstract wording implying phasic-tonic; [Zill et al. 2011](https://doi.org/10.1007/s00359-011-0647-4)). In walking-torque tests the same group's firing follows **+dF/dt** with r² = 0.98, while firing to force *decreases* is largely abolished ([Zill et al. 2021](https://doi.org/10.1152/jn.00120.2021)).
- **Explicit counter-evidence to pure load encoding:** "The tibial sensilla **do not directly convey the magnitude of the Newtonian force** or reflect the body weight (as, for example, a bathroom weighing scale)" ([Zill et al. 2024, Discussion](https://doi.org/10.1152/jn.00414.2023)).
- Blow fly tibial group is different in balance: it shows **sustained, strongly tonic** firing even at very low forces, *plus* high rate sensitivity; response to rate "is substantially higher than the response to the amplitude of the load" ([Zill et al. 2025](https://doi.org/10.1152/jn.00044.2025)).

**Subtypes and signed directions.** Within one anatomical group there are routinely two antagonistic subgroups that fire to opposite force directions, giving a *signed* representation of force change:

- Tibia: 6B fires to force **increases** in one direction, 6A to **decreases**; directions reverse when the applied force reverses ([Zill et al. 2011](https://doi.org/10.1007/s00359-011-0647-4)).
- Trochanter: group 3 fires to dorsal loading increase / ventral decrease; group 4 has the **reverse** directional sensitivities; together they form "an array that can detect forces in **all directions**" ([Zill et al. 2012](https://doi.org/10.1152/jn.00274.2012)).
- Cockroach tibia: proximal (cap ⟂ leg axis) fire to force **increases**; distal (cap ∥ leg axis) to **decreases** ([Noah et al. 2004, *J Comp Physiol A* 190:63–72](https://doi.org/10.1007/s00359-003-0487-y), abstract).

**Muscle force is an explicit part of the adequate stimulus.** Trochanteral CS give "vigorous, directional responses also … to contractions of the trochanteral depressor muscle and to forces applied **at the muscle insertion**" ([Zill et al. 2012](https://doi.org/10.1152/jn.00274.2012)). This matters for your model — see the last section.

---

## Q2. Quantitative stimulus–response data

This is the hardest question to answer in the units your model needs, because **most of the quantitative work reports stimulus in applied force (mN) or joint torque (µNm), and the two are not interchangeable across studies or species.**

**Blow fly (the only study I found that ties force range to the animal's own weight).** Tibial CS tonic firing increases **linearly up to about 0.3 mN**, which the authors note "exceeds the mass of the animal (mean weight of blow flies 0.29 mN)". Responses "consistently reflect the force amplitude **above 0.05 mN**", with r² = 0.93 over 191 tests in 3 animals ([Zill et al. 2025](https://doi.org/10.1152/jn.00044.2025)). Note that 0.29 mN is the *whole-body weight expressed as a force*; the paper's own framing is that the operating range brackets one body weight. Immobilising the femoro-tibial joint with glue "stiffened the cuticle and **shifted the response range to a slightly higher range of forces**" without changing sensitivity qualitatively.

**Stick insect tibial group.** Applied peak forces spanned **0.57–2.8 mN** and hold offsets **0 to 5.4 mN** ([Zill et al. 2021](https://doi.org/10.1152/jn.00120.2021); [Harris et al. 2022](https://doi.org/10.1152/jn.00285.2022)). Rate coding: pooled 6B firing vs +dF/dt gave **r² = 0.98** with regression slopes of **8.1** (large units) and **8.2** (all 6B) AP/s per unit dF/dt, and a separate extensor-torque dataset gave **r² = 0.92** (n = 609 tests, N = 3 animals) ([Zill et al. 2021](https://doi.org/10.1152/jn.00120.2021)). Responses saturate: smaller units reflect rate at very low forces but "responses begin to **saturate at moderate rates** of dF/dt".

**Stick insect trochanteral group — the only published torque threshold I found.** "The lowest torque and torque rate at which G3/G4 afferent activity was confirmed were **3.4 µNm and 10.5 µNm s⁻¹**, respectively", measured with a trochanter–femur lever of 11.7 mm, and these are explicitly labelled "sensory thresholds" that lie well below what is needed to signal unloading during walking ([Dallmann et al. 2017](https://doi.org/10.1098/rspb.2017.1755)). The same study measured the coxo-trochanteral torque plateau during walking as **54 ± 9 µNm** (n = 244 steps, N = 10 animals), with unloading rate peaking at **−440 ± 192 µNm/s**.

**Cockroach tibia.** Ramp-and-hold at mean amplitude **0.85 ± 0.42 mN**, offsets up to **2.45 mN**; *proximal* receptors give sustained tonic firing that tracks offset magnitude, *distal* receptors are progressively silenced (slope −13.7) ([Harris et al. 2022](https://doi.org/10.1152/jn.00285.2022)). I found **no published numeric firing-rate threshold for cockroach leg CS** in the accessible literature.

**Stick insect walking torques (context).** Femoro-tibial torques in downhill walking "could attain **twice** the values that occurred on level surfaces" ([Zill et al. 2024](https://doi.org/10.1152/jn.00414.2023)).

---

## Q3. Number, groups, and cell types in *Drosophila*

**Counts (Dinges et al. 2021, SEM of multiple flies).** A single leg carries **42 CS (prothoracic), 42 (mesothoracic), 41 (metathoracic)** — i.e. **125 per fly** across the six legs. Arrangement per leg: **2 fields, 6 groups, 2 single sensilla**. Breakdown:

| Group/field (per leg) | Count | Location |
|---|---|---|
| TrFpF / TrFpM / TrFpR | 7–8 (7 in 6/28 samples) | dorsal trochanter, posterior subfield |
| TrFaF / TrFaM / TrFaR | 5 (2/18 samples had 4) | dorsal trochanter, anterior subfield |
| TrGF / TrGM / TrGR | 3 | posterior trochanter |
| FeFF / FeFM / FeFR | 10–12 | ventral, proximal femur |
| FeSF / FeSM / FeSR | 1 | dorsal femur |
| TiGdF/M/R | 2 | dorsal, proximal tibia |
| TiGvF/M/R | 3 (1/12 had 2) | ventral, proximal tibia |
| Ta1GF/M/R | 2 | dorsal, distal tarsomere 1 |
| Ta1SF/SM/SR | 1 | ventral tarsomere 1 |
| Ta3GF/M/R | 2 | dorsal, distal tarsomere 3 |
| Ta5GF/M/R | 3–4 | ventral, distal tarsomere 5 |
| **Coxa** | **0** | "In spite of its large volume no CS was observed on the Coxa" |

Source: [Dinges et al. 2021, *J Comp Neurol* 529:905–925](https://doi.org/10.1002/cne.24987). Whole-body total: "**over 680 sensilla** arranged in 26 fields, 54 groups, and 34 single CS". Group/field definitions used: single = no neighbour within ~50 µm; group = 2–4 CS within 25 µm; field = >3 CS in a row/column at <10 µm.

**Is there a published anatomy→cell-type mapping? Partially, and it is thin.** This is the honest state of the field:

- **FlyBase/Drosophila anatomy ontology** provides an anatomical hierarchy but *by leg segment*, not by the fields above: `mechanosensory neuron of leg campaniform sensillum` (**FBbt:00048058**) with children for femoral (FBbt:00048060), trochanter (FBbt:00048059), tibial (FBbt:00058061), metatarsal (FBbt:00058065), tarsal 3 (FBbt:00058069), tarsal 5 (FBbt:00058073), each further split pro/meso/metathoracic. Source: [FlyBase CV report FBbt:00048058](https://flybase.org/cgi-bin/cvreport.pl?id=FBbt:00048058).
- **Connectome-derived cell types** come from [Phelps et al. 2021, *Cell* 184:759–774](https://doi.org/10.1016/j.cell.2020.12.013), which defined three connectivity classes alongside the anatomy: `bilateral campaniform sensillum neuron of leg` (**FBbt:00049572**; **2 per leg**, largest-diameter leg sensory neurons, project to *both* ipsi- and contralateral neuromeres and synapse **directly onto leg motor neurons**), `ipsilateral multi-neuromere campaniform sensillum neuron of leg` (**FBbt:00049573**), and `single-neuromere campaniform sensillum neuron of leg` (**FBbt:00049574**). Definitions quoted from the OBO records.
- **BANC's own scheme** does *not* use the `SNpp##` style. BANC annotates `primary class = campaniform sensillum neuron` (a sibling of `hair plate neuron`, `chordotonal neuron`, `bristle mechanosensory neuron`) and separately records `body part innervated`, e.g. `innervates T1 leg → innervates trochanter / femur / tibia / tarsus`. Source: [BANC wiki, "Annotations (cell types, etc.)"](https://github.com/jasper-tms/the-BANC-fly-connectome/wiki/Annotations-(cell-types,-etc.)).
- **`SNpp` is MANC nomenclature, and it does not mean "campaniform".** Per the MANC manuscript: sensory types are `SN` (non-ascending) or `SA` (ascending), then a **modality** abbreviation — `ch` chemosensory, **`pp` proprioceptive**, `ta` tactile, `xx` unknown — then a type number; `xx` as the number means unclassified ([MANC, Cambridge repository PDF](https://www.repository.cam.ac.uk/bitstreams/33ee5643-0c56-4ed1-8b05-80456af8fdf4/download)). MANC identified **46 CS types**: 21 wing, 6 wing-base, 14 haltere, **1 serial type from the legs**, 1 antennal — and states that it **could not distinguish leg CS from hair-plate neurons** in an unclassified proprioceptive pool annotated `SNppxx`. The one leg CS type it did resolve is the trochanter/bilateral CS, **type `SNpp53`**. MANC also reports ~1200 CS over legs+wings+halteres+antennae (citing Gnatzy et al. 1987). **So: your `campaniform_sensillum_neuron` channel is almost certainly a BANC *class* annotation, not a MANC `SNpp` type.**
- **`SNpt` did not appear in any source I opened.** I could not find it in the BANC wiki, the BANC *Nature* paper, the MANC manuscript, or FlyBase search. Treat it as unverified.

---

## Q4. Dynamic range in body-weight fractions, and cuticular strain magnitudes

**Direct answer: no, there is no published body-weight-fraction dynamic range for Drosophila leg CS, and no published microstrain measurement at the CS attachment sites.** Here is exactly what exists.

**(a) The one study that expresses CS range in body-weight units** is the blow fly tibial work: linear range up to **≈0.3 mN**, animal weight **0.29 mN** — i.e. the linear range is roughly **one body weight**, with a **0.05 mN** amplitude threshold ≈ **17 % of body weight** ([Zill et al. 2025](https://doi.org/10.1152/jn.00044.2025)). Extrapolating that fraction to *Drosophila* is not licensed by any paper I found.

**(b) Body weight correlates with CS *cap-size range*, not with numbers.** Harris et al. imaged mechanotransduction sites in four species and concluded the number of receptors is only loosely related to body weight (blow flies and *Drosophila* have **similar numbers despite a 30-fold weight difference**, and a 30-mg juvenile cockroach has a similar number to *Drosophila*), whereas "the gradient (range) of cap sizes may more closely correlate with the body weight" ([Harris et al. 2020, abstract](https://doi.org/10.1016/j.asd.2020.100970)). The authors state this "support[s] the idea that morphological properties of force-detecting sensory receptors in the legs may be tuned to reflect the body weight." **I am reporting this from the abstract only — the full text is paywalled at Elsevier and every route I tried (ScienceDirect, Unpaywall "hybrid OA" link, publisher PDF endpoint) returned 403, so I cannot quote their cap diameters or the actual weight numbers.** A companion paper by the same group puts *C. morosus* at **800 mg** and *Drosophila* at **1 mg** ([Dinges et al. 2023](https://par.nsf.gov/servlets/purl/10464276)).

**(c) Strain magnitudes — model, not measurement.** The only quantified strain/displacement result at a *Drosophila* CS site is the finite-element model: the femoral CS field sits at **2.5 % of femur length** from the TrF joint, and under simulated forward-walking ground reaction forces the displacement of the NompC transduction site **does not exceed 7 Å**, the thermal-noise floor; the femur bends **less than 2°**. To cross 7 Å you would need **×1.57–3.05** more displacement depending on axis. The authors compare this to jumping (**≈8×** walking force) and landing (**≈80×** walking force) as conditions that *could* suffice, and conclude "this CS field (and possibly others in comparable positions) encodes **muscle tension rather than ground reaction forces**" ([Saltin et al. 2025](https://doi.org/10.1098/rsif.2024.0559)).

**(d) Microstrain — recorded, but not published as numbers.** A dynamically-scaled robotic *Drosophila* leg with strain gauges at the TrF/TrG CS positions recorded **microstrain and microstrain/s** during stepping; the results are presented only as Figure 2 plots, and the text reports no numeric strain or strain-rate values, and no body-weight normalisation ([Dinges et al. 2023](https://par.nsf.gov/servlets/purl/10464276)). This is the closest thing to a microstrain measurement at *Drosophila* CS sites that exists, and it is a robot, not a fly.

**(e) What the CS groups are named in strain terms, if you want internal consistency.** Cockroach femoral CS caps have long-axis length **11.9 ± 4.9 µm**; stick insect femoral CS **16 ± 1.58** receptors, dorsal caps **9.5 ± 1.6 µm**, ventral **9.7 ± 2.2 µm** ([Zill et al. 2017](https://doi.org/10.1016/j.asd.2017.05.004)). **No equivalent cap measurements for *Drosophila* are available to me** — Harris et al. 2020 is the paper that has them and it is behind a paywall.

---

## Q5. Evidence that CS mediate a leg load/force reflex on motor neurons

Yes, this is well established — but predominantly in **stick insect and cockroach**, and the fly evidence is new and indirect.

1. **Stick insect, tibial 6A/6B → opposing reflexes.** Group 6B receptors **excite**, and group 6A **inhibit**, tibial extensor and trochanteral depressor motoneurons ([Zill et al. 2011, abstract](https://doi.org/10.1007/s00359-011-0647-4)).
2. **Stick insect, trochanteral G3/G4 → depressor MN reflex.** "Stimulation of the receptors produced **reflex effects in the depressor motoneurons** that could **reverse in sign** during active movements" ([Zill et al. 2012](https://doi.org/10.1152/jn.00274.2012)).
3. **Cockroach, freely standing — load reflex with body weight.** Magnets on the thorax plus a coil below the substrate changed effective body load; proximal tibial sensilla discharged to force increases and "could encode the **level of load** when increases were sustained", and **trochanteral extensor motoneuron firing was strongly modulated by changing load** ([Noah et al. 2004, abstract](https://doi.org/10.1007/s00359-003-0487-y)).
4. **Stick insect, trochanteral CS → thoraco-coxal MN pools, direction-dependent.** Load-sensor stimulation altered protractor/retractor coxae motoneuron activity; **ablation experiments showed the load influence is mediated by the three groups of trochanteral campaniform sensilla**, and the sign of the effect depends on walking direction ([Akay et al. 2007, *J Neurosci* 27:3285–3294](https://doi.org/10.1523/JNEUROSCI.5202-06.2007)).
5. **Stick insect, torque waveforms from real walking.** CS firing follows +dF/dt during stance on level, uphill and downhill substrates; feeding those torques to a published CS model reproduced the discharges without retuning, and the authors propose the signal "can be used to **counter the instability** in traversing sloped surfaces" ([Zill et al. 2024](https://doi.org/10.1152/jn.00414.2023)).
6. **Stick insect, load drives inter-leg coordination.** Single-leg ground reaction forces and inverse dynamics gave a CTr torque plateau of **54 ± 9 µNm**, with unloading onset marked by **−440 ± 192 µNm/s**; neighbouring legs unload each other by **23 ± 5 µNm** (~50 % of CTr torque), and G3/G4 "sensory thresholds" of **3.4 µNm / 10.5 µNm/s** are well below that ([Dallmann et al. 2017](https://doi.org/10.1098/rspb.2017.1755)).
7. **Drosophila (2026 preprint — abstract only).** Activation of CS **induces activity in many leg muscles**, demonstrating that CS stimuli can drive motor neuron activity; transient optogenetic inhibition of CS reduced walking speed, with shorter stance amplitudes of long duration plus loss of inter-leg coordination and postural control ([Custodio et al. 2026, bioRxiv](https://doi.org/10.64898/2026.07.22.740025)). **I could only read the abstract** — bioRxiv returned HTTP 429/403 on every full-text and PDF attempt.
8. **Drosophila connectome substrate for a monosynaptic load path.** The bilateral (`bCS`) CS neurons are **2 per leg**, are the **largest-diameter leg sensory neurons**, and **synapse directly onto leg motor neurons** ([Phelps et al. 2021](https://doi.org/10.1016/j.cell.2020.12.013) via the [FBbt:00049572](http://purl.obolibrary.org/obo/FBbt_00049572) record). MANC's resolved leg CS type `SNpp53` corresponds to exactly this trochanter/bilateral class and connects predominantly to **tergotrochanter motor neurons**, which the authors interpret as "a role in the posture changes due to mechanical load applied and sensed by the TrCS proprioceptors" ([MANC](https://www.repository.cam.ac.uk/bitstreams/33ee5643-0c56-4ed1-8b05-80456af8fdf4/download)). Note MANC describes TrCS contribution to leg motor circuits as **weak** ("strong association limited to the Tergotrochanter motor neurons").

---

## WHAT THIS DOES NOT SUPPORT

**The literature does not justify normalising this channel by body weight as a *general* load signal — and for the femoral/trochanteral `campaniform_sensillum_neuron` population specifically, it argues against GRF-derived input altogether.**

1. **The normaliser the project uses is not the variable the receptors encode.** The single most explicit statement in the literature is that tibial CS "**do not directly convey the magnitude of the Newtonian force or reflect the body weight** (as, for example, a bathroom weighing scale)" ([Zill et al. 2024](https://doi.org/10.1152/jn.00414.2023)). Firing is dominated by `dF/dt`, with amplitude a secondary, saturating, hysteretic term. Dividing a GRF by body weight produces a *force* signal; the receptor is closer to a *force-rate* sensor. Normalising by body weight does not convert one into the other.

2. **For *Drosophila* specifically, GRF-proportional input to the femoral CS field is contradicted by the best available model.** Saltin et al. found the transduction-site displacement under forward-walking GRF stays **below the 7 Å thermal-noise floor** and would need **×1.57–3.05** amplification, and concluded the field "**encodes muscle tension rather than ground reaction forces**", with jumping (≈8×) or landing (≈80×) forces needed to make GRF an adequate stimulus ([Saltin et al. 2025](https://doi.org/10.1098/rsif.2024.0559)). If your model's only suprathreshold channel is GRF-normalised, the most likely explanation is that the model's input scaling is a modelling choice, not a biophysical finding. That is not necessarily wrong as an engineering decision, but it should not be described as what campaniform sensilla do.

3. **Muscle force is an equal or better candidate than GRF.** CS attach adjacent to muscle insertions, and trochanteral CS respond "vigorously" to **depressor muscle contractions and to forces applied at the muscle insertion** ([Zill et al. 2012](https://doi.org/10.1152/jn.00274.2012)). Saltin et al.'s proposed resolution is co-contraction of antagonists near the CS field. The project's own description — that only this channel crosses threshold and removing it silences motor output — is consistent with a **muscle-derived strain** signal rather than a foot-contact force signal.

4. **Is there a body-weight-based normaliser in the literature? Only morphological, not functional.** Harris et al. 2020 propose that cap-size *range* is tuned to body weight and that CS "may be tuned to reflect the body weight" ([abstract](https://doi.org/10.1016/j.asd.2020.100970)) — but that is an argument about **cap diameter**, not about firing rate per body weight. The only functional normalisation I found is the blow fly study, and it works because the authors measure the animal's actual mass and report that the linear range brackets it ([Zill et al. 2025](https://doi.org/10.1152/jn.00044.2025)).

**A defensible normaliser, if you must have one:**
- If you keep a load channel: normalise by the **per-leg vertical ground reaction force measured in the same preparation**, not by body weight — because the study that provides a joint-torque threshold in physical units is `3.4 µNm` at an `11.7 mm` lever ([Dallmann et al. 2017](https://doi.org/10.1098/rspb.2017.1755)), and torque, not weight, is the quantity the CS groups demonstrably track.
- Physiologically best-supported: make the channel a function of **local cuticular strain at the CS attachment site**, or of **`dF/dt` thereof**. That is what every direct recording in the literature actually measures, and it is what Saltin et al. 2025 computed.
- If you want the strongest literature-licensed scaling: **do not normalise at all**; instead set the channel's operating point from the two published absolute anchors — **0.05 mN** amplitude threshold and **≈0.3 mN** linear ceiling for a small-dipteran tibial group ([Zill et al. 2025](https://doi.org/10.1152/jn.00044.2025)) — and state that this is a blow-fly-derived anchor applied to *Drosophila*, which the literature does not license.

---

## DISAGREEMENTS AND GAPS

**Live disagreements in the literature**

1. **Phasic vs. tonic emphasis differs by species and group.** Stick insect tibial 6B is dominated by `dF/dt` and adapts to <10 Hz during hold ([Zill et al. 2021](https://doi.org/10.1152/jn.00120.2021)); blow fly tibial CS are "sustained and strongly tonic even at very low forces" *and* highly rate-sensitive ([Zill et al. 2025](https://doi.org/10.1152/jn.00044.2025)). A single "campaniform sensillum neuron" channel in a connectome model conflates these.
2. **Whether larger caps mean larger or smaller forces.** `bCS` neurons have the **largest diameter** of any leg sensory neuron ([Phelps et al. 2021](https://doi.org/10.1016/j.cell.2020.12.013)); in cockroach trochanteral CS, larger caps are recruited as force **increases** (range fractionation). But in *Drosophila* the femoral field's placement is sub-threshold, so at least one large-diameter class is not a high-force sensor. Diameter alone does not predict dynamic range.
3. **Astraight contradiction on axon architecture.** [Phelps et al. 2021](https://doi.org/10.1016/j.cell.2020.12.013) reports functionally **bilateral** campaniform neurons synapsing onto contralateral motor neurons; [Zill et al. 2024](https://doi.org/10.1152/jn.00414.2023) and the larger-insect literature emphasise strictly **ipsilateral, directionally specialised** load reflexes. These may be different cell classes, but no paper reconciles them.
4. **Reflex sign is not fixed.** Trochanteral CS reflex effects "could **reverse in sign** during active movements" ([Zill et al. 2012](https://doi.org/10.1152/jn.00274.2012)), and load-signal processing depends on walking direction ([Akay et al. 2007](https://doi.org/10.1523/JNEUROSCI.5202-06.2007)). A model that hard-wires this channel as a fixed excitatory load input is not supported.
5. **MANC vs. fly physiology.** MANC found TrCS contribution to leg motor circuits **weak**, strongly associating only with tergotrochanter motor neurons; the 2026 preprint reports CS **activation induces activity in many leg muscles** ([Custodio et al. 2026](https://doi.org/10.64898/2026.07.22.740025)). Connectivity strength in a static connectome is a poor predictor of functional gain.

**Gaps — things I could not find published at all**

1. **No microstrain measurement at a *Drosophila* CS attachment site.** The "ideal" data you asked for does not exist. Saltin et al. 2025 is a model; Dinges et al. 2023 is a robot. Both are explicit that direct strain measurement in animals of this size is not currently feasible ("smaller insects like *D. melanogaster* … only produce minute forces that are difficult to monitor using modern tools").
2. **No firing-rate-versus-load curve in physical units for any *Drosophila* leg CS.** Every published intracranial recording of leg CS is in stick insect, cockroach, locust or blow fly. No paper I found records from *Drosophila* leg CS.
3. **No numeric CS firing thresholds for the cockroach**, and no numeric threshold at all for stick insect tibial 6A/6B. The only numeric torque threshold in the entire accessible corpus is 3.4 µNm / 10.5 µNm s⁻¹ for stick insect G3/G4 ([Dallmann et al. 2017](https://doi.org/10.1098/rspb.2017.1755)).
4. **No body-weight-fraction dynamic range for any species except the blow fly**, and even there the framing is "linear up to about 0.3 mN" against a 0.29 mN body weight — n = 3 animals, and the authors note "further quantification of response sensitivities were limited by the stiffness of the femoro-tibial joint."
5. **No anatomy→cell-type map at the level of the fields in Dinges et al. 2021.** FlyBase, MANC and BANC all stop at leg-segment resolution (or coarser). MANC explicitly says it **could not separate leg CS from hair-plate neurons**, leaving ~25 % of proprioceptive leg neurons as `SNppxx`. There is no published correspondence between `TrFpF`/`FeFF`/`TiGvF`-style anatomical groups and connectome cell types.
6. **The Harris et al. 2020 cap diameters are the key missing number** for any size-based argument about *Drosophila* CS and they are paywalled. If you need them, that is the one paper to obtain through institutional access.

**Access limitations of this report (stated so you can weight it)**

- **Full text read:** Dinges et al. 2021; Saltin et al. 2025; Zill et al. 2012; Zill et al. 2017; Zill et al. 2021; Zill et al. 2024; Zill et al. 2025; Harris et al. 2022; Dallmann et al. 2017; Akay et al. 2007; Dinges et al. 2023; the MANC manuscript; the BANC *Nature* paper; the BANC wiki; FlyBase and OBO ontology records.
- **Abstract only (paywalled or blocked):** Harris et al. 2020 (*Arthropod Struct Dev*); Zill et al. 2011 (*J Comp Physiol A*); Noah et al. 2004 (*J Comp Physiol A*); Custodio et al. 2026 bioRxiv preprint (cloudflare 429/403 on full text).
- **This report contains no number that was not either stated in accessible text or explicitly flagged as figure-only.** Where a value is figure-only I have said so.

---

### Full citation list

1. Dinges GF, Chockley AS, Bockemühl T, Ito K, Blanke A, Büschges A. (2021) Location and arrangement of campaniform sensilla in *Drosophila melanogaster*. *J Comp Neurol* 529:905–925. [10.1002/cne.24987](https://doi.org/10.1002/cne.24987)
2. Harris CM, Dinges GF, Haberkorn A, Gebehart C, Büschges A, Zill SN. (2020) Gradients in mechanotransduction of force and body weight in insects. *Arthropod Struct Dev* 58:100970. [10.1016/j.asd.2020.100970](https://doi.org/10.1016/j.asd.2020.100970) *(abstract only)*
3. Saltin BD, Goldsmith C, Haustein M, Büschges A, Szczecinski NS, Blanke A. (2025) A parametric finite element model of leg campaniform sensilla in *Drosophila* to study campaniform sensilla location and arrangement. *J R Soc Interface* 22:20240559. [10.1098/rsif.2024.0559](https://doi.org/10.1098/rsif.2024.0559)
4. Zill SN, Büschges A, Schmitz J. (2011) Encoding of force increases and decreases by tibial campaniform sensilla in the stick insect, *Carausius morosus*. *J Comp Physiol A* 197:851–867. [10.1007/s00359-011-0647-4](https://doi.org/10.1007/s00359-011-0647-4) *(abstract only)*
5. Noah JA, Quimby L, Frazier SF, Zill SN. (2004) Sensing the effect of body load in legs: responses of tibial campaniform sensilla to forces applied to the thorax in freely standing cockroaches. *J Comp Physiol A* 190:63–72. [10.1007/s00359-003-0487-y](https://doi.org/10.1007/s00359-003-0487-y) *(abstract only)*
6. Zill SN, Schmitz J, Chaudhry S, Büschges A. (2012) Force encoding in stick insect legs delineates a reference frame for motor control. *J Neurophysiol* 108:1453–1472. [10.1152/jn.00274.2012](https://doi.org/10.1152/jn.00274.2012)
7. Zill SN, Dallmann CJ, Szczecinski NS, Büschges A, Schmitz J. (2021) Evaluation of force feedback in walking using joint torques as "naturalistic" stimuli. *J Neurophysiol* 126:227–248. [10.1152/jn.00120.2021](https://doi.org/10.1152/jn.00120.2021)
8. Zill SN, Dallmann CJ, Zyhowski WP, Chaudhry H, Gebehart C, Szczecinski NS. (2024) Mechanosensory encoding of forces in walking uphill and downhill: force feedback can stabilize leg movements in stick insects. *J Neurophysiol* 131:198–215. [10.1152/jn.00414.2023](https://doi.org/10.1152/jn.00414.2023)
9. Zill SN, Chaudhry S, Chaudhry H, Szczecinski NS. (2025) Force sensing in small animals: recording response properties and modeling of tibial campaniform sensilla in blow flies. *J Neurophysiol* 133:1749–1760. [10.1152/jn.00044.2025](https://doi.org/10.1152/jn.00044.2025)
10. Harris CM, Szczecinski NS, Büschges A, Zill SN. (2022) Sensory signals of unloading in insects are tuned to distinguish leg slipping from load variations in gait: experimental and modeling studies. *J Neurophysiol* 128:790–807. [10.1152/jn.00285.2022](https://doi.org/10.1152/jn.00285.2022)
11. Dallmann CJ, Hoinville T, Dürr V, Schmitz J. (2017) A load-based mechanism for inter-leg coordination in insects. *Proc R Soc B* 284:20171755. [10.1098/rspb.2017.1755](https://doi.org/10.1098/rspb.2017.1755)
12. Zill SN, Neff D, Chaudhry S, Exter A, Schmitz J, Büschges A. (2017) Effects of force detecting sense organs on muscle synergies are correlated with their response properties. *Arthropod Struct Dev* 46:564–578. [10.1016/j.asd.2017.05.004](https://doi.org/10.1016/j.asd.2017.05.004)
13. Akay T, Ludwar BCh, Göritz ML, Schmitz J, Büschges A. (2007) Segment specificity of load signal processing depends on walking direction in the stick insect leg muscle control system. *J Neurosci* 27:3285–3294. [10.1523/JNEUROSCI.5202-06.2007](https://doi.org/10.1523/JNEUROSCI.5202-06.2007)
14. Phelps JS, Hildebrand DGC, Graham BJ, et al. (2021) Reconstruction of motor control circuits in adult *Drosophila* using automated transmission electron microscopy. *Cell* 184:759–774. [10.1016/j.cell.2020.12.013](https://doi.org/10.1016/j.cell.2020.12.013)
15. Marin EC, Morris BJ, Stürner T, et al. (2026) Systematic annotation of a complete adult male *Drosophila* nerve cord connectome (MANC). eLife reviewed preprint 97766. PDF: [repository.cam.ac.uk](https://www.repository.cam.ac.uk/bitstreams/33ee5643-0c56-4ed1-8b05-80456af8fdf4/download)
16. Dorkenwald S, et al. (2026) Distributed control circuits across a brain-and-cord connectome (BANC). *Nature*. [10.1038/s41586-026-10735-w](https://doi.org/10.1038/s41586-026-10735-w)
17. Dinges GF, Zyhowski WP, Goldsmith CA, Szczecinski NS. (2023) Comparison of proximal leg strain in locomotor model organisms using robotic legs. *Living Machines / LNCS*. [NSF PAR 10464276](https://par.nsf.gov/servlets/purl/10464276)
18. Custodio et al. (2026) Role of leg campaniform sensilla sensory feedback in *Drosophila melanogaster* adaptive walking. bioRxiv. [10.64898/2026.07.22.740025](https://doi.org/10.64898/2026.07.22.740025) *(abstract only)*
19. Szczecinski NS, Zill SN, Dallmann CJ, et al. (2020) Modeling the dynamic sensory discharges of insect campaniform sensilla. [NSF PAR 10202934](https://par.nsf.gov/servlets/purl/10202934)
20. Eich et al. (2025) An overview of campaniform sensilla arrangement and location on hexapod legs. *Arthropod Struct Dev*. [10.1016/j.asd.2025.101477](https://doi.org/10.1016/j.asd.2025.101477) *(not read)*
21. BANC connectome wiki, "Annotations (cell types, etc.)": [github.com/jasper-tms/the-BANC-fly-connectome](https://github.com/jasper-tms/the-BANC-fly-connectome/wiki/Annotations-(cell-types,-etc.))
22. FlyBase anatomy ontology entries: [FBbt:00048058](https://flybase.org/cgi-bin/cvreport.pl?id=FBbt:00048058), and OBO records for [FBbt:00049572](http://purl.obolibrary.org/obo/FBbt_00049572), [FBbt:00049573](http://purl.obolibrary.org/obo/FBbt_00049573), [FBbt:00049574](http://purl.obolibrary.org/obo/FBbt_00049574).
