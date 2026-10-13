# Claw / Hook / Club chordotonal neurons: literature report for BANC→MuJoCo sensory front end

Scope: what the three BANC cell types `claw_chordotonal_organ_neuron`, `hook_chordotonal_organ_neuron`,
`club_chordotonal_organ_neuron` actually are, what they transduce, published ranges/thresholds/gains, and
whether the project's current mappings are defensible.

**Bottom line up front:** all three are *subpopulations of one organ, the femoral chordotonal organ (FeCO),
which sits in the femur and transduces movement of the **femur–tibia** joint.* There is no separate "claw
organ", "hook organ" or "club organ" — those words describe the shape of the axon arborization in the VNC.
The names are therefore **not** in tension with each other anatomically, but they also **cannot** be used to
address different leg joints. In particular, the project's mapping of `club` to a mean over several leg joints
is **refuted** by the literature (club = bidirectional tibia movement + tibia vibration).

---

## Summary table

| Organ / cell type (BANC name → FBbt) | Segment / synonyms | Physical stimulus | Published range | Threshold | Gain (published) | Source |
|---|---|---|---|---|---|---|
| **FeCO overall** (femoral chordotonal organ) `FBbt:00007592` | Proximal **femur**; synonyms FCO, FeCO. ~135 cell bodies ([Mamiya 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/)); 152 reconstructed ([Mamiya 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10644877/), citing Kuan et al. 2020) | Femur–tibia joint kinematics; also substrate vibration | Joint angle ~18°–180° in the controlled-stimulus literature | not stated for the organ as a whole | not published | [Mamiya 2018 Neuron](https://doi.org/10.1016/j.neuron.2018.09.009); [Mamiya 2023 Neuron](https://doi.org/10.1016/j.neuron.2023.07.009) |
| **claw** `claw_chordotonal_organ_neuron` → `FBbt:00049557` (T1: `FBbt:00049560`) | Femur, **ventral scoloparium** (exact synonym "FeCO claw neuron"); BANC also records Primary Cell Type `SNpp50`; older anatomy = "FeCO 1/2/3" groups, mapping unresolved | **Tibia position** (femur–tibia joint angle); flexion- and extension-tuned cells exist as separate sub-branches | Tonic and ~linear over the tested 18°–180° ramp; flexion-tuned active ~18°–90°, extension-tuned ~90°–180°; single cells peak at specific angles (e.g. 70°, or fully flexed 20°) | **not published in degrees.** The only published sensitivity comparison: claw mechanosensory threshold is **>10× higher** than club | **not published** (no spikes/s/deg for any *Drosophila* FeCO neuron) | [Mamiya 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/); [Mamiya 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10644877/); [VFB FBbt_00049557](https://www.virtualflybrain.org/blog/2022/01/01/femoral-chordotonal-claw-neuron-fbbt_00049557/) |
| **hook** `hook_chordotonal_organ_neuron` → `FBbt:00049558` (T1: `FBbt:00049559`) | Femur, split: **lateral scoloparium** = hook *flexion* cells; **ventral scoloparium** = hook *extension* cells. BANC Primary Cell Type seen = `SNpp39`. **Not** a synonym of FeCO — it is a subpopulation of it | **Tibia movement, directionally tuned** (phasic, flexion OR extension, not both) | Phasic at the swing speeds tested 100–800 °/s; tested joint range 18°–180°; response weaker at full extension | not published; **never tonic during hold** | **not published.** Only qualitative statement: calcium-signal slope similar across 100–800 °/s (unlike club, which peaks ~400 °/s) | [Mamiya 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/); [Mamiya 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10644877/); [VFB FBbt_00049558](https://virtualflybrain.org/blog/2022/01/01/femoral-chordotonal-hook-neuron-fbbt_00049558/) |
| **club** `club_chordotonal_organ_neuron` → `FBbt:00049556` (T1: `FBbt:00049561`) | Femur, **dorsal scoloparium** (largest, most dorsal compartment); proximal part of the organ. BANC FANC reconstruction counted 37 club axons; Mamiya 2018 labelled ~30 somata | **Two things:** (a) bidirectional tibia movement; (b) low-amplitude **high-frequency tibia vibration** — regarded as an **exteroceptive** channel | (a) any movement direction, amplitude slightly larger near 90°, smaller at 180°. (b) **100–2000 Hz** tested; peaks at **400 Hz** (0.9 µm) or **800 Hz** (0.054 µm); single cells peak anywhere 200–1600 Hz | **smallest vibration actually tested = 0.054 µm peak-to-peak** (= 54 nm). Reported as *the* vibration-sensitive subtype; claw and hook did not respond | **not published.** Club threshold stated only *relative* to claw (>10× lower). Vibration gain in spikes/s/µm not published | [Mamiya 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/); [Lee et al. 2025 Nat Commun](https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/) |
| *Downstream comparison (not a receptor)* 13Bα | VNC neuron, GABAergic, 13B hemilineage | Femur–tibia **joint angle** | Increases only when tibia extended past ~90° | n/a | **No action potentials at all**; tonic membrane-potential code | [Agrawal et al. 2020 eLife](https://pmc.ncbi.nlm.nih.gov/articles/PMC7752136/) |
| *Downstream comparison (not a receptor)* 9Aα | VNC neuron, GABAergic, 9A hemilineage | Tibial **flexion speed** + **high-frequency vibration** | Max depolarization at 1600–2000 Hz | n/a | Peak firing rate increases with flexion speed (720 °/s > 240 °/s, p<0.005); numeric Hz/° not published | [Agrawal et al. 2020 eLife](https://pmc.ncbi.nlm.nih.gov/articles/PMC7752136/) |

---

## Q1. What the claw, hook and club organs are, where they are, what they attach to, and whether the names are standard

**They are not three organs.** They are three (functionally five) subpopulations of the **femoral
chordotonal organ (FeCO)**, the largest somatosensory organ in the fly leg, whose ~135–152 sensory neurons
sit in the **proximal femur** ([Mamiya et al. 2018, Neuron](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/);
[Mamiya et al. 2023, Neuron](https://pmc.ncbi.nlm.nih.gov/articles/PMC10644877/);
[Lee et al. 2025, Nat Commun](https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/)). The FeCO ontology term
defines it flatly as "Large chordotonal organ found in the proximal femur", synonyms FCO and FeCO
([FBbt:00007592](https://www.ebi.ac.uk/ols4/api/ontologies/fbbt/terms?iri=http%3A%2F%2Fpurl.obolibrary.org%2Fobo%2FFBbt_00007592)).

**Structure and attachment.** Each receptor pair is ensheathed by a scolopale cell that connects via an
actin-rich **cap cell** to a sensory **tendon** ([Mamiya 2023, Fig. 1B](https://pmc.ncbi.nlm.nih.gov/articles/PMC10644877/)).
The organ is organised into **three compartments (scoloparia)**, and the compartments attach to **different
tendons**:

- The tendons of the **claw** and **hook** compartments merge and fuse ~100 µm distal to the FeCO into a
  **medial tendon**.
- The **club** compartment's tendon stays separate, as the **lateral tendon**.
- Both tendons converge on a tooth-shaped sclerite in the **distal femur** called the **arculum** (a
  structure named in beetles), which is in turn coupled to the base of the **tibia-extensor tendon** and to
  the **femur–tibia joint**.

So the chain of attachment is: **tibia → arculum → (medial tendon → claw + hook) and (lateral tendon →
club) → cap cells → dendrites.** Nothing attaches to tibia cuticle directly; the coupling is via the arculum
and tendon, and Mamiya et al. explicitly note they found **no efferent innervation of the FeCO and no
connections between the FeCO and the surrounding muscles** (contradicting an earlier report).

**Which joint?** All three transduce the **femur–tibia joint**. There is no claw/hook/club structure in the
tibia, tarsus, coxa or trochanter. Mamiya et al. state the femur–tibia joint "is the major movement of this
joint during walking in *Drosophila*", and that flexion/extension of the tibia moves the arculum
distally/proximally within the femur, altering strain on the FeCO dendrites.

**Are the names standard?** Partly, and *not* as anatomical terms.

- "**claw**" and "**club**" originate from **Phillis et al. 1996**, and were defined **by the shape of the
  axon arborization in the VNC**, not by peripheral anatomy ([cited by Mamiya 2018 and by the FBbt
  definitions](https://www.virtualflybrain.org/blog/2022/01/01/femoral-chordotonal-claw-neuron-fbbt_00049557/)).
- "**hook**" was coined later by **[Mamiya et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/)** —
  "We call this projection the *hook* due to its resemblance to a lumberjack's peavey hook."
- The **classical insect chordotonal literature does not use these words.** It uses **scoloparium / scolopidia**
  counts, and for *Drosophila* specifically the older anatomy literature (Shanbhag et al. 1992, FlyBase:
  FBrf0056378) names three femoral groups **FeCO 1, FeCO 2 and FeCO 3** in the prothoracic femur
  ([FBbt:00004272/3/4](https://www.ebi.ac.uk/ols4/api/ontologies/fbbt/terms?iri=http%3A%2F%2Fpurl.obolibrary.org%2Fobo%2FFBbt_00004272)).
  **The correspondence between FeCO 1/2/3 and dorsal/ventral/lateral scoloparium is not established in the
  ontology** — the ontology deliberately withholds it, with the comment that "Mamiya et al. (2023) are
  inconsistent with use of 'group 1', 'group 2' and 'group 3' terminology, so these synonyms have not been
  added".

**Is "hook" the FeCO?** No. "hook" is **not** a synonym for the femoral chordotonal organ and **not** a
subunit in the classical sense either. The correct relation is: `hook chordotonal organ neuron` ⊂
`mechanosensory neuron of femoral chordotonal organ` ([FBbt:00048051](https://www.virtualflybrain.org/blog/2022/01/01/femoral-chordotonal-hook-neuron-fbbt_00049558/)).
It is a **VNC-projection-defined subpopulation with two different peripheral origins**: the FBbt ontology
records hook **flexion** neurons as having dendrites in the **lateral scoloparium** and hook **extension**
neurons in the **ventral scoloparium** — which is why the ontology attaches "hook neuron" to the generic
parent term `scoloparium of femoral chordotonal organ` (FBbt:00052626) rather than to one specific
compartment. A useful side effect: the BANC name is closer to the biology than a naïve one-organ-per-name
reading would suggest, because claw and hook really do share a tendon (medial) while club is separate (lateral).

**Inside BANC specifically:** the community annotation table uses `claw chordotonal neuron`,
`hook chordotonal neuron` and `club chordotonal neuron` under `chordotonal neuron`
([BANC wiki, cell_info](https://github.com/jasper-tms/the-BANC-fly-connectome/wiki/Annotations-(cell-types,-etc.))).
VFB pages for individual BANC cells show `Sub Class - front_leg_claw_chordotonal_organ_neuron` with
`Primary Cell Type - SNpp50` ([example](https://www.virtualflybrain.org/blog/2022/01/01/banc_626720575941413112212-vfb_00106ggj/))
and `front_leg_hook_chordotonal_organ_neuron` with `SNpp39` ([example](https://www.virtualflybrain.org/blog/2022/01/01/banc_626720575941438410028-vfb_00106c20/)).
The `SNpp*` codes come from the leg-sensory nomenclature of Takemura et al. 2024 / Marin et al. 2024 (MANC);
the BANC↔SNpp code assignment is **BANC-internal** and, as of the FBbt release I checked (fbbt.obo master),
**SNpp50/57/58/59 are not yet in the Drosophila anatomy ontology** — so that specific mapping is not yet
authoritatively published. I could not retrieve the BANC `codex_annotations`/`cell_info` table directly
(the FANC CAVE server returned 404 on the endpoints available to me, and `pip`/`caveclient` are not installed
in this environment), so **the complete SNpp↔claw/hook/club table is a gap** (see last section).

---

## Q2. Published stimulus–response relations, ranges, thresholds and gains

### claw = tibia position
Tonic, non-adapting. Flexion-tuned claw neurons increase activity when the tibia is between ~18° and ~90°;
extension-tuned when between ~90° and ~180°; neither is active near 90° at the middle of the range.
The population response "increased as a relatively linear function of the femur–tibia joint angle", with
**directional hysteresis** — steady-state activity at a given angle is larger when the joint arrived by the
activating direction ([Mamiya 2018, Fig. 5A, D, E](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/)).
Single claw neurons are more sharply tuned than the population: one peaked at 70° and declined at more flexed
angles; another responded maximally at the most flexed position (20°). Mamiya 2023 added that claw cells
form a **goniotopic map** of joint angle across the linear array of cell bodies.

**No gain in spikes/s/deg is published.** No *Drosophila* FeCO sensory neuron has been recorded with
single-unit electrophysiology in any of the three functional papers; the measurements are **two-photon
calcium imaging (ΔF/F)**, which is not convertible to firing rate. I searched for a *Drosophila* FeCO
firing-rate-versus-angle relation and found none. The only quantitative sensitivity comparison published is
that claw's mechanosensory threshold is **more than 10× higher than club's** ([Mamiya 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10644877/)).

### hook = directional tibia movement
Phasic (adapting), direction-selective: hook **flexion** neurons respond transiently to tibia flexion and
only weakly to extension; hook **extension** neurons are the mirror. Direction selectivity index
DSI = 0.811 ± 0.028 (mean ± SEM, n = 37 regions from 23 flies) for the flexion-tuned set.
Crucially for the project: "Phasic responses to tibia movement decayed rapidly, and we **never observed
tonic activity during the hold period**." So hook is a **velocity/direction** channel and carries no usable
position signal. Speed tuning was tested over **100–800 °/s**: calcium slope in hook was *similar across the
whole range*, i.e. hook does not appear strongly velocity-gain-modulated over that band, whereas club's slope
peaked near 400 °/s ([Mamiya 2018, Fig. S5B](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/)).

### club = bidirectional movement + vibration
Phasic to movement in **both** directions (DSI = 0.117 ± 0.022, n = 25 regions from 15 flies), largest around
90°, smaller at full extension (180°). Then, uniquely among the three, it responds to **vibration**:
stimuli were sinusoids at **0.9 µm or 0.054 µm peak-to-peak** and **100 to 2000 Hz**. With 0.9 µm the
response was significant at all frequencies and **peaked at 400 Hz**; with 0.054 µm it was slightly weaker and
**peaked at 800 Hz**. Claw and hook did not respond to these stimuli at all. Single club neurons are tuned to
different bands, e.g. one peaked at 200 Hz, another at 1600 Hz, and neighbouring cells are systematically
ordered (a **tonotopic** map along the anterolateral→posteromedial axis of the axon projection), which
[Lee et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/) independently replicated as
connectivity clusters tiling the dorsal–ventral axis in FANC.
**0.054 µm (54 nm) peak-to-peak is the smallest amplitude actually tested** — I found no published absolute
displacement threshold or detectability curve for club, so treat 54 nm as an upper bound on threshold, not as
the threshold.

### The biomechanical account of *why* they differ
[Mamiya et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10644877/) explain the split. Their finite-element
model of the arculum predicts that tibia vibration makes the arculum rock, moving the **lateral** root (club)
mainly **on-axis** along the femur and the **medial** root (claw/hook) mainly **orthogonally**, with on-axis
movement transmitted to claw **reduced by as much as 30×**. Club cells are also anchored more firmly
(<2 µm movement when the tibia is fully flexed/extended) while claw and hook-extension cells shift distally
during flexion. For slow macroscopic motion (**1–50 µm** arculum displacement) the tendons simply push/pull
along the femur's long axis, which excites **both claw and club**.

This gives a clean physical statement of the project's channels: **the distal attachment structure decomposes
one joint's motion into two mechanically distinct channels — a position/strain channel (claw), a
fast-directional channel (hook), and a low-amplitude/high-frequency channel (club).**

---

## Q3. Which studies characterise these organs functionally, and what do they call the cells?

The functional literature is small and self-consistent. Exact cell-type names as used in each source:

1. **Phillis et al. 1996** — origin of "claw" and "club" as *VNC arborization* names. (Cited via
   [Mamiya 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/) and the FBbt definitions; I did not
   access the primary text.)
2. **[Mamiya, Gurung & Tuthill 2018, *Neuron* 100:636–650](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/)**
   (doi:10.1016/j.neuron.2018.09.009) — defines the three functional subclasses and the genetic driver lines:
   **claw** (`R73D10-Gal4`, ~20 cells, position), **club** (`R64C04-Gal4`, ~30 cells, bidirectional movement +
   vibration), **hook** (`R21D12-Gal4`, 3 cells labelled, directional movement). Methods: two-photon calcium
   imaging (GCaMP6f) of VNC axons, magnetic leg control, piezo vibration, FlpOut single-cell imaging.
3. **[Agrawal et al. 2020, *eLife* 9:e60299](https://pmc.ncbi.nlm.nih.gov/articles/PMC7752136/)**
   (doi:10.7554/eLife.60299) — central processing. Uses the same three names plus downstream cell types
   **13Bα, 9Aα, 10Bα**. States the FeCO has **152 neurons** divided into claw / hook / club.
4. **[Mamiya et al. 2023, *Neuron* 111:3230–3243.e14](https://pmc.ncbi.nlm.nih.gov/articles/PMC10644877/)**
   (doi:10.1016/j.neuron.2023.07.009) — X-ray holographic nano-tomography anatomy + finite-element model +
   single-nucleus RNA-seq + volumetric calcium imaging. Introduces **scoloparium** nomenclature
   (dorsal = club, ventral = claw + hook extension, lateral = hook flexion), the **arculum**, and the term
   **goniotopic** for the joint-angle map.
5. **[Lee, Dallmann, Cook, Tuthill & Agrawal 2025, *Nat Commun* 16:4105](https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/)**
   (doi:10.1038/s41467-025-59302-3) — FANC + FlyWire connectome analysis. Uses the **same names** and states
   explicitly that the five subtypes are "(1) extension- and (2) flexion-encoding **claw** neurons encode
   tibia position, (3) extension- and (4) flexion-encoding **hook** neurons encode tibia movement, and
   (5) **club** neurons encode bidirectional tibia movement and low-amplitude (<1 µm), high-frequency tibia
   vibration", and that "**Claw, hook, and club neurons are named after the shape of their axons in the VNC**".
   This is the paper that assigns club a predominantly **exteroceptive** role (no direct synapses onto leg
   motor neurons; ascending and intersegmental targets) versus proprioceptive claw/hook (direct/indirect
   connections to leg motor neurons).
6. **Takemura et al. 2024 / Marin et al. 2024 (MANC)** — source of the `SNpp*` leg-sensory codes used as
   BANC Primary Cell Type. Referenced in the FBbt definitions of the SNpp terms with
   doi:10.7554/eLife.96084.1, doi:10.7554/eLife.97769.1 and doi:10.1101/2023.06.05.543407 / 543757.
7. **[Bates et al. 2025 / eLife reviewed preprint 97766](https://doi.org/10.7554/eLife.97766)** — "Systematic
   annotation of a complete adult male Drosophila nerve cord connectome" (the BANC paper). Source of the
   BANC cell-type labels used in the wiki; the VFB licence notes cite it as "Bates et al. (2025)".

**Correspondence table between nomenclature systems (as far as it is published):**

| Functional subclass | BANC community label | FlyWire/FANC | MANC `SNpp` code | FBbt term | Scoloparium | Classical *Drosophila* anatomy |
|---|---|---|---|---|---|---|
| claw | `claw chordotonal neuron` | claw | `SNpp50` (BANC cell observed) | FBbt:00049557 | ventral | FeCO 1/2/3, mapping **unknown** |
| hook | `hook chordotonal neuron` | hook | `SNpp39` (BANC cell observed; `SNpp57_R` also exists in male-CNS explorer) | FBbt:00049558 | lateral (flexion) + ventral (extension) | FeCO 1/2/3, mapping **unknown** |
| club | `club chordotonal neuron` | club | not determined by me | FBbt:00049556 | dorsal | FeCO 1/2/3, mapping **unknown** |
| (nothing in BANC) | — | — | — | — | — | FeCO 1/2/3 counts 32 / 14 / 25–28 scolopidia |

I attempted to enumerate the SNpp codes programmatically. The FBbt ontology (fbbt.obo master, downloaded in
this session) contains **18** SNpp terms only (SNpp12/13/14/15/20/21/23/25/34/35/42/44/46/48/49/55/61/62) —
**none of which are the leg-FeCO ones** — and those 18 all describe haltere/wing/notum receptors. This means
the leg SNpp codes are a MANC/BANC-side naming that has not been merged into the anatomy ontology, and the
authoritative source would be the BANC `cell_info`/`codex_annotations` tables or the BANC paper supplement.

---

## Q4. Is there a published tibia-joint receptor, and which cell type is it?

**Yes — but it is the claw neuron, and it is not a separate organ from the FeCO.**

Three facts combine:

1. **No leg chordotonal organ other than the FeCO is known in *Drosophila*.** The Drosophila Anatomy
   Ontology's `chordotonal organ of leg` (FBbt:00100292, synonym **Lco**) has exactly one child organ,
   `femoral chordotonal organ` (FBbt:00007592). The ontology's entry for `mechanosensory neuron of leg
   chordotonal organ` even carries the comment: "Tsubouchi et al., 2017 have a diagram showing **tibial**
   chordotonal receptor neurons, but they do not clearly show these among their example neurons (Lco1N1 and
   Lco1N2)." So a tibial chordotonal organ has been drawn but is **not established** in the ontology.
   (Contrast the classical insect literature, where the locust and stick insect do have separate leg
   chordotonal organs, including tibial ones — see [Field & Matheson 1998](https://doi.org/10.1016/S0065-2806(08)60038-4)
   as the standard review, which I did not access in full.)
2. **All three BANC cell types are femoral, and all attach (via arculum + tendons) to the femur–tibia joint.**
3. **Claw is the position channel:** "Here, we found that claw neurons encode femur-tibia joint position"
   ([Mamiya 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/), abstract-level and Fig. 5). FBbt
   encodes the same: claw "encodes sensory information about tibial position" while hook encodes "directional
   tibial movement" and club "tibial vibration and bidirectional movement".

Therefore the project's **tibia-angle signal belongs on the claw channel.**
The hook channel should receive **tibia joint angular velocity, ideally signed**, and the club channel should
receive **tibia vibration (and optionally the bidirectional movement signal)** — not a mean over several
joints. If the project only has one tibia dof (the femur–tibia joint, which is what these neurons encode),
then claw ← tibia **angle** is correct and defensible, and hook ← tibia **angular velocity** is correct and
defensible. What is not defensible is anything that mixes in the *tibia–tarsus* joint or other leg joints for
the club channel.

**Important caveat on the name "TIBIA joint":** in the fly leg there are two joints adjacent to the tibia —
**femur–tibia** and **tibia–tarsus**. Everything in this report concerns the **femur–tibia** joint, which is
the one carried by the FeCO and the arculum. I found **no** published *Drosophila* FeCO-subtype receptor for
the **tibia–tarsus** angle. Position/load information about the tibia–tarsus joint and about leg loading is
carried by **campaniform sensilla** (mechanosensory neuron of leg campaniform sensillum, FBbt:00048058) and
**hair plates** — a different family of receptors, with different cell-type names, and not in the BANC
chordotonal list used by this project
([Lee et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/); [BANC wiki](https://github.com/jasper-tms/the-BANC-fly-connectome/wiki/Annotations-(cell-types,-etc.))).

---

## Q5. Where the natural stimulus is NOT a joint angle

Only **claw** (and the extension-tuned / flexion-tuned division within it) encodes something that is
correctly described as a joint angle. Two of the three channels should not be modelled as angle:

- **hook — natural stimulus is joint velocity with a sign, not angle.** Published quantity: phasic
  responses during joint movement, tested over **100–800 °/s**; never tonic during hold; DSI = 0.811 ± 0.028.
  Directions are carried by separate cell populations (hook flexion vs hook extension), so a signed velocity
  input is the right abstraction. No °/s gain is published.
- **club — natural stimulus is (a) bidirectional joint movement and (b) low-amplitude high-frequency
  vibration, i.e. a *displacement amplitude at a frequency*, not an angle.** Published quantities:
  peak-to-peak vibration amplitude **0.054 µm and 0.9 µm** (not an amplitude response curve), frequency
  **100–2000 Hz**, response peaks at **400 Hz** (0.9 µm) and **800 Hz** (0.054 µm); the population is
  tonotopically organised; and [Lee et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/) argue
  from connectivity that club is predominantly **exteroceptive** (substrate vibration), not proprioceptive.
  Club also responds to large movements, so it is not purely exteroceptive — but a *mean over several leg
  joints* is not a quantity any source assigns to it.
- **claw — the natural stimulus is best described as strain on the dendrites produced by tendon tension, with
  joint angle as the controlled variable that modulates it.** Mamiya 2023 is explicit that the proximal
  mechanotransduction variable is tendon/dendritic **strain** (their model predicts a strain gradient across
  the cell array), because strain could not be measured directly *in vivo*. So "claw encodes joint angle" is
  a correct *behavioural* description, but the physical variable at the receptor is strain — relevant if the
  project ever wants to model adaptation or the goniotopic gradient.

---

## WHAT THIS DOES NOT SUPPORT

| Project's current mapping | Verdict | Basis |
|---|---|---|
| `claw_chordotonal_organ_neuron` ← **tibia joint deviation/angle** | **SUPPORTED** | Claw cells are tonic, non-adapting, and "encode femur-tibia joint position"; activity increases ~linearly with joint angle; flexion- and extension-tuned cells exist separately, so model a *pair* of units (or a signed input) rather than one. Hysteresis is real and should be expected. ([Mamiya 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/)) |
| `hook_chordotonal_organ_neuron` ← **tibia joint velocity** | **SUPPORTED as to the physical variable, with caveats** | The variable is right (phasic, direction-selective movement encoding; no tonic hold response). But (a) direction must be **signed**, because flexion and extension are carried by different cells (hook flexion in lateral scoloparium, hook extension in ventral); (b) hook is **not** gain-flat to velocity in principle — the published data show a similar calcium slope from 100–800 °/s, which does not establish linearity; (c) hook receives the strongest presynaptic inhibition of the FeCO types (majority GABAergic input, strongest from 9A neurons) — [Lee et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/). |
| `club_chordotonal_organ_neuron` ← **mean deviation over several leg joints** | **REFUTED** | No source assigns club a multi-joint quantity. Club is defined as bidirectional **femur–tibia** movement plus low-amplitude high-frequency **tibia vibration** (100–2000 Hz, peak at 400/800 Hz, tested amplitudes 0.9 µm and 0.054 µm p–p). Its downstream targets integrate across legs, but that is a *central* pooling property of its postsynaptic partners (0A/0B, 8B, 9A, 10B interneurons), not a property of the receptor's stimulus. ([Mamiya 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/); [Lee et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/)) |
| Implicit assumption that claw/hook/club are three separate **organs** that could each be assigned a different joint | **REFUTED** | They are three compartments/subpopulations of **one** organ (FeCO, in the proximal femur), all coupled to the **femur–tibia** joint through the arculum. Assigning them to different joints is not anatomically expressible. ([Mamiya 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10644877/)) |
| Implicit assumption that a **tibia–tarsus** angle channel exists among these three cell types | **REFUTED / literature is silent, and the named cell types are femoral** | No *Drosophila* FeCO subtype is reported to transduce the tibia–tarsus joint. Tibia–tarsus and load information is campaniform sensilla / hair plate territory, which is a different BANC annotation family. |

**Concrete recommendation implied by the literature** (the literature does not state this, it follows from
the mapping above): give `claw` the femur–tibia **angle**, `hook` the femur–tibia **signed angular velocity**,
and `club` the femur–tibia **movement magnitude plus any substrate/tarsal vibration signal** — and do not feed
club a cross-joint mean.

---

## DISAGREEMENTS AND GAPS

### Numbers I could not find (stated plainly, not estimated)

1. **No gain in spikes/s/degree or spikes/s/rad for any *Drosophila* FeCO neuron exists in the literature I
   could access.** The three functional *Drosophila* papers used GCaMP6f two-photon calcium imaging for the
   sensory neurons. There is no single-unit electrophysiology of *Drosophila* claw/hook/club afferents. Any
   `spikes/s per degree` figure fed into the MuJoCo model will therefore be a **modelling choice, not a
   published constant** — this should be documented as such in the project.
2. **No absolute threshold is published for claw, hook or club**, in degrees for claw/hook or in µm for club.
   The only published threshold statements are relational: claw's mechanosensory threshold is **>10× higher**
   than club's ([Mamiya 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10644877/)), and 0.054 µm is the
   smallest vibration amplitude *tested* (so an upper bound on club threshold).
3. **No velocity gain (°/s → response) is published for hook** beyond the qualitative statement that calcium
   slope was similar across 100–800 °/s and that DSI = 0.811 ± 0.028.
4. **No full published SNR / detectability curve for club** across amplitudes.
5. **The `SNpp` codes for club (and the complete BANC SNpp ↔ claw/hook/club table) are not published in the
   FBbt ontology.** I could retrieve `SNpp50` (= claw) and `SNpp39` (= hook) from VFB per-cell pages, and
   `SNpp57_R` exists as a page in the Reiser-lab male-CNS cell-type explorer, but I could not access the BANC
   `cell_info` / `codex_annotations` CAVE tables from this environment (CAVE endpoints 404; `pip` and
   `caveclient` unavailable). Treat "SNpp50/57/58/59 ↔ claw/hook/club" as **unverified**.
6. **I could not access several paywalled primaries in full text**, so I report only what the accessible text
   or abstract states: **Phillis et al. 1996** (origin of the claw/club names) — cited second-hand;
   **Field & Matheson 1998** (insect chordotonal organ review); **Hofmann, Koch & Bässler 1985** (stick
   insect *Cuniculina impigra* FeCO physiology, *J Exp Biol* 114:207); and **locust FeCO single-unit
   physiology (Field & Pflüger 1989; Theophilidis 1985)** —
   these are the papers that *do* contain firing-rate-versus-angle relations, but for **locust/stick insect,
   not *Drosophila***, and I did not read them, so **no locust/stick-insect gain is quoted in this report**.
   Publisher pages for *J Exp Biol* and *Nature* returned HTTP 403 from this environment.
7. **Walking joint-angle ranges** for freely walking flies: the controlled-stimulus papers use 18°–180°, but
   I could not verify a published *behavioural* femur–tibia angle range (e.g. from
   [Chockley et al. 2022, JEB](https://doi.org/10.1242/jeb.244245), which is paywalled here). Note that
   Mamiya 2018 report they could not fully flex the femur–tibia joint in their rig because the tibia/pin
   collided with the abdomen — so **180° is the tested maximum for extension, not necessarily the anatomical
   maximum**.

### Contradictions between sources

1. **Neuron counts disagree.** Mamiya 2018: FeCO contains **~135** cell bodies and `iav-Gal4` labels only
   **~80%** of them. Agrawal 2020: **152** neurons. Mamiya 2023: **152** total cell bodies reconstructed,
   and `iav-Gal4` labels ~80% of ~135. Kuan et al. 2020 (per FBbt) gives **152** for the prothoracic leg
   ([FBbt:00048052](https://www.ebi.ac.uk/ols4/api/ontologies/fbbt/terms?iri=http%3A%2F%2Fpurl.obolibrary.org%2Fobo%2FFBbt_00048052)).
   FBbt's parent term says "around 135 per leg". This is a genuine ~11% discrepancy, probably
   counting-vs-reconstruction differences; **use 135–152 as a range**.
2. **Subtype counts disagree across methods.** Mamiya 2018's driver lines labelled **~20 claw**, **~30 club**
   and only **3 hook** cell bodies; but Lee et al. 2025 reconstructed **37 club** axons in FANC (and quote
   n = 13, 9, 13, 37 etc. for subtypes), and Agrawal 2020 records from n = 37 hook regions. So the driver-line
   counts are **not** the true population sizes and should not be used as such.
3. **Scoloparium-group terminology is self-inconsistent in the primary source.** Mamiya 2023 labels its
   compartments "Group 1/2/3" but also states the compartments in a different order, and FlyBase curators
   explicitly refused to add group-1/2/3 as synonyms because of this inconsistency. **Do not use
   "group 1/2/3"**; use dorsal/ventral/lateral scoloparium.
4. **The classical anatomy and the connectome nomenclature have never been reconciled.** Shanbhag et al.
   1992 (FlyBase FBrf0056378) describe **three** femoral organ groups with **32, ~14 and 25–28 scolopidia**
   = **71–74 scolopidia**, and each scolopidium carries **two** neurons, predicting ~142–148 neurons — which
   sits between the 135 and 152 estimates but does not resolve them. There is **no published correspondence**
   between FeCO 1/2/3 and dorsal/ventral/lateral scoloparium, or between either and claw/hook/club.
5. **Club's functional label is genuinely dual in the literature and readers should not flatten it.** Mamiya
   2018/2023 call club a proprioceptor (bidirectional movement + vibration); Lee et al. 2025 argue it is
   predominantly **exteroceptive** for substrate vibration, with weak/no local motor connectivity. The FBbt
   term keeps it inside the "proprioceptive system" hierarchy
   ([FBbt:00049556](https://www.virtualflybrain.org/blog/2022/01/01/femoral-chordotonal-claw-neuron-fbbt_00049557/)).
   Either way, **multi-joint mean deviation** is inconsistent with both readings.
6. **An explicit contradiction the primary source raises against earlier work:** Mamiya 2023 state they "did
   not observe efferent innervation of the FeCO, nor did we see connections between the FeCO and the
   surrounding muscles, as was reported in a previous study." Worth knowing if the model assumes muscle
   coupling to the organ.

---

### Sources actually opened in this session

- [Mamiya, Gurung & Tuthill 2018, *Neuron* 100:636–650 — Neural coding of leg proprioception in *Drosophila*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/) (doi:10.1016/j.neuron.2018.09.009)
- [Mamiya et al. 2023, *Neuron* 111:3230–3243.e14 — Biomechanical origins of proprioceptor feature selectivity and topographic maps in the *Drosophila* leg](https://pmc.ncbi.nlm.nih.gov/articles/PMC10644877/) (doi:10.1016/j.neuron.2023.07.009)
- [Agrawal et al. 2020, *eLife* 9:e60299 — Central processing of leg proprioception in *Drosophila*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7752136/) (doi:10.7554/eLife.60299)
- [Lee, Dallmann, Cook, Tuthill & Agrawal 2025, *Nat Commun* 16:4105 — Divergent neural circuits for proprioceptive and exteroceptive sensing of the *Drosophila* leg](https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/) (doi:10.1038/s41467-025-59302-3)
- [BANC connectome wiki — Annotations (cell types, etc.), `cell_info` full annotation list](https://github.com/jasper-tms/the-BANC-fly-connectome/wiki/Annotations-(cell-types,-etc.))
- [Virtual Fly Brain — femoral chordotonal claw neuron FBbt:00049557](https://www.virtualflybrain.org/blog/2022/01/01/femoral-chordotonal-claw-neuron-fbbt_00049557/)
- [Virtual Fly Brain — femoral chordotonal hook neuron FBbt:00049558](https://virtualflybrain.org/blog/2022/01/01/femoral-chordotonal-hook-neuron-fbbt_00049558/)
- [Virtual Fly Brain — BANC_626:720575941413112212 (claw; SNpp50)](https://www.virtualflybrain.org/blog/2022/01/01/banc_626720575941413112212-vfb_00106ggj/)
- [Virtual Fly Brain — BANC_626:720575941438410028 (hook; SNpp39)](https://www.virtualflybrain.org/blog/2022/01/01/banc_626720575941438410028-vfb_00106c20/)
- [Drosophila Anatomy Ontology (fbbt.obo, master) — queried locally for FeCO, scoloparium, SNpp and leg-chordotonal terms](https://github.com/FlyBase/drosophila-anatomy-developmental-ontology)
- [OLS4 API — FBbt term records (FBbt:00052626/27/28, 00049556/57/58, 00007592, 00048051/52/74/75/76, 00004271–44)](https://www.ebi.ac.uk/ols4/ontologies/fbbt)
- [Crossref — DOI verification for 10.7554/eLife.97766 (Bates et al., BANC) and 10.1242/jeb.244245 (Chockley et al. 2022)](https://api.crossref.org/works/10.7554/eLife.97766)
- [Reiser lab — Male CNS Cell Type Explorer, SNpp57_R](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/types/SNpp57_R.html)

*Not opened in full text (paywalled / blocked from this environment): Phillis et al. 1996; Field & Matheson 1998; Hofmann, Koch & Bässler 1985 (JEB 114:207); Field & Pflüger 1989; Chockley et al. 2022 (JEB, doi:10.1242/jeb.244245); Kuan et al. 2020.*
