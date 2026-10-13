# 接口文档（从代码生成）

> **本文件由 `tools_interface_doc.py` 从 AST 生成，不手工编辑。**
> 改了模块就跑一次 `./venv/bin/python tools_interface_doc.py`；用 `--check` 可以查出**与上次相比哪些签名变了**。

生成方式：`tools_interface_doc.py (AST, no imports)`　模块数：**55**


---

## 1. 闭环的四条接缝（本平台最该先看的一张表）

| 接缝 | 方向 | 状态 | 实现 | 边界 |
|---|---|---|---|---|
| **SensoryDrive** | body -> brain | **IMPLEMENTED** | `MechanoReceptorBank` | mechanoreception AND vision now, with three measured caveats.  (1) BANDWIDTH: the default tier holds 11 visual-class neurons of 12,000 (0.092 %) against 79,538 in the dataset -- the tier captures 0.014 % of the visual population, and the eye-side annotation is absent for all 11 (left 0, right 0, unassigned 11), so the two eye channels CANNOT be separated.  (2) THE CONVERSION IS ASSUMED: `drive_scale_nA_per_unit` is a placeholder with a sweep, because this project has no measured photoreceptor-to-projection gain; `visual_drive_nA` still refuses to run without it.  (3) A CONFIGURATION CONFLICT: enabling `TierConfig.include_visual_route` rebuilds the selection and leaves leg `lh` with ZERO descending neurons, so its per-leg command is UNAVAILABLE and held neutral -- the six-leg channel and the visual route cannot both be full at this tier size.  Also MEASURED and worth knowing: with the default n_channels=2 the drive is TEMPORAL contrast, which is exactly zero for a still scene, so this channel only responds to motion unless n_channels=4 (temporal + spatial) is used.  FIXED EN ROUTE: `vision.py`'s `bind()` never re-initialised its state after discovering the ommatidia count, so the first `read()` ALWAYS raised a broadcast error -- the front end was unusable, which is very likely why vision was never wired.  `bind()` now resets, and a shape guard names the fix. |
| **MotorCommand** | brain -> body | **IMPLEMENTED** | `CommandedTripodCPG` | SIX knobs now: one intrinsic-frequency scale per leg, each a function of that leg's own descending readout, with leg specificity measured at 0.00 % cross-leg (the scalar speed/turn path is retained bit-identically for historical episodes).  The GAIT itself is still a pre-recorded motion-capture chase, so the connectome MODULATES the gait; it does not generate it.  112 of 195 readout neurons reach no leg's motor neurons and carry no leg-specific command. |
| **NeuralRecorder** | brain -> electrode | **IMPLEMENTED** | `InLoopElectrode` | the spike -> current kernel is ILLUSTRATIVE (charge per spike, pulse duration), and 4263 of 12000 tier neurons have no soma position, so 64 % of the network is recordable.  Only the electrode-network COUPLING is established, never the amplitude.  CAUSE, DIAGNOSED: those 4263 root ids are simply ABSENT from somas_v1.parquet -- not invalid (0 rows have valid=False), not at the origin (0 rows), and not a query or resolution artefact.  The other locally available BANC file (banc_neurons.csv.gz) carries annotations only and NO coordinates, so this coverage CANNOT be raised without another data source; it can only be declared, which is why 7737 must never be described as the whole network. |
| **NeuralStimulus** | electrode -> brain | **IMPLEMENTED** | `InLoopStimulus` | the per-phase charge is checked against the stim module's own limits -- a MEASURED_CITED density ceiling (Shepherd et al.) and an ASSUMED current ceiling with a sweep -- and those provenances are carried through.  THE ASSUMPTION IS THE SIGN, and it was misdescribed at first: MEASURED, at the tier's dt (0.5 ms) the pulse's recovery phase falls between substeps and is never sampled (sampled signs = {+}), so both modes inject the same cathodic train and differ ONLY in sign -- recruiting depolarises (12544 spikes/30 substeps), the literal sign hyperpolarises (0).  Charge balance is therefore NOT the mechanism and no self-cancellation occurs.  RECRUITMENT IS TINY: the DECLARED 50 um capture radius reaches 53 neurons = 0.7 % of the 7737 recordable, 25 um reaches 11, and 10 um reaches ZERO (swept, not asserted).  So amplitude effects are meaningful; the spatial recruitment pattern is not.  NO charge-safety claim. |

读法：`DECLARED_NOT_IMPLEMENTED` 不是遗漏，而是**平台形状的一个事实** ——
它意味着那条信息流**不存在**（例如电极刺激从未注入神经元）。


### 1b. 能**同时**观测的通道（与上表是两个不同问题）

同一次 `run()`、同一个躯体、同一个 interval 计数器里真正一起录下来的通道：**2** 个 —— `camera_ring`, `electrode`；结构性缺席：`vision`。

| 通道 | 所在栈 | 能否同时 | 实测 | 边界 |
|---|---|:--:|---|---|
| `camera_ring` | B (FlyMimic muscles) | ✅ | 6 cameras, 20 frames, health gate PASS at 80.6 % fiducial hit rate; reconstruction reaches 7 of 12 fiducials, world RMS 0.9663 mm.  THE FLY'S OWN MARKERS ARE NOW ATTACHED -- 20 of them, one per segment, massless -- so `limb_tracking.py` runs on these episodes: 119 of 400 frame-marker slots accepted, marker position error median 0.0675 mm, and JOINT ANGLES come out (leg RF's femur in 16 of 20 frames, median error 0.939 deg, p95 2.673 deg) | the angles are PARTIAL, and that is the honest headline: only ONE bone clears 10 frames, because a six-camera ring around a SELF-OCCLUDING body leaves most markers visible from one or two cameras per frame while triangulation needs three.  Frame 0's marker identity is seeded from the SIMULATOR as a stand-in for a human annotation -- every marker is the same sphere, so the pixels alone cannot say which is which -- and the truth the errors are scored against is the simulator's own: there is no real-fly recording anywhere in this project.  All six legs bind their FEMUR, because stack B has no trochanter on the mid and hind legs, so a 'femur' bone here is MARKER-TO-MARKER, not joint-centre-to-joint-centre.  DEFECTS FIXED HERE, all measured: (1) the recorder threw away each blob's PIXEL AREA, which is the only thing that can separate the arena's 0.30 mm fiducials from the fly's 0.12 mm markers, so the nearest-blob association let the fly's markers steal the floor-level fiducial tier and the fiducial triangulation fell from 7/12 to 3/12.  A global area cutoff was then tried and MEASURED not to work -- the two size distributions nearly touch (fly markers to 37 px, fiducials from 39 px) because perspective makes a near fly marker as large as a far fiducial -- so the test is now DEPTH-RELATIVE, with an effective radius CALIBRATED from the data: 0.230 mm measured against 0.300 mm nominal, because the detector thresholds at grey 240 while the emissive markers peak near 251 and only the saturated core is counted.  (2) The recorders converted RGB to grey with the arithmetic MEAN while engine/embodied/natural_scene.py uses standard LUMINANCE; the mean understates the warm emissive markers (243 against 250.9 at the brightest pixel), and that is what destroyed the size separation.  Both now go through `arena.marker_grey`.  (3) THE ONE THAT MATTERED MOST: the per-camera offset fitted on the static fiducials was being applied to the fly's markers, and it is MEASURED to make them worse -- median residual 1.57 px without it against 2.92 px with it, which took the triangulated fly markers from 19 of 20 to 0 of 20 and silently produced an EMPTY joint-angle table.  It stays off, and `limb_tracking` records `projection_offsets.applied` in its output. |
| `electrode` | B (FlyMimic muscles) | ✅ | 4 channels, fed the POST-lesion spikes -- the ones that set the muscle activations -- on the same interval counter as the frames; coupling guard PASS | 7737 of 12000 tier neurons have a soma position, so 64 % is recordable; the spike -> current kernel is ILLUSTRATIVE, so amplitude is never established. |
| `vision` | A (NeuroMechFly, CPG replay) ONLY | ❌ | binding the front end to stack B FAILS with `AttributeError: 'mujoco._structs.MjData' object has no attribute 'get_raw_vision'` -- it needs a FlyGym `Simulation` and stack B is a raw `MjData` built from FlyMimic's MJCF | so this platform observes TWO channels on the muscle-driven body, never three.  The visual channel also carries the bandwidth shortfall registered under SensoryDrive (11 of 12000 tier neurons, 0.092 %). |

**该文件修掉的缺陷**：the first version wrote `truth/time_s = interval_times[:n_frames]`, which understated the frame clock by exactly `stride` -- MEASURED: declared 0..0.095 s while the frames were rendered at 0..0.38 s (dt ratio 0.25 = 1/4).  Any velocity across frames would have been 4x wrong, and the 'shared clock' claim was false in the artifact itself.  `record()` now derives the frame times from the recorded indices and raises if `frame_dt != stride * DT_COMMAND_S`.

**注入验证**：ONE lesion must move BOTH channels of the same run, and it is checked: silencing leg LF's motor neurons changes the last frame (mean |diff| 0.0177) and the electrode trace (max |diff| 0.005072 mV) in the SAME episode.


---

## 2. 模块索引（按体量排序，`units`/`ILLUSTRATIVE` 两列表示该模块是否自带声明）

| 模块 | 行数 | 自身声明的用途 | 声明单位 | 声明 ILLUSTRATIVE | 声明未确立 |
|---|---:|---|:--:|:--:|:--:|
| `engine/embodied/slow_physiology.py` | 2609 | engine.embodied.slow_physiology -- SLOW PHYSIOLOGY state + behaviour/rest recorder. | ✅ | ✅ | — |
| `engine/electrode_sorting.py` | 2082 | SOURCE SEPARATION (spike sorting / demultiplexing) for the FIXED 20 um-pitch electrode array of this project. | ✅ | — | ✅ |
| `engine/electrode_frontend.py` | 1977 | REAL recording front end for the electrode array: finite CONTACT-AREA averaging, electrode-electrolyte INTERFA | ✅ | — | ✅ |
| `engine/touch_damage_scenario.py` | 1937 | ONE scenario that puts the data-anchored neck cut and the micrometre-electrode damage model into the same pair | ✅ | ✅ | — |
| `engine/embodied/natural_scene.py` | 1899 | A genuinely NATURAL scene layer, built ON TOP of ``engine.embodied.scene``. | ✅ | — | — |
| `engine/electrode_damage.py` | 1870 | engine.electrode_damage -- mechanical damage from inserting a micrometre-scale electrode into Drosophila nervo | ✅ | ✅ | — |
| `engine/embodied/scene.py` | 1849 | Declarative, parameterised scene layer for the embodied fly body. | ✅ | — | — |
| `engine/electrode_tissue.py` | 1588 | engine.electrode_tissue -- TISSUE MECHANICS of electrode insertion into Drosophila nervous tissue, and its con | ✅ | ✅ | — |
| `engine/embodied/access_map.py` | 1582 | Channel <-> neuron ACCESS MAP: where a fixed-geometry electrode array could be placed, in the REAL voxel coord | ✅ | — | ✅ |
| `engine/neck_cut_data.py` | 1508 | Data-anchored neck cut for the BANC body-touch -> ascending -> brain route. | ✅ | — | — |
| `engine/embodied/adapters.py` | 1503 | engine.embodied.adapters -- the neural <-> body adapters for the first closed loop. | ✅ | ✅ | — |
| `engine/embodied/plasticity.py` | 1365 | engine.embodied.plasticity -- a SWITCHABLE plasticity layer for the neural tier. | ✅ | ✅ | — |
| `engine/electrode_stim.py` | 1248 | STIMULATION extension of the electrode model: current injection -> extracellular field -> neural activation, u | ✅ | ✅ | ✅ |
| `engine/da_protocol.py` | 1146 | Illustrative dopamine release/clearance model driven by a MEASURED protocol. | ✅ | ✅ | — |
| `engine/multirate_scenario.py` | 1126 | Bounded multirate scenario: BANC coarse tier + histamine vision + injury + HH neck. | ✅ | ✅ | — |
| `engine/synapse_map.py` | 1097 | Real FlyWire presynaptic site coordinates mapped onto the validated FAFB subtree. | ✅ | ✅ | — |
| `engine/embodied/body_backend.py` | 981 | Body backend: a physical adult-fly body in a scene, wrapped behind one API. | ✅ | — | — |
| `engine/embodied/vision.py` | 914 | engine.embodied.vision -- the visual pathway that the embodied loop did not have. | ✅ | — | — |
| `engine/embodied/loop.py` | 876 | engine.embodied.loop -- ONE multirate scheduler that owns time, and the episode record. | ✅ | ✅ | — |
| `run_arena_reconstruct.py` | 853 | MULTI-CAMERA RECONSTRUCTION: markers -> 3D points -> ABSOLUTE angles. | ✅ | — | — |
| `limb_tracking.py` | 797 | IDENTITY-SAFE LIMB TRACKING: the anomaly filter that makes a joint angle trustworthy. | ✅ | — | — |
| `run_arena_record.py` | 600 | SIX-CAMERA HIGH-SPEED RECORDING in the grass meadow. | ✅ | — | — |
| `engine/embodied/electrode_in_loop.py` | 538 | ELECTRODE IN THE CLOSED LOOP.  The seam that was missing between the brain and the body. | ✅ | ✅ | ✅ |
| `engine/embodied/seams.py` | 506 | THE FOUR SEAMS OF THE CLOSED LOOP, AS EXPLICIT INTERFACES. | ✅ | ✅ | ✅ |
| `engine/neural.py` | 434 | engine.neural -- spiking neural layer for the fly nervous system. | ✅ | ✅ | — |
| `engine/cable.py` | 351 | Passive multicompartment cable approximation (not an active neuron backend). | ✅ | — | — |
| `engine/receptors.py` | 329 | engine.receptors -- sensory transduction: environment -> receptor current. | ✅ | ✅ | — |
| `engine/layers.py` | 308 | engine.layers -- the layered, strictly feed-forward engine core. | ✅ | — | — |
| `run_tactile_record.py` | 300 | TACTILE RECORDING: per-leg contact FORCE and contact AREA over a walking episode. | ✅ | — | — |
| `engine/units.py` | 289 | engine.units -- explicit physical units with dimensional arithmetic. | ✅ | — | — |
| `engine/sensorimotor.py` | 281 | engine.sensorimotor -- the sensorimotor interface to a virtual environment. | ✅ | — | — |
| `engine/local_tissue.py` | 279 | README — illustrative conservative local tissue / neuromodulation prototype. | ✅ | ✅ | — |
| `engine/neural_active.py` | 276 | Bounded, idealized active NEURON cables -- NOT fly-calibrated physiology. | ✅ | ✅ | — |
| `engine/neural_hist.py` | 263 | Dedicated histamine conductance channel on top of ConductanceNetwork. | ✅ | ✅ | — |
| `engine/project_layers.py` | 227 | engine.project_layers -- wrap the project's existing modules as engine layers. | ✅ | — | — |
| `engine/params.py` | 217 | engine.params -- parameter registry with ENFORCED provenance. | ✅ | ✅ | — |
| `engine/profile.py` | 217 | engine.profile -- organism/scale profiles that encode verified constraints. | ✅ | ✅ | — |
| `engine/rules.py` | 215 | engine.rules -- a small declarative rule grammar for cell behaviour. | ✅ | ✅ | — |
| `engine/embodied/force_channel.py` | 210 | THE CONTACT-FORCE CHANNEL IS A NON-PHYSICAL LOAD.  ONE DECLARATION, ENFORCED. | ✅ | — | ✅ |
| `engine/injury_tissue.py` | 208 | Illustrative injury reduction, NOT measured fly injury/viability physiology. | ✅ | ✅ | — |
| `engine/embodied/visual_drive.py` | 202 | VISUAL DRIVE: the body -> brain channel the platform declared but never wired. | ✅ | — | ✅ |
| `engine/isolation_scenario.py` | 191 | Integrated IDEALIZED graph experiment, hand implemented, not fly physiology. | ✅ | ✅ | — |
| `engine/electrode_recording.py` | 176 | Fixed-geometry recording diagnostic. One-sided PSDs in V²/Hz. Analog anti-aliasing precedes ideal sampling; ca | ✅ | — | — |
| `engine/neural_cond.py` | 176 | Conductance point tier, explicit units and illustrative parameters. | ✅ | ✅ | — |
| `engine/embodied/bridge_body.py` | 170 | Independent engineered intent -> shipped FlyGym hybrid turning controller. | ✅ | — | — |
| `engine/checks.py` | 166 | engine.checks -- the verification harness. | ✅ | — | — |
| `engine/neural_hybrid.py` | 162 | Fixed-owner bounded hybrid coordinator, NOT a whole-brain implementation. | ✅ | ✅ | — |
| `engine/record.py` | 156 | engine.record -- self-describing run records. | ✅ | ✅ | — |
| `engine/electrode.py` | 121 | Finite spherical volume contacts in infinite homogeneous ohmic medium. | ✅ | ✅ | — |
| `engine/embodied/neural_motor_bridge.py` | 92 | Engineered, literature-informed spike-to-intent bridge (not anatomical reconstruction). | ✅ | — | — |
| `engine/graded_vision.py` | 74 | Continuous graded photoreceptor release with an explicit chloride contract. | ✅ | ✅ | — |
| `engine/embodied/neural_command_audit.py` | 52 | Executable checks of neural-only command origin, not biological validation. | ✅ | — | ✅ |
| `engine/embodied/feco_identity.py` | 43 | Resolve BANC FeCO subtype by exact CSV subclass, never name similarity. | ✅ | — | — |
| `engine/embodied/feco_translation.py` | 36 | Fail-closed FeCO input translation prerequisites; no movement decisions. | — | — | ✅ |
| `engine/embodied/autonomy_contract.py` | 25 | Fail-closed reporting contract; not proof that a runtime has no hidden controls. | ✅ | — | — |

---

## 3. 与上次生成的差异（drift）

**无变化。** 所有公开签名与上次一致。


---

## 4. 逐模块公开接口

每个条目给出**完整签名**与作者自己写的第一行契约，所以改一个模块之前不需要通读其它模块。


### `engine/embodied/slow_physiology.py`  (2609 行)

engine.embodied.slow_physiology -- SLOW PHYSIOLOGY state + behaviour/rest recorder.

项目内依赖：`..local_tissue`, `.loop`

- **`class TrachealConfig`**
    The tracheal supply route.  All parameters ILLUSTRATIVE.
    - `def validate(self)`
    - `def bath_o2_from_partial_pressure(self)` — nmol/mm^3 from the stated partial pressure and solubility (Henry's law).
    - `def units(self)`
- **`class TissueConfig`**
    The consuming compartment.  All parameters ILLUSTRATIVE.
    - `def validate(self)`
    - `def units(self)`
- **`class HaemolymphConfig`**
    The SEPARATE circulating compartment.  It carries NO O2.
    - `def validate(self)`
    - `def units(self)`
- **`class LoadConfig`**
    How a MEASURED quantity of the body becomes metabolic demand.
    - `def validate(self)`
    - `def units(self)`
- **`class SlowPhysiologyConfig`**
    Everything that fixes one slow-physiology state.  Validated, refused if not.
    - `def validate(self)`
    - `def as_dict(self)`
    - `def units(self)`
- **`def slowphys_fingerprint(cfg, extra=None)`**
    sha256[:32] of a slow-physiology config.
- **`class TimeJumpError(ValueError)`**
    Raised when a slow step larger than the declared bound is not declared as one.
- **`class TimeJumpRecord`**
    An EXPLICIT, labelled quasi-steady advance of the slow module alone.
    - `def as_dict(self)`
- **`class LocomotorActivity`**
    One command interval of MEASURED body quantities, with units attached.
    - `def units(self)`
    - `def as_dict(self)`
- **`class SubstanceLedger`**
    amount_now = initial + inflow - outflow - consumption, with the residual SHOWN.
    - `def amount_now(self)`
    - `def residual(self)`
    - `def residual_relative(self)`
    - `def compartment_residuals(self)` — Per-compartment identities, for the compartments declared as flows' sides.
    - `def all_supply_terms_nonnegative(self)`
    - `def as_dict(self)`
- **`class LedgerRow`**
    One time sample of every ledger, in one flat row (units carried alongside).
    - `def units(self)`
    - `def as_dict(self)`
- **`class EnergyAvailabilityProxy`**
    ``NonPhysiologicalAvailabilityProxy``: dimensionless availability in [0, 1].
    - `def step(self, dt_s, supply_s=0.0, demand_s=0.0)`
- **`class SlowPhysiology`**
    Finite slow compartments with explicit ledgers, driven by measured load.
    - `def concentration_nmol_per_mm3(self, which)`
    - `def tissue_o2_fraction(self)` — O2 concentration in the tissue relative to the air side, in [0, 1].
    - `def demand_nmol_s(self, load)` — ILLUSTRATIVE absolute consumption from the DIMENSIONLESS measured load.
    - `def measured_load(self, speed_mm_s, joint_speed_rad_s)` — Dimensionless load from the two MEASURED quantities (and their units).
    - `def reduce_interval(self, time_s, duration_s, thorax_xy_mm, joint_speed_sum_rad_s, contact_leg_count)` — Reduce ONE interval of recorded body arrays to a measured activity record.
    - `def step(self, dt_s, activity=None, load=None, demand_nmol_s=None, declared_time_jump=False, record_activity=True)` — Advance the slow state by dt_s, consuming O2 at the measured demand.
    - `def energy_proxy_in_any_ledger(self)`
    - `def sample(self) -> LedgerRow`
    - `def check_ledgers(self, tolerance_nmol=1e-09, tolerance_relative=1e-06)` — Reported diagnostics.  Returns a dict; corrects nothing.
    - `def ledgers(self)`
    - `def traces(self)`
- **`class BehaviourConfig`**
    The pre-registered behaviour criteria, plus the guards that make them usable.
    - `def validate(self)`
    - `def predicted_labels(self)` — The labels the DEMO's scripted phase plan is expected to produce.
- **`class BehaviourMeasures`**
    The per-interval measures the labels are computed from (units in one place).
    - `def units(self)`
    - `def as_dict(self)`
- **`def measure_behaviour(time_s, thorax_xy_mm, heading_rad, contact_leg_count, cfg=None)`**
    Reduce recorded body arrays to the per-interval measures.  Units: mm, deg, mm/s.
- **`def measure_long_scale(time_s, thorax_xy_mm, heading_rad, cfg=None)`**
    The LONG-window measures that tell a loop from a wander.
- **`def classify_behaviour(measures, cfg=None, long_scale=None)`**
    The PRE-REGISTERED decision rule, in this exact order.
- **`def epoch_table(labels, measures, reasons=None)`**
    Collapse a per-interval label series into contiguous epochs, with evidence.
- **`def exploration_coverage(thorax_xy_mm, cell_mm=5.0)`**
    MEASURABLE exploration statistics of a recorded trajectory.
- **`class ModulatorConfig`**
    One local modulator with a labelled SOURCE, a labelled grid and a KNOWN
    - `def validate(self)`
    - `def as_dict(self)`
    - `def units(self)`
- **`class ModulatorLedger`**
    The modulator grid's own ledger, in the grid module's own units.
    - `def residual(self)`
    - `def as_dict(self)`
- **`class LocalModulator`**
    A local release site, a diffusive distribution and a receptor occupancy probe.
    - `def step(self, dt_s)`
    - `def sample(self)`
    - `def load_multiplier(self, load)` — Bounded, monotone effect of occupancy on the illustrative load.
- **`class SlowPhysiologyDriver`**
    Drives a ``MultirateScheduler`` and one ``SlowPhysiology`` with a scripted plan.
    - `def write_command(self, k, speed, turn)` — Write the loop's own two CPG knobs, identically in BOTH arms.
    - `def run(self, episode, schedule=(), trace=True, on_interval=None)` — Run the episode on the scheduler's clock; return (scheduler, episode, driver).
    - `def reduce_behaviour(self, bcfg=None)` — Classify every interval from the MEASURED truth series (thresholds fixed).
    - `def sch_truth(self)`
    - `def attach(self, episode)` — Extract the reduced behaviour inputs from a run episode (mm, deg, count).
- **`def slowphys_report(driver, episode, bcfg=None, ledger_tolerance_nmol=1e-09, wall_seconds=None, peak_ram_mb=None, gl_backend=None, extra=None)`**
    Assemble the JSON-serialisable report for one slow-physiology run.
- **`def save_traces(path, driver, episode, report)`**
    Save every recorded array, with the units attached in the same file.

### `engine/electrode_sorting.py`  (2082 行)

SOURCE SEPARATION (spike sorting / demultiplexing) for the FIXED 20 um-pitch electrode array of this project.

项目内依赖：`.electrode`, `.electrode_frontend`, `.embodied.access_map`

- **`class PROV`**
    The same four classes as the rest of the stack.  There is no fifth.
- **`class ProvenanceRecord`**
    - `def as_dict(self)`
- **`class SortingLedger`**
    Provenance ledger that ENFORCES the honesty rules instead of recording them.
    - `def record(self, key, value, unit, provenance, *, source=None, note='', sweep=None)`
    - `def grid_row(self, key, value, unit, *, source, note='')` — One row of a computed sweep grid: always MEASURED_LOCAL, key may repeat.
    - `def get(self, key)`
    - `def as_list(self)`
    - `def as_dict(self)`
    - `def counts(self)`
    - `def unresolved(self)` — Entries a reader must treat as NOT established from any source.
- **`def per_neuron_gain_matrix(access_map, *, neuron_current_nA=0.01, conductivity_S_m=0.3, contact=None, quadrature=None, normal=(0.0, 0.0, 1.0), amplitude_floor_nA_per_channel=None, neuron_rows=None, progress=None)`**
    Per-neuron, per-channel VOLTAGE gain of the array, in mV.
- **`def noise_floor_uV_rms(interface, *, f_lo_Hz=1e-06, f_hi_Hz=10000.0, include_amplifier=False)`**
    Johnson-Nyquist noise floor of the SAME interface, in uV RMS.
- **`def output_referred_noise_uV(interface, *, n_steps, dt_ms, f_lo_Hz=1e-06, n_grid=20000)`**
    Noise RMS AT THE RECORDING'S OUTPUT, on the trace's OWN frequency grid.
- **`def gaussian_spike_current_nA(t_ms, *, sigma_ms=1.0, peak_nA=1.0, centre_ms=0.0)`**
    A DECLARED single-lobe Gaussian current pulse, in nA.
- **`class SpikeTruth`**
    The DECLARED ground-truth spike set the recorded waveforms were built from.
    - `def as_dict(self, *, max_list=0)`
- **`class SignalModel`**
    Synthesises per-channel waveforms for a stated ground-truth spike set.
    - `def clean_trace_mV(self, spikes)` — Per-channel UNFILTERED-BY-NOISE trace for a spike list, plus the filtered
    - `def record(self, spikes, *, noise_sd_mV=None, seed=None)` — The RECORDED trace: clean waveform plus Gaussian white noise.
    - `def waveform_report(self)` — What the front end did to the DECLARED waveform, with numbers.
- **`class SeparationResult`**
    One demultiplexing run: what the method claimed, and what it was given.
    - `def as_dict(self, *, max_list=0)`
- **`def separate(recorded_mV, templates_mV, waveform_nA, *, method='omp_matched_filter', noise_sd_mV=None, dt_ms=1.0, t_offset_ms=0.0, far_per_test=1e-06, max_per_time=3, max_iterations=None, channel_floor_fraction=0.001, neuron_mask=None, max_detections=None, n_time_bins=None, min_rel_amp=0.05)`**
    Demultiplex one recorded window into spikes and neuron identities.
- **`def evaluate(truth, estimate, *, dt_ms=1.0, tolerance_ms=1.5, neuron_is_shared=None)`**
    Detection and ASSIGNMENT metrics of one estimate against the truth.
- **`def resolvability_geometry(templates_mV, waveform_nA, *, neuron_is_shared=None, max_neurons=600, n_pairs_per_neuron=400, seed=0, coherence_cutoffs=(0.5, 0.9, 0.99, 0.999, 0.9999))`**
    The coherent-pair structure of the templates: the MECHANISM of the ambiguity.
- **`def matched_filter_snr(model_or_templates, waveform_nA=None, noise_sd_mV=None)`**
    Per-neuron matched-filter SNR: ``|G[:,j] . w| / sigma``.
- **`def make_trial_spikes(model, *, n_signal_neurons, rng, neuron_is_shared=None, shared_only=False, min_time_ms=1.0, edge_ms=5.0)`**
    One DECLARED ground-truth spike set: each chosen neuron fires ONCE.
- **`def run_noise_level(model, *, noise_sd_mV, neuron_is_shared, n_trials=8, neurons_in_window=6, method='omp_matched_filter', tolerance_ms=1.5, dt_ms=1.0, far_per_test=1e-06, max_per_time=3, seed=0, compare_methods=None, shared_only=False, keep_per_trial=False, max_detections=None)`**
    One noise level: synthesise -> separate -> evaluate, ``n_trials`` times.
- **`def sweep_noise(model, *, noise_levels_uV, neuron_is_shared, n_trials=8, neurons_in_window=6, far_per_test=1e-06, max_per_time=3, seed=0, dt_ms=1.0, tolerance_ms=1.5, compare_methods=None, shared_only=False, progress=None, keep_per_trial=False)`**
    Sweep the NOISE level with the signal fixed -- the binding constraint.
- **`def separability_limit(model, *, neuron_is_shared, floor_uV=None, levels_uV=None, n_trials=8, resolvable_target=0.5, neurons_in_window=6, far_per_test=1e-06, max_per_time=3, seed=0, dt_ms=1.0, tolerance_ms=1.5, max_refine=6, progress=None, shared_only=False)`**
    The noise level at which separation of the SHARED neurons stops working.
- **`def estimate_templates_from_calibration(model, *, n_calibrated, rng, min_snr=5.0, noise_sd_mV=None, max_channels_per_neuron=24)`**
    Estimate each neuron's channel template from a recording where ONE neuron
- **`def build_sorting_ledger(*, model, gain_info, noise_floor_uV, noise_floor_detail, access_map, spike_sigma_ms, spike_peak_nA, n_neurons_captured, n_shared, sweep_noise_levels_uV, access_map_json_source, far_per_test=1e-06)`**
    Assemble the provenance ledger for a sorting run.

### `engine/electrode_frontend.py`  (1977 行)

REAL recording front end for the electrode array: finite CONTACT-AREA averaging, electrode-electrolyte INTERFACE IMPEDANCE, and CHANNEL CROSSTALK.

项目内依赖：`.electrode`

- **`class PROVENANCE`**
    The four allowed provenance classes.  There is no fifth.
- **`class ProvenanceRecord`**
    - `def as_dict(self)`
- **`class FrontendLedger`**
    Provenance ledger that enforces the honesty rules rather than recording them.
    - `def record(self, key, value, unit, provenance, *, source=None, note='', sweep=None)`
    - `def measured_for_sweep(self, key, value, unit, *, source, note='')` — Record one row of a computed sweep grid.
    - `def get(self, key)`
    - `def as_list(self)`
    - `def as_dict(self)`
    - `def counts(self)`
    - `def unresolved(self)` — Entries a reader must treat as NOT established from any source.
- **`class ContactGeometry`**
    A FINITE contact patch of a stated area, from the FIXED shaft diameter.
    - `def from_diameter(cls, diameter_um, **kw)` — The plain reading of the fixed spec: a bare disc of that diameter.
    - `def radius_um(self)`
    - `def active_radius_um(self)`
    - `def area_um2(self)` — Geometric area of the ACTIVE metal, in square micrometres.
    - `def area_cm2(self)`
    - `def area_m2(self)`
    - `def equivalent_disc_radius_um(self)` — Radius of the equal-area disc (equals ``active_radius_um`` here).
    - `def as_dict(self)`
- **`class ContactQuadrature`**
    The DECLARED tensor-product rule used for every area integral here.
    - `def n_points(self)`
    - `def nodes_weights(self, contact)` — ``(rho, phi, w)`` in the contact's own disc frame, ``w`` in um^2.
    - `def convergence(self, contact, field_fn, *, orders=((4, 16), (8, 32), (16, 128), (32, 512)), source_um=(0.0, 0.0, 4.0), normal=(0.0, 0.0, 1.0))` — Area-average ONE declared field for several orders; report the spread.
- **`def contact_points_um(contact, *, source_um=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), quadrature=None)`**
    Quadrature points and area weights of the ACTIVE metal, in micrometres.
- **`def contact_area_average(field_fn, contact, *, quadrature=None, source_um=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), return_points=False)`**
    AREA INTEGRAL of a potential field over the contact patch, in mV.
- **`def contact_area_average_converged(field_fn, contact, *, coarse=ContactQuadrature(4, 16), fine=ContactQuadrature(16, 128), tol=0.001, source_um=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))`**
    Area average with an EXPLICIT convergence check on THIS field.
- **`def interface_sweep(contact, *, c_dl_uF_per_cm2=(5.0, 10.0, 20.0, 40.0, 100.0), rho_ct_ohm_cm2=(38.5, 385.0, 3850.0), sigma_S_m=(0.3,), amplifier_input_ohm=(math.inf,), f_probe_Hz=1000.0, f_lo_Hz=1e-06, f_hi_Hz=100000.0)`**
    Declared ASSUMED ranges, moved one at a time, with the effect on the output.
- **`def area_average_analytic_point_source(contact, *, centre_um, source_um, q_nA, conductivity_S_m)`**
    EXACT area integral (not the mean) of a point source's 1/r potential over a disc.
- **`def nA_to_A(i_nA)`**
- **`class InterfaceImpedance`**
    Randles-type electrode-electrolyte interface of ONE finite contact.
    - `def area_cm2(self)`
    - `def r_spread_ohm(self)` — Ideal disc/half-space spreading resistance assumption.
    - `def c_dl_F(self)`
    - `def r_ct_ohm(self)`
    - `def r_dc_ohm(self)` — The resistance that sets the DC/low-frequency level of the transfer.
    - `def f_corner_Hz(self)` — Frequency of the +3 dB point, MEASURED NUMERICALLY off :meth:`transfer`.
    - `def dc_transfer(self)` — ``H(0)``: the low-frequency limit of :meth:`transfer` (1.0 if ideal).
    - `def impedance_ohm(self, f_Hz)` — ``Z_e(f)``, complex, in ohms (the interface WITHOUT the amplifier).
    - `def load_impedance_ohm(self, f_Hz)` — ``Z_L(f)``: the amplifier input as the CONTACT sees it.
    - `def transfer(self, f_Hz)` — From OPEN-CIRCUIT potential to MEASURED potential, complex, dimensionless.
    - `def min_magnitude(self)` — The low-frequency floor of |H| on a 0.1 mHz .. 100 kHz grid.
    - `def magnitude(self, f_Hz)`
    - `def phase_deg(self, f_Hz)`
    - `def noise_spectral_density_V_per_rtHz(self, f_Hz)` — ``sqrt(4 k T Re[Z_e(f)])`` at the CONTACT (electrode thermal noise).
    - `def amplifier_noise_spectral_density_V_per_rtHz(self, f_Hz)` — ``sqrt(4 k T R_in) * |Z_e/(Z_e+R_in)|`` at the AMPLIFIER input.
    - `def with_overrides(self, **kw)` — A copy of this interface with some values replaced (for sweeps).
    - `def noise_rms_uV(self, f_lo_Hz=1e-06, f_hi_Hz=100000.0, *, include_amplifier=False, n_grid=6000)` — Band-limited Johnson-Nyquist noise from this contact, in uV RMS.
    - `def as_dict(self)`
- **`def thermal_noise_variance(interface, *, f_lo_Hz=1e-06, f_hi_Hz=100000.0, n_grid=6000, include_amplifier=False)`**
    Rectangular-band thermal variance at the amplifier input, in V^2.
- **`class CrosstalkMatrix`**
    Per-channel coupling at the FIXED pitch, with TWO separate contributions.
    - `def n_channels(self)`
    - `def pair_distances_um(self)`
    - `def medium_coupling_ohm(self, d_um)` — ``Z_c(d)`` from the solver's own law, with an explicit metal return.
    - `def added_crosstalk_matrix(self, f_Hz=0.0, *, contact=None)` — ``C_add`` at one frequency: fraction of channel k's signal on channel j.
    - `def inherent_crosstalk(self, access_map, *, channel_neurons, max_pairs=None)` — (a) Medium-inherent coupling, MEASURED from ``transfer_mV_per_nA``.
    - `def offdiagonal_structure(self, C, *, top=8)` — Structure of an off-diagonal matrix: nearest-neighbour rings by pitch.
    - `def ring_table(self, f_Hz=0.0, *, contact=None)` — Off-diagonal coupling by pitch ring, at one frequency.
    - `def as_dict(self, *, f_Hz=0.0, C=None)`
- **`class FrontendResult`**
    Everything the front end produced, with the chain stages kept separate.
    - `def as_dict(self)`
- **`def apply_frontend(access_map, *, contact, interface, neuron_current_nA=0.01, conductivity_S_m=0.3, f_Hz=1000.0, frequencies_Hz=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0), quadrature=None, source_mode='soma_position', normal=(0.0, 0.0, 1.0), convergence_check_channels=4, noise_f_lo_Hz=1e-06, noise_f_hi_Hz=100000.0, seed=0, measurement_reference_um=None)`**
    Global unique-root field -> identical reference -> area -> complex H(probe).
- **`def build_frontend_ledger(contact, interface, *, pitch_um, n_channels, sigma_S_m, capture_radius_um=None, solver_r_spread_ohm=None, access_map=None) -> FrontendLedger`**
    Assemble the provenance ledger for a front-end run.

### `engine/touch_damage_scenario.py`  (1937 行)

ONE scenario that puts the data-anchored neck cut and the micrometre-electrode damage model into the same paired experiment.

项目内依赖：`.cable`, `.electrode_damage`, `.neck_cut_data`, `.neural_cond`, `.profile`

- **`def local_mask(tier, global_mask, semantics=DEFAULT_MASK_SEMANTICS)`**
    Tier-LOCAL boolean mask for a global neuron mask.
- **`def mask_membership_table(tier, groups)`**
    Per named mask: membership under the correct and the legacy semantics.
- **`def mask_semantics_contrast(a, tier, cut, groups, scenario=None, cfg=None, tier_cfg=None)`**
    OLD-vs-NEW numbers for one arm differing ONLY in the mask semantics.
- **`def mask_wraparound_audit(a, tier, groups, semantics=None)`**
    MEASURED size of the round-1 wraparound defect, per named mask.
- **`class TouchDamageSpec`**
    One arm of the paired grid.  Only the intervention differs between arms.
    - `def validate(self)`
    - `def has_electrode(self)`
    - `def cut_is_active(self)`
    - `def arm(self)`
    - `def as_dict(self)`
- **`def arm_name(scenario_name, coupling_mode)`**
    Report key of one (scenario, coupling) arm.
- **`def base_scenario_of(arm)`**
- **`def with_arm(spec, *, route=None, sign_mode=None, coupling=None)`**
    A copy of ``spec`` carrying an explicit route, sign mode and/or coupling.
- **`class CableFixture`**
    A DECLARED fibre segment crossed by the DECLARED track.
    - `def as_dict(self)`
- **`def declared_crossing_fixture(shaft_radius_um, *, fibre_diameter_um=0.5, length_um=500.0, compartments=250, crossing_fraction_of_length=0.75, extra_clearance_um=0.0, name='declared_crossing_fixture')`**
    The declared fixture: an unbranched neurite crossed by the track.
- **`def threshold_for_radius(field, radius_um)`**
    Back-solve the ASSUMED field's own law for the threshold that yields a radius.
- **`class ShuntConductanceNetwork(ConductanceNetwork)`**
    ``engine.neural_cond.ConductanceNetwork`` plus a static per-neuron shunt.
    - `def set_shunt(self, node_indices, g_uS, E_rev_mV=None)` — Set (not increment) the static shunt at chosen point neurons.
    - `def step(self, current_nA=0.0, exc_events_uS=0.0, inh_events_uS=0.0, enabled=None)`
- **`def cable_reference()`**
    Fixtures and provenance-bearing parameters, adopted once, guard-first.
- **`def measure_cable_damage(reference, fixture, damage_radius_um, *, leak_density_S_cm2, dt_ms=0.05, duration_ms=50.0, cable_kwargs=None, threshold_label=None, radius_source=None)`**
    Inserted-vs-sham on the declared fixture, with every number kept exact.
- **`def variant_damage_table(reference, fixture, *, threshold_key='optimal', leak_density=None, free_radius_um=None, dt_ms=0.05, duration_ms=50.0, cable_kwargs=None)`**
    The sheath-intact vs sheath-removed damage table on the declared fixture.
- **`def chemical_damage_fraction(radius_um, element_side_um=8.0)`**
    Fraction of a DECLARED local tissue element inside the damage cylinder.
- **`def measure_chemistry(reference, *, damage_fraction_by_variant, dt_s=0.05, duration_s=20.0, fractions_sweep=(0.1, 0.5, 1.0))`**
    The K / barrier chemistry change through the existing injury_tissue API.
- **`def reached_owner_local(tier, groups, route_group='crossing_ascending')`**
    Tier-LOCAL indices of the fibre population the track is DECLARED to meet.
- **`def deterministic_share_subset(local_idx, share)`**
    A deterministic prefix of the sorted owner list, of size ceil(share*N).
- **`def build_coupling(mode, reached_local, *, tier_n, per_compartment_g_uS=None, fibre_total_g_uS=None, fibre_reached_compartments=None, q=None, E_shunt_mV=None)`**
    One exact coupling of the measured local damage into the point tier.
- **`def reached_column_mask(matrix, reach)`**
    CSR column mask: True at stored entries whose presynaptic index is reached.
- **`def scale_columns_in_place(matrix, column_mask, factor)`**
    A new matrix with the SAME sparsity pattern and ``data[column_mask]*factor``.
- **`def couple_edge_set(edges, coupling)`**
    Apply one coupling to an edge set: exact, and a strict no-op when q == 1.
- **`def build_electrode_plan(reference, *, threshold_key='optimal', leak_density=None, dt_ms=0.05, duration_ms=50.0, cable_kwargs=None, fixture=None)`**
    Measure the electrode damage once, on the declared fixture.
- **`def zero_damage_sham_cable(reference, fixture, damage_radius_um, *, dt_ms=0.05, duration_ms=50.0, cable_kwargs=None)`**
    The sham arm: the SAME geometry and code path with a leak density of ZERO.
- **`def build_couplings(reference, fixture, cable_table, tier, groups, *, reach_share, tier_n)`**
    All couplings used by the grid, from the measured cable numbers only.
- **`def run_damage_scenario(a, tier, cut, groups, spec, coupling, cfg=None, tier_cfg=None, keep_spikes=False, mask_semantics=DEFAULT_MASK_SEMANTICS)`**
    Run one arm of the grid with the SAME initialisation and seed as every other.
- **`def run_grid(a, tier, cut, groups, couplings, *, sign_modes=SIGN_MODES, routes=ACTIVATED_ROUTES, cfg=None, tier_cfg=None, mask_semantics=DEFAULT_MASK_SEMANTICS, extra_spike_scenarios=(), extra_spike_sign_mode=PRIMARY_SIGN_MODE, extra_spike_route='both')`**
    Run every arm of the six-scenario grid for every active route.
- **`def exact_attribution(a, tier, cut, groups, spike_matrix, intact_coupling, cfg=None, tier_cfg=None, sign_mode=PRIMARY_SIGN_MODE, mask_semantics=DEFAULT_MASK_SEMANTICS)`**
    Both exact attributions with ACTIVITY HELD FIXED, from one recorded train.
- **`def build_context(root, *, reach_share=DEFAULT_REACH_SHARE, threshold_key='optimal', leak_density=None, sign_modes=SIGN_MODES, routes=ACTIVATED_ROUTES, run_cfg=None, tier_cfg=None, dt_ms=0.05, duration_ms=50.0, cable_kwargs=None, extra_spike_scenarios=())`**
    Load the data and build every fixture, measurement, coupling and run.
- **`def route_drive_table(grid, routes=ACTIVATED_ROUTES)`**
    route -> arm -> the numbers the aggregate-vs-route demonstration needs.
- **`def replay_electrode_conductance(a, tier, cut, groups, spike_matrix, coupling, config=None, tier_cfg=None, sign_mode='counterfactual_ach', mask_semantics=DEFAULT_MASK_SEMANTICS, window='full')`**
    EXACT conductance the reached fibres deliver, with ACTIVITY HELD FIXED.
- **`def brain_side_owner_contributions(a, tier, cut, groups, spike_matrix, config=None, tier_cfg=None, sign_mode=PRIMARY_SIGN_MODE, window='settled', mask_semantics=DEFAULT_MASK_SEMANTICS)`**
    EXACT conductance delivered to brain-side cells, split by PRESYNAPTIC owner.
- **`def retention_vs_reached_owners(contrib, owners_all, q, *, shares=REACH_SHARE_SWEEP, share_note='', reference_total=None)`**
    Route-drive retention as a function of the DECLARED reached share.
- **`def aggregate_hides_route_loss(sign_mode, route_table, scenario_arm, baseline_arm='intact')`**
    Aggregate retention vs the route-resolved pair, with the hiding quantified.
- **`def declared_parameter_sweeps(reference, fixture, *, dt_ms=0.05, duration_ms=50.0, cable_kwargs=None)`**
    Sensitivities of the model's own declarations and assumed parameters.
- **`def surgery_what_if(reference, fixture, *, radii_um=SURGERY_WHAT_IF_RADII_UM, leak_density_S_cm2=None, dt_ms=0.05, duration_ms=50.0, cable_kwargs=None)`**
    WHAT-IF ONLY: retention vs an ASSUMED surgery damage radius.
- **`def audit_reachability(obj, path='$')`**
    Every NUMERIC value under a reachability-named key is a violation.
- **`def audit_surgery_claims(obj, path='$')`**
    No numeric surgery damage may be claimed as measured or illustrative-fitted.
- **`def audit_surgery_term(term=None)`**
    The registered surgery term must stay unquantified.

### `engine/embodied/natural_scene.py`  (1899 行)

A genuinely NATURAL scene layer, built ON TOP of ``engine.embodied.scene``.

项目内依赖：`engine.embodied`

- **`class TextureProvenance`**
    Everything needed to re-fetch and legally reuse one texture asset.
    - `def as_dict(self) -> dict`
- **`def radial_power_spectrum(array, *, n_bins=None, window=True) -> dict`**
    Radially averaged power spectrum of a 2-D field, with a log-log fit.
- **`def periodicity_peak(array, *, k_floor=3, harmonics=2, exclude_axes=False, smooth_bins=9) -> dict`**
    The periodicity detector: how far the strongest peak stands above the trend.
- **`def spectral_audit(array, *, sample_pitch=None, sample_unit='sample', n_bins=None, k_floor=3, band=(0.02, 0.45), exclude_axes=False) -> dict`**
    Objective answer to "is this field natural or periodic?".
- **`def muoco_builtin_checker_field(width=OLD_GROUND_TEX_PX, height=OLD_GROUND_TEX_PX, rgb1=(0.3, 0.3, 0.3), rgb2=(0.4, 0.4, 0.4)) -> dict`**
    The OLD ground texture, read from MuJoCo itself (not re-drawn by hand).
- **`def generate_pink_noise(size, *, beta, seed, periodic=True) -> np.ndarray`**
    Zero-mean unit-variance 2-D pink noise with radial slope ``-beta``.
- **`def generate_fbm_heightfield(n_samples=257, *, extent_mm=120.0, hurst=DEFAULT_HURST, seed=0, amplitude_std_mm=0.16, amplitude_peak_to_peak_mm=None, micro_relief_mm=0.0, micro_cutoff_fraction=0.25, periodic=True, metadata_extra=None) -> tuple[np.ndarray, dict]`**
    Seeded multi-octave fBm terrain.  Returns ``(elevation, metadata)``.
- **`def generate_natural_texture_png(path, size=512, *, seed=0, beta=2.4, target_std=0.12, grey=False, clahe_clip=None) -> dict`**
    Write a natural-statistics PNG and return its metadata.
- **`def commons_imageinfo(titles, *, timeout=30.0) -> dict`**
    Query the Commons API for url + size + licence metadata of each title.
- **`def is_free_license(license_name) -> bool`**
    True only for the licences this module is willing to use.
- **`def download_natural_texture(out_png, *, size=1024, prefer='dry_decay_leaves_preview', candidate_titles=None, timeout=60.0, grey=False, max_source_pixels=4096) -> dict`**
    Route A.  Download a licence-verified Commons photo and write a square PNG.
- **`def resolve_ground_texture(*, out_png=None, size=1024, allow_network=True, seed=0, beta=2.4, grey=False) -> dict`**
    Whichever route is available: real photo first, procedural fallback.
- **`def write_provenance_sidecar(meta, png_path=None) -> str`**
- **`class NaturalSceneConfig(scene.SceneConfig)`**
    A :class:`scene.SceneConfig` plus the fields needed for a natural scene.
- **`def natural_scene_provenance_report(*, texture_meta=None) -> dict`**
    Ledger dump + the asset provenance rules this module enforces.
- **`class NaturalSceneApplicationReport(scene.SceneApplicationReport)`**
    ``scene.SceneApplicationReport`` plus what the natural layer changed.
- **`def raycast_terrain(model, data, *, half_extent_mm, n=9) -> dict`**
    Sample the compiled terrain's height by ray-casting straight down.
- **`def apply_natural_scene(world, fly, cfg, ledger=None, *, measure=True) -> NaturalSceneApplicationReport`**
    Realise ``cfg`` in ``world``'s MJCF spec.  Call BEFORE ``world.compile()``.
- **`def preset_configs() -> dict`**
- **`def apply_natural_scene_to_backend(backend, preset, **kwargs)`**
    Apply a natural preset to an already-built ``BodyBackend``.
- **`def assert_module_contract() -> dict`**
    Cheap self-check of the exports the integration depends on.

### `engine/electrode_damage.py`  (1870 行)

engine.electrode_damage -- mechanical damage from inserting a micrometre-scale electrode into Drosophila nervous tissue.

项目内依赖：`.cable`, `.injury_tissue`, `.params`, `.profile`

- **`def build_registry(name='electrode_damage')`**
    Every number this module can use, with enforced provenance.
- **`class Adopted`**
    One parameter's value plus the provenance a report must carry.
    - `def to_dict(self)`
    - `def label(self)` — Short human label used on figures.
- **`class SpeciesGuard`**
    Refuses, or force-flags, foreign-species numbers in a fly model.
    - `def adopt(self, name)`
    - `def adopted_record(self, adopted)` — Verify a record really is flagged, then return its report dict.
    - `def foreign_parameters_used(self, adopted_list)`
    - `def record(self, adopted_list=())`
- **`def audit_species_labeling(obj, path='$')`**
    Return the list of species-labelling violations in a report object.
- **`def circle_rectangle_overlap_area_um2(radius_um, center_x_um, center_y_um, width_um, height_um)`**
    Exact (quadrature, kink-split) area of disc n rectangle, in um^2.
- **`class FlyBrainGeometry`**
    A rectangular fly-brain box; damage is reported as a fraction of it.
    - `def from_adopt(cls, adopt)`
    - `def volume_um3(self)`
    - `def footprint_um2(self)`
    - `def track_area_um2(self, radius_um)` — Cross-sectional area of a centred cylindrical damage zone.
    - `def track_damage_volume_um3(self, radius_um, depth_um=None)` — Cylinder n brain box (exact lateral clip) + hemispherical tip cap.
    - `def damage_fraction(self, radius_um, depth_um=None)`
    - `def thickness_ratio(self, radius_um)` — 2R / thickness: 2.0 means the zone spans the whole brain thickness.
    - `def footprint_fraction(self, radius_um)`
    - `def to_dict(self)`
- **`class CavityExpansionField`**
    ASSUMED plane-strain incompressible cavity expansion around a shaft.
    - `def wall_displacement_um(self)`
    - `def wall_pressure_Pa(self)`
    - `def displacement_um(self, r_um)`
    - `def hoop_strain(self, r_um)`
    - `def radial_strain(self, r_um)`
    - `def max_abs_principal_strain(self, r_um)`
    - `def max_shear_strain(self, r_um)`
    - `def radial_stress_Pa(self, r_um)`
    - `def hoop_stress_Pa(self, r_um)`
    - `def max_shear_stress_Pa(self, r_um)`
    - `def strain_radius_um(self, strain_threshold)` — r where |eps| = threshold, floored at the shaft radius a.
    - `def shear_stress_radius_um(self, shear_stress_Pa)`
    - `def incompressibility_residual_um2(self, r_um, h_rel=0.001)` — d(r*u)/dr, exactly zero analytically; central difference here.
    - `def equilibrium_residual_Pa_per_um(self, r_um, h_rel=0.001)` — d(sigma_rr)/dr + (sigma_rr - sigma_tt)/r, zero analytically.
    - `def profile_arrays(self, r_um)`
    - `def to_dict(self)`
- **`class AffineDamageRadius`**
    AFFINE damage-radius alternative r = offset + slope*a (pure assumption).
    - `def radius_um(self, shaft_radius_um)`
    - `def to_dict(self)`
- **`def provenance_tier(source)`**
    Which evidence tier a citation string belongs to.
- **`class SheathLayer`**
    The sheath (neural lamella + perineurium) as a DISTINCT load-bearing layer.
    - `def breached(self, shaft_radius_um)` — PRIMARY criterion: the sheet must open wider than it can sustain.
    - `def hole_edge_stretch(self, shaft_radius_um)` — Degenerate second criterion: hoop strain at the hole rim = delta/a.
    - `def stretch_criterion_breached(self, shaft_radius_um, critical_stretch)`
    - `def delamination_radius_um(self, shaft_radius_um)`
    - `def breach_area_um2(self, shaft_radius_um)`
    - `def opening_radius_sweep(self, shaft_radii_um, critical_values)` — Breach state for every (shaft radius, assumed criterion) pair.
    - `def to_dict(self)`
- **`class PreparationVariant`**
    Which preparation the electrode is inserted into.
    - `def to_dict(self)`
- **`def preparation_variants(adopt)`**
    The two preparations requested by the addendum, with provenance attached.
- **`def parenchymal_damage_estimate(field, strain_threshold, geometry, sheath, variant, free_radius_um=None, threshold_record=None)`**
    Damage as a FRACTION of the fly brain, for one preparation variant.
- **`def rodent_radius_records(adopt)`**
    Provenance records for the rodent 'damage radius' numbers people quote.
- **`def rodent_radius_transfer_table(geometry, adopt=None, mouse_thickness_um=900.0, radii_um=None)`**
    Why a rodent damage radius cannot be transferred to the fly.
- **`class SheathRepairModel`**
    Slow sheath repair vs fast membrane resealing: do NOT reuse one for the other.
    - `def sheath_over_pore_ratio(self)`
    - `def sheath_over_transected_ratio(self)`
    - `def to_dict(self)`
- **`def chemistry_config_for_variant(variant, adopt)`**
    Chemistry coupling parameters for a preparation variant.
- **`def injury_threshold_options(guard)`**
    The externally verified thresholds, force-flagged as cross-species.
- **`class ResealingTimescales`**
    Membrane resealing constants, applied exactly as exp(-t/tau).
    - `def from_adopt(cls, adopt)`
    - `def tau_s(self, kind)`
    - `def decay_factor(self, t_since_damage_s, kind='transected')`
    - `def decay_factors(self, times_s, kind='transected')`
    - `def to_dict(self)`
- **`def affected_node_indices(morph, shaft_point_um, shaft_direction, damage_radius_um)`**
    Cable rows whose centre lies within damage_radius of the shaft AXIS.
- **`class ProximityExcitabilityCoupling`**
    OPTIONAL proximity -> reduced-drive coupling.  DEFAULT OFF.
    - `def scale(self, distances_um)`
    - `def to_dict(self)`
- **`class CableDamageConfig`**
    Leak density + reversal for the cable end-leak coupling.
    - `def to_dict(self)`
- **`def apply_membrane_damage(cable, node_indices, leak_density_S_cm2, E_rev_mV)`**
    Add an end leak to the damaged rows via the public set_end_leak API.
- **`def run_paired_cable(morphology, field, shaft_point_um, shaft_direction, strain_threshold, config=None, inject_index=0, target_deflection_mV=10.0, dt_ms=0.05, duration_ms=50.0, cable_kwargs=None, resealing=None)`**
    Inserted vs sham on two identical passive cables.
- **`class ChemistryDamageConfig`**
    - `def scaled_values(self, damage_fraction)`
    - `def to_dict(self)`
- **`def apply_chemistry_coupling(potassium, ligand, barrier_edge, damage_fraction, config=None)`**
    Drive the existing injury_tissue public API from a damage fraction.
- **`def run_paired_chemistry(damage_fraction, config=None, barrier_edge=(1, 2), dt_s=0.05, duration_s=20.0, shape=(4, 1, 1), spacing_um=(5.0, 5.0, 5.0), diffusion_um2_s=10.0, source_amount_s=1.0, potassium_config=None, ligand_extra=None, include_supported=False, supported_reservoir_um3_s=5.0, supported_buffer_amount=4000.0)`**
    Inserted vs sham for the chemistry: identical models, only coupling differs.

### `engine/embodied/scene.py`  (1849 行)

Declarative, parameterised scene layer for the embodied fly body.

- **`class ProvenanceError(ValueError)`**
    A parameter record violates the provenance contract.
- **`class UncitedMeasuredClaimError(ProvenanceError)`**
    A record claims MEASURED_CITED without a usable citation.
- **`class ParameterRecord`**
    One number used by the scene layer, with where it came from.
    - `def as_dict(self) -> dict`
- **`class ParameterLedger`**
    An append-only bag of :class:`ParameterRecord` objects, keyed by name.
    - `def add(self, record) -> ParameterRecord`
    - `def record(self, name, value, unit, provenance, source='', sweep_range=None) -> ParameterRecord`
    - `def record_vector(self, name, values, unit, provenance, source='', sweep_range=None) -> list[ParameterRecord]`
    - `def get(self, name) -> ParameterRecord`
    - `def names(self) -> tuple[str, ...]`
    - `def counts(self) -> dict[str, int]`
    - `def as_dicts(self) -> list[dict]`
- **`class RoughnessConfig`**
    Micro-relief of the ground.
    - `def grid_cell_mm(self) -> float`
- **`class CapillaryConfig`**
    Tarsal secretion / capillary adhesion term.
- **`class LightSpec`**
    One MuJoCo light.  ``kind`` is directional/point/spot; ``mode`` is a
- **`class ObjectSpec`**
    One scene object for the visual pathway.
- **`class SceneConfig`**
    A complete, reproducible description of the world the fly stands in.
    - `def tarsus_friction(self) -> tuple[float, float, float]`
    - `def body_friction(self) -> tuple[float, float, float]`
    - `def ground_friction(self) -> tuple[float, float, float]`
- **`def scene_provenance_report() -> dict`**
    Machine-readable dump of the ledger plus a count per provenance class.
- **`class SceneApplicationReport`**
    What :func:`apply_scene` did, and (when it could compile) what MuJoCo says.
    - `def as_dict(self) -> dict`
    - `def measured_after_compile(cls, model, *, fly_name='nmf', ground_geom_names=(), tarsus_pattern='_tarsus', n_roughness_geoms=None) -> dict` — RE-READ a compiled ``mujoco.MjModel`` and report what is actually there.
- **`def mujoco_geom_type_name(gtype) -> str`**
    Human-readable MuJoCo geom type (``7`` -> ``'mesh'``).
- **`def measured_after_compile(model, **kwargs) -> dict`**
    Functional alias for :meth:`SceneApplicationReport.measured_after_compile`.
- **`def apply_scene(world, fly, cfg, ledger=None, *, measure=True) -> SceneApplicationReport`**
    Mutate ``world``'s MJCF spec (and ``fly``) in place to realise ``cfg``.
- **`def capillary_actuator_names(cfg) -> tuple[str, ...]`**
- **`def drive_capillary_actuators(model, data, cfg) -> dict`**
    Set ``data.ctrl`` for the capillary actuators and report what was written.
- **`def measure_ground_contact_forces(model, data, ground_geom_names) -> dict`**
    Read the ACTUAL contact friction and forces out of the live ``MjData``.
- **`def set_scene_parameter(cfg, path, value) -> SceneConfig`**
    Return a copy of ``cfg`` with a nested field replaced.
- **`def get_scene_parameter(cfg, path) -> Any`**
- **`def sensitivity_sweep(cfg, parameter_name, values, *, evaluator, ledger=None, rollout_seconds=None) -> dict`**
    Sweep one parameter of ``cfg`` and return a MEASURED outcome per value.

### `engine/electrode_tissue.py`  (1588 行)

engine.electrode_tissue -- TISSUE MECHANICS of electrode insertion into Drosophila nervous tissue, and its connection to the REAL access map.

项目内依赖：`.cable`, `.electrode_damage`, `.embodied.access_map`, `.params`, `.profile`

- **`def build_tissue_registry(name='electrode_tissue')`**
    The existing electrode-damage registry PLUS this module's new parameters.
- **`class TissueProvenance`**
    A provenance ledger for tissue mechanics that REFUSES, not merely records.
    - `def adopt(self, name, *, sweep=None, note='', ledger_class=None)` — guard.adopt + ledger.record, with the ledger's refusals preserved.
    - `def record_adopted(self, adopted, *, sweep=None, note='', ledger_class=None, force_cross_flag=True)`
    - `def record(self, key, value, unit, provenance, *, source=None, note='', sweep=None)`
    - `def foreign_parameters_used(self)`
    - `def record_refusal(self, rule, attempted, message)`
    - `def as_dict(self)`
- **`def refuse_foreign_species(registry=None, profile=DROSOPHILA_CNS)`**
    Demonstrate rule R4: build a STRICT guard and return its refusal.
- **`class DimplingModel`**
    Surface indentation (dimpling) of the sheath BEFORE the probe penetrates.
    - `def elastic_stiffness_N_per_m(self)`
    - `def tension_stiffness_N_per_m(self)`
    - `def contact_stiffness_N_per_m(self)` — Sum of the two terms: the model's ASSUMED total stiffness.
    - `def elastic_fraction_of_stiffness(self)`
    - `def depth_um(self, force_uN)`
    - `def force_uN_for_depth(self, depth_um)`
    - `def failure_depth_um(self)` — Pre-penetration dimpling depth: LINEAR in shaft radius (assumed form).
    - `def failure_depth_over_diameter(self)`
    - `def failure_force_uN(self)`
    - `def cap_strain_at(self, depth_um)`
    - `def to_dict(self)`
- **`class InsertionMechanics`**
    Tissue displacement during insertion, at the USER-FIXED shaft diameter.
    - `def from_provenance(cls, prov, *, shaft_diameter_um=None, insertion_depth_um=None, variant_name='sheath_intact')` — Build from a :class:`TissueProvenance`, adopting every parameter.
    - `def shaft_radius_um(self)`
    - `def dimpling_depth_um(self, force_uN=None)`
    - `def pre_penetration_dimpling_um(self)` — The dimpling depth at which the sheet fails: the pre-penetration depth.
    - `def dimpling_regime_for(self, force_uN)`
    - `def cap_strain_at_depth(self, depth_um)`
    - `def displacement_um(self, r_um)`
    - `def hoop_strain(self, r_um)`
    - `def radial_strain(self, r_um)`
    - `def max_abs_principal_strain(self, r_um)`
    - `def max_shear_strain(self, r_um)`
    - `def wall_strain(self)` — eps at r = a: exactly the assumed wall displacement fraction.
    - `def wall_pressure_Pa(self)`
    - `def strain_profile(self, r_um)`
    - `def damage_radius_um(self, strain_threshold)` — REUSED: the strain-threshold crossing of the cavity-expansion field.
    - `def damage_radius_over_shaft_radius(self, strain_threshold)`
    - `def damage_fraction_of_brain(self, strain_threshold)` — REUSED FlyBrainGeometry: damage as a fraction of the fly brain box.
    - `def sheath_breach(self)`
    - `def to_dict(self)`
- **`class MicromotionStrain`**
    Shear strain from PHYSIOLOGICAL MICROMOTION, on tissue already containing
    - `def displacement_um(self, r_um)`
    - `def max_shear_strain(self, r_um)`
    - `def shear_stress_Pa(self, r_um)`
    - `def amplitude_ratio(self)` — A/a: the dimensionless driver of the whole mechanism.
    - `def wall_shear_strain(self)`
    - `def as_cavity_field(self)` — The REUSED field class, when the amplitude is inside ITS domain.
    - `def kernel_matches_cavity_field(self, r_um=())` — Max |difference| between this kernel and the reused class, in-domain.
    - `def small_strain_valid_at(self, r_um)`
    - `def validity_radius_um(self)` — r beyond which gamma <= SMALL_STRAIN_LIMIT (0 if never above it).
    - `def validity(self)`
    - `def damage_radius_um(self, shear_strain_threshold)`
    - `def damage_radius_um_from_stress(self, shear_stress_Pa)`
    - `def cycles_per_day(self)`
    - `def to_dict(self)`
- **`def probe_axis_from_access_map(access_map, *, extension_fraction=0.5, insertion_depth_um=None)`**
    The shaft axis, DERIVED from the array the access map actually built.
- **`def probe_neurons_at_risk(access_map, mechanics, strain_threshold, *, shaft=None, tier_root_ids=None, extension_fraction=0.5, insertion_depth_um=None, yield_verdict=False)`**
    How many of the ACTUALLY-ADDRESSABLE neurons fall inside a strain criterion.
- **`def couple_to_excitable_model(mechanics, at_risk, *, leak_density_S_cm2=0.5, E_rev_mV=None, cable_length_um=None, cable_diameter_um=None, nseg=None, excitability=None, resealing=None, dt_ms=0.05, duration_ms=50.0, target_deflection_mV=10.0, max_neurons=32, cross_check_with_run_paired_cable=True, cable_kwargs=None)`**
    Feed the mechanical damage into the EXISTING passive cable and measure it.

### `engine/embodied/access_map.py`  (1582 行)

Channel <-> neuron ACCESS MAP: where a fixed-geometry electrode array could be placed, in the REAL voxel coordinates of BANC somata, and which neurons each channel could access.

项目内依赖：`..electrode`

- **`class PROVENANCE`**
    The four allowed provenance classes.  There is no fifth.
- **`class ProvenanceRecord`**
    - `def as_dict(self)`
- **`class Ledger`**
    Ordered provenance ledger.  Enforces the honesty rules, not just records.
    - `def record(self, key, value, unit, provenance, *, source=None, note='', sweep=None)`
    - `def get(self, key)`
    - `def as_list(self)`
    - `def as_dict(self)`
    - `def counts(self)`
    - `def unresolved(self)` — Entries a reader must treat as NOT established from a source.
- **`class VoxelResolution`**
    Voxel -> micrometre conversion for the BANC frame.
    - `def nm_per_voxel(self)`
    - `def isotropic(self)`
    - `def voxels_to_um(self, vox)`
    - `def um_to_voxels(self, um)`
    - `def as_dict(self)`
- **`def banc_voxel_resolution() -> VoxelResolution`**
    The VERIFIED BANC native voxel size: **4 x 4 x 45 nm (anisotropic)**.
- **`def measure_resolution_from_parquet(somas_parquet_path) -> VoxelResolution`**
    Read the coordinate-frame resolution out of the parquet's OWN metadata.
- **`class ElectrodeArraySpec`**
    The electrode array.  PITCH and DIAMETER are FIXED user inputs.
    - `def grid_side(self)` — Channels per side for a planar grid (near-square).
    - `def relative_offsets_vox(self, voxel_resolution)` — Channel centre offsets from the array origin, in VOXELS.
    - `def as_dict(self)`
- **`class SomaTable`**
    The measured BANC soma cloud, in VOXEL coordinates.
    - `def duplicate_report(self)` — What the deduplication did, so it is reported rather than implied.
    - `def lookup(self, root_ids)` — Indices into the soma table for ``root_ids``.
    - `def positions_of(self, root_ids)`
- **`def load_soma_table(somas_parquet_path, *, resolution=None) -> SomaTable`**
    Read the BANC soma parquet.  Requires ``pyarrow`` on the path.
- **`class AccessMap`**
    Which neurons each channel can reach, from REAL measured soma positions.
    - `def n_channels(self)`
    - `def counts_by_channel(self)`
    - `def extent_um(self)` — Physical extent of the array (bounding box of the channel centres).
    - `def captured_mask(self)`
    - `def as_dict(self, *, include_channel_lists=True)`
- **`def build_access_map(spec, tier_root_ids, somas_parquet_path, voxel_resolution_nm, *, soma_table=None, target_mask=None, target_names=None, tier_names=None, resolution=None)`**
    Place ``spec`` using the REAL soma positions and map channels to neurons.
- **`def access_map_to_recording(access_map, *, channel_index=0, neuron_current_nA=0.01, conductivity_S_m=0.3, bandwidth_Hz=1000.0, noise_sd_mV=0.0, dt_ms=0.1, n_steps=None, source_mode='channel_site', seed=0)`**
    Turn ONE channel of ``access_map`` into a signal, using the EXISTING front end.
- **`def sweep_access_map(base_spec, base_resolution, tier_root_ids, somas_parquet_path, *, target_mask=None, param='n_channels', values=None, soma_table=None)`**
    Vary ONE declared quantity and report how the answer moves.

### `engine/neck_cut_data.py`  (1508 行)

Data-anchored neck cut for the BANC body-touch -> ascending -> brain route.

项目内依赖：`.neural_cond`

- **`def receptor_class(verified_label, forced_exc=False)`**
    Explicit receptor class, or None for 'excluded and counted'.
- **`class TouchTierConfig`**
    Deterministic selection rule for the bounded touch-route coarse tier.
    - `def validate(self)`
- **`class NetworkRunConfig`**
    Illustrative point-tier physiology for the paired scenario runs.
    - `def validate(self)`
    - `def steps(self)`
    - `def phase_step(self)`
    - `def settle_step(self)`
    - `def delay_steps(self)`
- **`class ScenarioSpec`**
    One paired run: (sign coverage) x (intervention) x (which route is driven).
    - `def validate(self)`
    - `def name(self)`
    - `def as_dict(self)`
- **`def load_banc(path)`**
    Load the cached BANC snapshot, read-only, no pickle.
- **`def mechanosensory_partition(a)`**
    Body-route vs head-route mechanosensory split (the parent's rule, verbatim).
- **`def derive_crossing_set(a)`**
    Re-derive the neck-crossing fibre set from the RAW cached annotation.
- **`def build_cut_set(a, crossing)`**
    Severed fibre set = outgoing rows of the crossing neurons, split by side.
- **`def route_budget(a, cut, mech)`**
    Synapse-weighted body-touch vs head-touch budget, and what the cut removes.
- **`def select_touch_tier(a, mech, cut, config=None)`**
    Deterministic, documented, size-bounded tier for the touch routes.
- **`def tier_index_map_is_consistent(tier)`**
    True iff ``local_of_global`` maps each IN-TIER neuron to a UNIQUE row in [0, n).
- **`def local_mask(tier, global_mask, semantics=DEFAULT_MASK_SEMANTICS)`**
    Tier-LOCAL boolean mask for a GLOBAL neuron mask. This is the ONLY definition.
- **`def named_group_masks(groups)`**
    The 13 NAMED masks the reports quote, with their table-qualified names.
- **`def mask_membership_table(tier, groups)`**
    Per NAMED mask: local membership under the CORRECT and under the LEGACY semantics.
- **`def mask_semantics_audit(tier, groups)`**
    MEASURED size of the round-1 wraparound defect, per named mask.
- **`def mask_semantics_contrast(a, tier, cut, groups, scenario=None, cfg=None, tier_cfg=None)`**
    OLD-vs-NEW numbers for ONE arm that differs ONLY in the mask semantics.
- **`def presynaptic_groups(a, mech, cut)`**
    Two group tables.
- **`def build_edges(a, tier, cut, sign_mode='annotation', cut_mode='intact', config=None)`**
    Signed conductance edge set of the tier for one (sign_mode, cut_mode).
- **`def conductance_ledger(spike_matrix, We, Wi, target_local, group_local, delay_steps, step_slice=None)`**
    Exact conductance (uS) delivered to ``target_local`` per presynaptic group.
- **`def run_scenario(a, tier, cut, groups, scenario, cfg=None, tier_cfg=None, keep_spikes=False, mask_semantics=DEFAULT_MASK_SEMANTICS)`**
    Run one bounded conductance scenario. Returns arrays + JSON-safe metadata.
- **`def replay_severed_conductance(a, tier, cut, groups, spike_matrix, config=None, tier_cfg=None, sign_mode='annotation', mask_semantics=DEFAULT_MASK_SEMANTICS)`**
    EXACT severed conductance under IDENTICAL activity.

### `engine/embodied/adapters.py`  (1503 行)

engine.embodied.adapters -- the neural <-> body adapters for the first closed loop.

项目内依赖：`..neck_cut_data`, `..neural_cond`, `..receptors`

- **`class TierConfig`**
    The bounded BANC subnetwork: reused selection rule, bounded twice.
    - `def validate(self)`
- **`class ReceptorConfig`**
    ILLUSTRATIVE mechanosensory transduction parameters.
    - `def validate(self)`
    - `def provenance(self)`
- **`class DecodeConfig`**
    The ASSUMED map from descending firing rate to the two CPG knobs.
    - `def validate(self)`
- **`class OwnershipError(RuntimeError)`**
    Raised when a tier writes a resource it does not own.
- **`class OwnershipLedger`**
    Audit log of who wrote which resource, and how often.
    - `def write(self, resource, writer)`
    - `def total(self, resource)`
    - `def writers(self, resource)`
    - `def as_dict(self)`
- **`def build_neural_tier(config=None, data_path=None)`**
    Select the bounded tier and build its signed conductance edges.
- **`class NeuralTier`**
    The conductance point tier, advanced only by its owner (the scheduler).
    - `def reset(self)` — Return to the exact rest state, with the same background draw.
    - `def advance(self, n_substeps, sensory_current_nA)` — Advance the tier by ``n_substeps``; return the interval's spike matrix.
    - `def group_rates_hz(self, spikes)` — Per-interval rates (Hz) of the readout groups, in spikes/neuron/second.
- **`class MechanoReceptorBank`**
    Leg contact force + leg joint angle -> MechanoReceptor -> external current.
    - `def set_reference_pose(self, joint_angles_rad)`
    - `def reset(self)`
    - `def stimuli(self, obs)` — Normalised strains per leg (contact, proprioceptive) from one body state.
    - `def transduce(self, obs)` — One interval of transduction.  Returns (current_nA, drive dict).
- **`class Command`**
    One decoded command.  ``source_interval`` makes the causal order auditable.
    - `def clip_report(self, cfg)`
- **`class CommandDecoder`**
    Spike counts -> (speed scale, turn), with a pre-registered reference.
    - `def calibrate(self, tier, calibration_spikes, n_intervals, settled_from_step=0)` — Fix the reference rates from the SETTLED part of a fixed calibration run.
    - `def decode(self, spikes, source_interval, forced_rate=None)`
    - `def rate_hz(self, spikes)`
    - `def bind_readout(self, tier)`
    - `def neutral(self)` — The command used before any interval has completed: exactly neutral.
- **`class CommandedTripodCPG`**
    FlyGym's tripod CPG, with exactly two commanded knobs.
    - `def set_command_freqs(self, freq_scale_per_leg)` — Write ONE intrinsic-frequency scale PER LEG, in LEGS order.
    - `def set_command(self, speed_scale, turn)`
    - `def posture_rows(self, target=None)` — {leg: DOF row} for the posture target joint, resolved by NAME, never by index.
    - `def set_posture_target(self, target)` — Switch which joint the posture command acts on.  Simpler than it sounds: the rows are
    - `def set_posture_bias(self, bias)` — PER-LEG knee offsets.  THIS IS THE THIRD COMMAND, and it has to be per-leg.
    - `def step(self)`
    - `def state(self)`
- **`def model_fingerprint(tier_fingerprint, tier_config, receptor_config, decode_config, loop_config_dict)`**
- **`def dof_names(dof_order)`**
    The model joint names the DOF order refers to, as "<parent>-<child>-<axis>".
- **`def joint_name_index_map(model)`**
    Map a compiled model's joint NAME (without the fly prefix) to its index.
- **`def leg_rows_from_dof_order(dof_order, model=None)`**
    Map the 42 actuated DOFs onto the six legs, and (with a model) onto ANGLES.
- **`def sensor_matrix_row_map(sensory_groups, joint_rows)`**
    The (receptor -> neurons) and (receptor -> DOFs) maps, checked for overlap.

### `engine/embodied/plasticity.py`  (1365 行)

engine.embodied.plasticity -- a SWITCHABLE plasticity layer for the neural tier.

项目内依赖：`.adapters`

- **`def tripod_alternation_reward(drive_per_leg, gain=2.0)`**
    ENGINEERING REWARD PROXY in [-1, 1] from the observed per-leg drives.
- **`class STPConfig`**
    Finite presynaptic release resource (Tsodyks-Markram form), ILLUSTRATIVE.
    - `def validate(self)`
- **`class HomeostasisConfig`**
    Slow activity target -> BOUNDED gain, ILLUSTRATIVE.
    - `def validate(self)`
    - `def gain_of(self, z)` — Bounded gain for a scalar integrator state z (in [-z_limit, +z_limit]).
- **`class LongTermConfig`**
    Bounded three-factor weight change: eligibility x observable feedback.
    - `def validate(self)`
- **`class RewiringConfig`**
    Bounded structural rewiring inside an explicitly declared candidate set.
    - `def validate(self)`
- **`class PlasticityConfig`**
    The four switches.  ALL OFF BY DEFAULT; the frozen arm is exactly this.
    - `def validate(self)`
    - `def enabled_names(self)`
    - `def any_enabled(self)`
    - `def as_dict(self)`
- **`class LesionBarrier`**
    Edges removed for a DECLARED reason, which no plasticity rule may restore.
    - `def add(self, pairs, reason, time_ms=0.0, weight_uS=0.0, channel='unknown')`
    - `def forbids(self, pre, post)`
    - `def reason_of(self, pre, post)`
    - `def log_refusal(self, time_ms, rule, pre, post, parent_state=None)`
    - `def apply_to(self, We, Wi)` — Physically remove every barred pair from a (post, pre) CSR pair.
    - `def as_dict(self)`
- **`class PlasticNeuralTier(NeuralTier)`**
    ``NeuralTier`` + four switchable plasticity mechanisms (all off by default).
    - `def reset(self)`
    - `def set_feedback(self, value)` — Supply an external task-feedback proxy; overrides the built-in one.
    - `def n_edges(self)`
    - `def advance(self, n_substeps, sensory_current_nA)` — Advance the tier; return the interval's spike matrix (spikes at END).
    - `def plasticity_report(self)`
- **`def attach_plasticity(scheduler, plasticity=None, lesion=None)`**
    Swap a built scheduler's tier for a :class:`PlasticNeuralTier` (same tier).
- **`def plasticity_honesty()`**
    The honesty block for this round, restated on the figure and in the report.

### `engine/electrode_stim.py`  (1248 行)

STIMULATION extension of the electrode model: current injection -> extracellular field -> neural activation, under charge-injection safety limits.

项目内依赖：`.cable`, `.electrode`, `.embodied.access_map`

- **`class StimWaveform`**
    A current-injection waveform on the FIXED contact geometry.
    - `def contact_area_cm2(self)` — Sphere surface area of the contact, in cm^2 (FIXED 7 um geometry).
    - `def phase_sign(self)`
    - `def pulse_width_us(self)`
    - `def period_us(self)`
    - `def duty_cycle(self)`
    - `def charge_balanced(self)` — True only for shapes whose two phases are exact mirrors.
    - `def sample_outward_current_nA(self, t_us)` — Outward current at the ACTIVE contact, nA, at time ``t_us``.
    - `def samples(self, dt_us=1.0, n_periods=1)` — (t_us, outward current nA) sampled over ``n_periods`` full periods.
    - `def charge_per_phase_uC(self)` — Charge delivered in ONE phase, uC, by exact integration of the shape.
    - `def charge_per_phase_nC(self)`
    - `def charge_density_per_phase_uC_cm2(self)` — Charge per phase divided by the FIXED 7 um contact's sphere area.
    - `def net_charge_per_pulse_uC(self)` — Net charge after a whole pulse: 0 for a balanced shape, 2x phase
    - `def as_dict(self)`
    - `def at_charge_density(cls, density_uC_cm2, **kw)` — The amplitude whose charge density equals ``density_uC_cm2``.
- **`def straight_neurite(centre_um, direction, length_um, n_points=201)`**
    Equally spaced points along a straight, STATED neurite trajectory.
- **`def chain_morphology(points_um, diameter_um)`**
    A single unbranched chain through ``points_um``, for engine.cable.
- **`class ActivatingFunction`**
    Second spatial derivative of the extracellular potential on a neurite.
    - `def potential_mV(self, xyz_um)` — Extracellular potential at ``xyz_um``, referenced to the declared
    - `def along(self, xyz_um)` — Activating function along an equally spaced trajectory.
    - `def at_point(self, xyz_um, i=None)` — Activating function at one index of an equally spaced trajectory.
    - `def analytic_monopole_mV_per_um2(current_nA, conductivity_S_m, x_um, d_um)` — Ve'' for an ideal 1/r monopole at perpendicular distance ``d``.
- **`def orientation_sweep(contacts, currents_nA, electrode_um, distance_um, angles_deg, length_um=400.0, n_points=201, conductivity_S_m=DEFAULT_CONDUCTIVITY_S_M, reference_um=DEFAULT_REFERENCE_UM, perpendicular=None)`**
    Measure the activating function at the membrane point nearest the
- **`def passive_membrane_response(points_um, diameter_um, currents_nA, contacts, conductivity_S_m=DEFAULT_CONDUCTIVITY_S_M, reference_um=DEFAULT_REFERENCE_UM, dt_ms=0.05, duration_ms=60.0, node=None)`**
    MEASURED membrane deviation of the project's PASSIVE cable in a field.
- **`def shank_channel_centres(origin_um, axis=(0.0, 1.0, 0.0), n_channels=16, pitch_um=CHANNEL_PITCH_UM)`**
    Contact CENTRES of a linear shank: pitch 20 um, centred on ``origin_um``.
- **`def shank_contacts(centres_um, radius_um=CONTACT_RADIUS_UM)`**
    ``Contact`` objects for the same fixed geometry the solver integrates.
- **`def densest_soma_anchor(positions_um, axis, radius_um=25.0)`**
    A REAL data anchor for placing the shank.
- **`class ChargeInjectionLimits`**
    Safety limits for the FIXED 7 um contact, with their basis attached.
    - `def charge_limit_per_phase_uC(self)`
    - `def charge_limit_per_phase_nC(self)`
    - `def amplitude_limit_nA(self, phase_width_us, density_uC_cm2=None)` — Amplitude at which the chosen density ceiling is reached.
    - `def current_limit_nA(self, phase_width_us)` — The smaller of the derived charge-derived current and the ASSUMED
    - `def margin(self, waveform)` — Margin of a waveform against BOTH ceilings.
    - `def as_dict(self)`
- **`def charge_injection_limits(contact_radius_um=CONTACT_RADIUS_UM, *, ledger=None, record=True)`**
    The safety/limit model at the FIXED contact geometry.
- **`def record_limits(ledger, limits, src=None)`**
    Write the limit model into a :class:`StimLedger`, provenance enforced.
- **`class StimLedger(Ledger)`**
    The project's ENFORCING provenance ledger, specialised for stimulation.
    - `def record_limit(self, key, value, unit, provenance, **kw)`
    - `def selftest_refusals(self)` — Attempt both forbidden records; return what happened.
- **`def stimulate_and_record(stim_contacts, stim_currents_nA, record_sites_um, reference_um=DEFAULT_REFERENCE_UM, conductivity_S_m=DEFAULT_CONDUCTIVITY_S_M, bandwidth_Hz=1000.0, noise_sd_mV=0.0, dt_ms=0.1, n_steps=1, membrane_contacts=(), membrane_currents_nA=(), seed=0)`**
    Inject through the stimulating contacts; record on OTHER sites.
- **`def nearest_soma_distances(centres_um, soma_positions_um)`**
    Distance from each contact to the nearest REAL soma point.
- **`def attenuation_law_check(distances_um, potentials_mV, current_nA, conductivity_S_m=DEFAULT_CONDUCTIVITY_S_M)`**
    Compare a measured potential profile to the ideal 1/r monopole.

### `engine/da_protocol.py`  (1146 行)

Illustrative dopamine release/clearance model driven by a MEASURED protocol.

项目内依赖：`.params`

- **`def amount_pmol_to_concentration_nM(amount_pmol, volume_um3)`**
    Explicit pmol -> nM conversion in a stated well-mixed volume [um^3].
- **`def concentration_nM_to_amount_pmol(concentration_nM, volume_um3)`**
    Inverse of :func:`amount_pmol_to_concentration_nM` (amount in pmol).
- **`class ProtocolSpec`**
    MEASURED timing structure of the source experiment (PMC9897283).
    - `def bolus_times_s(self)` — Absolute bolus times [s] counted from the first record's start.
    - `def record_starts_s(self)`
    - `def record_ends_s(self)`
    - `def session_end_s(self)`
    - `def representative_ends_s(self)`
    - `def n_record_samples(self, sample_dt_s)`
    - `def timing_report(self)` — Machine-checkable statement of the measured timeline.
- **`class DAKinetics`**
    ILLUSTRATIVE kinetics and geometry. None of these is a measurement.
    - `def clearance_k_s(self)` — First-order removal rate [1/s].
    - `def clearance_half_life_s(self)` — ln2/k [s], the clearance-only half-decay.
    - `def refill_recovery_per_interval(self, protocol)` — Fraction of a pool deficit erased during one inter-stimulus gap.
    - `def refill_tau_over_isi(self, protocol)`
    - `def conversion_report(self, protocol=None)` — Explicit pmol -> concentration arithmetic for the assumed volume.
- **`def curve_features(c_at_bolus_nM, q_nM, release_tau_s, k_s, pre_s=0.0, c_start_nM=None)`**
    Analytic (peak_nM, peak_time_s, half_time_from_bolus_s, t_half_s).
- **`def measure_trace_features(time_s, concentration_nM)`**
    Measure (peak, peak time, half time, t1/2) from a SAMPLED trace.
- **`def release_sequence(protocol=None, kinetics=None)`**
    Per-stimulus release amounts [pmol] and pool contents; no sampling.
- **`class DAProtocolResult`**
    - `def release_ratio_last_first(self)`
    - `def relative_release_decline(self)`
    - `def declines_monotonically(self)`
    - `def t_half_relative_spread(self)`
    - `def t_half_relative_spread_analytic(self)`
    - `def summary(self)`
- **`def simulate(protocol=None, kinetics=None, sample_dt_s=0.02, coarse_dt_s=0.5, seed=None)`**
    Run the measured protocol; return per-stimulus scalars and traces.
- **`def pattern_verdict(result, decline_threshold=DEFAULT_DECLINE_THRESHOLD, t_half_tolerance=DEFAULT_T_HALF_TOLERANCE)`**
    Does this parameter point show 'release declines, t1/2 approximately fixed'?
- **`def sweep_pool_refill_efficacy(protocol=None, kinetics=None, pool_sizes_pmol=None, refill_taus_s=None, efficacies=None, release_shape_coupling=None, decline_threshold=DEFAULT_DECLINE_THRESHOLD, t_half_tolerance=DEFAULT_T_HALF_TOLERANCE)`**
    Grid over (pool size, refilling time constant, release efficacy).
- **`def sweep_clearance(protocol=None, kinetics=None, clearance_taus_s=None, pool_sizes_pmol=None, efficacies=None)`**
    Sweep clearance only, for the fixed pool: shows release unchanged, t1/2 moved.
- **`def sweep_shape_coupling(protocol=None, kinetics=None, couplings=None)`**
    Where the invariance BREAKS: release waveform shape tracking pool content.
- **`def build_registry(protocol=None, kinetics=None)`**
    Provenance registry: measured protocol vs illustrative kinetics.

### `engine/multirate_scenario.py`  (1126 行)

Bounded multirate scenario: BANC coarse tier + histamine vision + injury + HH neck.

项目内依赖：`.electrode`, `.graded_vision`, `.injury_tissue`, `.neural_active`, `.neural_hist`

- **`def receptor_class(verified_label)`**
    Map a pure `ann_nt_verified` label onto an explicit receptor class.
- **`class SubnetworkConfig`**
    Deterministic selection rule for the bounded BANC coarse tier.
    - `def validate(self)`
- **`class MultirateConfig`**
    - `def validate(self)`
- **`def select_banc_subnetwork(config=None, root=None)`**
    Deterministic, documented, bounded, CONNECTED BANC subnetwork.
- **`def photo_seed_info(n, visual_global, pure_hist, photo_chosen)`**
    Descriptive block about the photosensitive-seed step of the selection.
- **`def barrier_edges(conductance_um3_s, shape=CHEM_SHAPE)`**
    Leaky x/y face barrier map for the injury ligand barrier rewrite.
- **`class SlowInjuryLayer`**
    Owns the slow clocks. Advances ONLY at completed intervals.
    - `def neural_inputs(self, k_gain_uS, ligand_gain_uS, hist_shift_gain)` — State handed to one neural step, built ONLY from completed intervals.
    - `def apply_injury(self, step_index)` — RATE-ONLY injury event: rewrite permeability/reservoir and ligand barrier.
    - `def advance(self, steps_done)` — Advance whichever slow clock completed an interval at this neural-step end.
- **`def neck_segment_world_um(cable)`**
    Local segment centers -> world um through the explicit rigid transform.
- **`def run_multirate_scenario(config=None, subnetwork=None, subnetwork_config=None, root=None)`**
    Run one scenario. Returns arrays + JSON-compatible metadata; writes nothing.
- **`def scenario_set(base=None, scenarios=SCENARIOS)`**
    The intervention set: identical fields, only `scenario` differs.
- **`def compare_scenarios(configs, subnetwork=None, subnetwork_config=None, root=None)`**
    Run several scenarios on ONE shared subnetwork/initialization and collect.

### `engine/synapse_map.py`  (1097 行)

Real FlyWire presynaptic site coordinates mapped onto the validated FAFB subtree.

项目内依赖：`.cable`

- **`def api_token(root=None, filename='.flywire_api_token')`**
    Read the Codex api_token from disk.  Never log or return it in output.
- **`def probe_header(product=SYNAPSE_PRODUCT, dataset=DEFAULT_FAFB_DATASET, root=None, sample_bytes=4 << 20, timeout=180)`**
    Probe a data product and return its CSV header + declared size.
- **`def stream_filter(product=SYNAPSE_PRODUCT, dataset=DEFAULT_FAFB_DATASET, pre_root_ids=(), post_root_ids=(), any_root_ids=(), root=None, timeout=600, progress_every_s=20.0, log=None, max_rows=None, dest_npz=None, label=None)`**
    Stream the synapse table, keep only rows matching the bounded id set.
- **`def load_sites_npz(path)`**
    Load a `stream_filter` npz back into a Sites object (plain-numeric npz).
- **`class Sites`**
    A bounded set of real presynaptic sites, in FAFB nanometres.
    - `def from_fields(cls, fields, source='npz', column_prefix='pre')`
    - `def synthetic(cls, xyz_nm, **kw)`
    - `def select_root(self, root_id, column='pre_root_id')` — Exact-integer filter: ONLY rows whose chosen column equals root_id.
    - `def as_fields(self)`
- **`def assign_nearest_node(sites_nm, morph, *, far_um=2.0, chunk=20000, node_ids_nm=None, node_ids=None)`**
    Nearest skeleton node for every site; distances measured, not assumed.
- **`def map_sites_to_nodes(sites_nm, morph, node_ids_nm, node_swc_ids, *, far_um=2.0)`**
    Convenience wrapper returning the assignment plus per-row match flags.
- **`def truncation_accounting(assignment, excluded_swc_ids, node_swc_ids)`**
    Split the mapped sites into retained-subtree vs excluded-node matches.
- **`def truncation_boundary_accounting(assignment, boundary_retained_swc_ids, excluded_nodes_nm, sites_nm)`**
    How close the real sites come to the artificial truncation boundary.
- **`def distribute_sites(n_sites, weights)`**
    Distribute an integer site count across compartments by largest remainder.
- **`def distribution(site_nodes, n_compartments=None, weights=None, n_sites=None)`**
    Count real sites per compartment row.
- **`def compartment_current(per_compartment_conductance_uS, v_mV, E_rev_mV, extra_scale=1.0)`**
    Inward current vector (nA) for `CableNeuron.step(i_syn=...)`.
- **`def event_conductance(g_per_site_uS, events_per_compartment, g_decay=None)`**
    Conductance per compartment (uS) from integer event counts.
- **`def explicit_stability_limit(neuron, per_compartment_sites, safety=1.0)`**
    Largest `g_per_site` for which explicit current sampling stays stable.
- **`def solve_implicit_synaptic(neuron, g_uS, i_inject=None)`**
    One backward-Euler step with the synaptic conductance INSIDE the solve.
- **`def placement_matrix(site_rows, n_compartments, *, mode='mapped', single_row=None)`**
    Map each site to its compartment: (n_compartments, n_sites) one-hot matrix.
- **`def discrete_release_schedule(site_weights, *, dt_ms, duration_ms, rate_hz, t_start_ms, t_end_ms, seed)`**
    Release counts per SITE per step, with an exactly shared step total.
- **`def project_schedule(site_schedule, placement)`**
    Project per-site events onto compartments: (n_steps, n_comp) counts.
- **`def schedule_totals(schedule)`**
    Total events per step of a schedule (compare across arms for equality).
- **`def run_drive(morph, *, site_nodes=None, site_weights=None, g_per_site_uS=0.0005, E_rev_mV=0.0, tau_ms=2.0, rate_hz=80.0, dt_ms=0.025, duration_ms=20.0, t_start_ms=5.0, t_end_ms=15.0, seed=12345, record_rows=None, passive=None, implicit=False, clamp_to_stability=False, stability_safety=0.5, schedule=None)`**
    Drive a CableNeuron at real-synapse compartments; return traces.
- **`def pairwise_response_compare(morph, *, site_nodes, total_sites, single_row, record_rows, **kw)`**
    (a) one arbitrary injection point vs (b) real distributed sites.
- **`def sha256_file(path, chunk=1 << 20)`**
- **`def read_swc_rows(path)`**
    Raw seven-column SWC rows (no validation; callers assert what they need).
- **`def json_ready(obj)`**
    Recursively convert numpy scalars/arrays to JSON-safe values.

### `engine/embodied/body_backend.py`  (981 行)

Body backend: a physical adult-fly body in a scene, wrapped behind one API.

项目内依赖：`engine.embodied`

- **`class BodyConfig`**
    Everything needed to rebuild the identical body and scene.
    - `def validate(self)`
    - `def units(self)`
- **`class BodyObservation`**
    One body state.  Units are in the field names; nothing is converted.
    - `def contact_present(self) -> np.ndarray` — Boolean (6,): a leg is in contact iff its raw found channel is > 0.
- **`class BodyBackend`**
    A pinned FlyGym body+scene behind a minimal, unit-explicit API.
    - `def attach_cpg_baseline(self)` — Attach FlyGym's OWN tripod CPG.  ENGINEERING BASELINE, not the brain.
    - `def attach_action_source(self, callable_source)` — Attach any object with ``step() -> LocomotionAction``.
    - `def step(self)` — Advance one physics step; the attached action source is applied first.
    - `def whole_body_contact(self) -> dict` — Every contact on the fly, summed INTO THE FLY'S OWN FRAME.
    - `def observe(self) -> BodyObservation`
    - `def raw_vision(self) -> np.ndarray` — Fisheye-corrected RGB from the two compound eyes: (2, H, W, 3).
    - `def ommatidia_readouts(self) -> np.ndarray` — Per-ommatidium yellow/pale channel readings: (2, n_ommatidia, 2).
    - `def describe(self)`
    - `def close(self)`
- **`def summarise_walk(times_s, thorax_mm, contact_present, cfg, contact_found_raw=None)`**
    Reduce a recorded walk to pre-registered engineering metrics.
- **`def select_gl_backend(candidates=GL_BACKENDS)`**
    Return the first GL backend that can actually render here.

### `engine/embodied/vision.py`  (914 行)

engine.embodied.vision -- the visual pathway that the embodied loop did not have.

项目内依赖：`..receptors`

- **`def ideal_observer_quantisation_floor(n_pixels_per_ommatidium=230, n_pixels_perturbed=16, n_levels=255) -> float`**
    Recompute the quantisation allowance used to derive the threshold.
- **`class VisionUnavailable(RuntimeError)`**
    Raised when the compiled model carries no eye cameras (add_vision was not called).
- **`class UnitConversionRefused(ValueError)`**
    Raised when a visual drive would be turned into nanoamperes without an explicit factor.
- **`class VisionConfig`**
    Everything needed to reproduce one visual front end exactly.
    - `def validate(self) -> 'VisionConfig'`
    - `def as_dict(self) -> dict`
- **`class VisualFeatures`**
    One visual sample.  All feature arrays are in READOUT UNITS (dimensionless).
    - `def as_dict(self) -> dict`
- **`class OmmatidiaFrontEnd`**
    Render one fly's compound eyes and turn them into features.
    - `def bind(self, sim, fly_name) -> None` — Check the eye cameras exist and record the readout geometry.
    - `def reset(self) -> None` — Clear the temporal filter state, the clock and the scramble permutation.
    - `def read(self, sim, fly_name, time_s) -> VisualFeatures` — Render (or, if blind, do not render) and return one ``VisualFeatures`` sample.
- **`def visual_drive_nA(features, config, to_nA=None) -> dict`**
    Turn one visual sample into a per-channel current for the neural tier.
- **`def non_degeneracy_statistics(series) -> dict`**
    Decide DEGENERATE vs INFORMATIVE for one recorded series, with the raw numbers.

### `engine/embodied/loop.py`  (876 行)

engine.embodied.loop -- ONE multirate scheduler that owns time, and the episode record.

项目内依赖：`.adapters`, `.body_backend`

- **`def truth_only_names()`**
- **`def heading_yaw_rad(quat_wxyz)`**
    Yaw of the thorax's body x-axis from a unit quaternion (w, x, y, z).
- **`def config_hash(cfg)`**
    sha256[:32] of the loop config, so an episode names the exact configuration.
- **`class LoopConfig`**
    Everything that fixes one episode.  Validated, and refused if inconsistent.
    - `def validate(self)`
    - `def neural_substeps(self)`
    - `def body_substeps(self)`
    - `def n_intervals(self)`
    - `def calibration_settle_steps(self)`
    - `def calibration_intervals(self)`
    - `def as_dict(self)`
- **`class Episode`**
    One recorded episode with SEPARATED truth / observed / commands / events.
    - `def add_event(self, kind, time_s, **payload)`
    - `def finalise(self, wall_seconds, peak_ram_mb)`
    - `def arrays(self)`
    - `def save_npz(self, path)`
    - `def summary(self)`
- **`class MultirateScheduler`**
    Owns the body, the neural tier and the clock.  Nothing else steps anything.
    - `def attach_renderer(self, renderer)`
    - `def make_renderer(self, camera_res=(240, 320), output_fps=25)`
    - `def renderer_frames(self)`
    - `def calibrate(self, episode)` — Pre-registered reference rates: the tier driven by the SPAWN sensation.
    - `def run(self, want_frames=False)`
    - `def measure(self, ep)` — Engineering metrics of one episode, in model units (mm, s, rad).
    - `def check_ownership(self, n_intervals=None)` — The ownership invariants, as numbers a test can assert on.
    - `def close(self)`

### `run_arena_reconstruct.py`  (853 行)

MULTI-CAMERA RECONSTRUCTION: markers -> 3D points -> ABSOLUTE angles.

- **`class PinholeCamera`**
    A camera with a declared pose and no distortion, as the renderer provides it.
    - `def depth_mm(self, pts_mm) -> np.ndarray` — Distance in front of the camera, mm.  Negative/NaN behind it.
    - `def expected_area_px(self, pts_mm, radius_mm) -> np.ndarray` — How many pixels a marker of THIS radius should cover at THAT depth, per point.
    - `def project(self, pts_mm) -> np.ndarray` — World mm -> (row, col).  Points behind the camera return NaN.
    - `def as_dict(self) -> dict`
- **`def triangulate(cams, pts_px, n_refine=3) -> np.ndarray | None`**
    Linear DLT followed by Gauss-Newton refinement on reprojection error, in mm.
- **`def reprojection_error(cams, X, pts_px) -> tuple[float, np.ndarray]`**
    RMS pixel error of a 3D point over the views that saw it, and the per-view errors.
- **`def fit_rigid(src, dst, with_scale=False) -> dict`**
    Kabsch fit of a rigid transform mapping ``src`` onto ``dst``.
- **`def rigid_euler_zyx(R) -> tuple[float, float, float]`**
    Intrinsic Z-Y-X (yaw-pitch-roll) angles of a rotation matrix, in degrees.
- **`def frame_from_three_points(a, b, c) -> np.ndarray | None`**
    An orthonormal basis from three points: origin a, x along a->b, z along the normal.
- **`def angle_between(u, v) -> float`**
- **`def blobs_from_png(path, thr=240.0, min_px=3)`**
    (rows, cols) of the saturated blobs in ONE frame, detected from the IMAGE itself.
- **`def gate_px_at_rows(max_px, rows) -> float`**
    Convert a gate expressed in pixels AT THE REFERENCE HEIGHT into this frame's pixels.
- **`def associate_static(cands, cams, declared_mm, max_px=3.0, normalise_gate=True) -> dict`**
    Assign detected blobs to the DECLARED static fiducials, by projection + gating.
- **`def reconstruct_static(cands, cams, declared_mm, max_px=3.0) -> dict`**
    Triangulate every static fiducial from its assignments, then fit the world frame.
- **`def load_projection_offsets(path) -> dict`**
    Read ``camera_offsets.json`` if it exists, else an empty map.
- **`def build_camera(spec, res, pose=None) -> PinholeCamera`**
    One camera with its image size from the frames and its pose from the best source.
- **`def load_cameras_for_episode(geom, ep_dir, pose_path=None)`**
    ``(cameras, pose_source)`` for ONE episode, using that episode's own frame size.
- **`def load_cameras_compiled(pose_path) -> list[PinholeCamera]`**
    Cameras from the pose READ BACK from the compiled model.
- **`def load_cameras(geom) -> list[PinholeCamera]`**
- **`def reconstruct_tip_candidates(cands, cams, max_views=6, gate_px=2.5) -> dict`**
    Find 3D points that are consistent across ALL views, with no ID assignment.
- **`def main() -> int`**

### `limb_tracking.py`  (797 行)

IDENTITY-SAFE LIMB TRACKING: the anomaly filter that makes a joint angle trustworthy.

- **`def episode_blobs(z, cam_name, frame, min_area=None, max_area=None)`**
    The episode's OWN detector output for one camera and frame.
- **`def marker_size_threshold(z, cam_names, min_gap_frac=0.25) -> dict`**
    Split the episode's blobs into the FIDUCIAL class and the FLY-MARKER class by area.
- **`def blobs_from_png(path, thr=250.0, min_px=6, max_px=2000, min_circularity=0.45)`**
    Marker candidates in one frame: saturated, right-sized, round-ish components.
- **`def triangulate_checked(cams, px, min_views, reproj_gate_px)`**
    GATE 2: triangulate from >= min_views and require every view to agree.
- **`def calibrate_effective_radius(cams, z, declared_mm, blob_getter, area_getter, nominal_radius_mm, frames=(0,), search_px=4.0) -> dict`**
    Measure the marker's EFFECTIVE radius, instead of assuming its nominal one.
- **`def screen_points(cams, per_cam_blobs, seeds_mm=None, min_views=3, reproj_gate_px=2.0, gate_px=3.0, per_cam_area=None, seed_radius_mm=None, area_band=None)`**
    Find 3D candidate points that several cameras agree on, with NO identity assignment.
- **`def assign_one_to_one(predicted_mm, points, weights=None)`**
    GATE 3: Hungarian assignment between predicted markers and 3D candidates.
- **`def main() -> int`**

### `run_arena_record.py`  (600 行)

SIX-CAMERA HIGH-SPEED RECORDING in the grass meadow.

项目内依赖：`engine.embodied`

- **`def marker_specs() -> tuple[dict, ...]`**
    The six static fiducials as MJCF geoms for ``BodyConfig.extra_geoms``.
- **`def detect_markers(rgb, thresh_sigma=None, min_px=1, max_px=60, abs_level=250.0) -> tuple[np.ndarray, np.ndarray]`**
    Marker detector: intensity-weighted connected-component centroids above a level.
- **`def run_episode(seed, seconds, fps, cam_res, rig, gl, with_render=True, marker_radius_mm=FLY_MARKER_RADIUS_MM, max_px=None) -> dict`**
- **`def main() -> int`**

### `engine/embodied/electrode_in_loop.py`  (538 行)

ELECTRODE IN THE CLOSED LOOP.  The seam that was missing between the brain and the body.

项目内依赖：`engine.electrode`, `engine.electrode_stim`

- **`class SomaPositions`**
    Real BANC soma voxel coordinates for a set of neurons, in micrometres.
    - `def from_banc(cls, root_ids, *, soma_parquet=None, resolution_um=None)` — Look up soma positions for `root_ids`, reporting how many were NOT found.
- **`class InLoopElectrode`**
    Records a channel array from the SAME spike matrix that the loop already computes.
    - `def geometry(self) -> dict`
    - `def local_index_searchsorted(self, fired) -> np.ndarray` — Tier neuron index -> row in `soma_um`, or -1 when that neuron has no soma.
    - `def record(self, spikes, *, t0_ms=0.0) -> np.ndarray` — (n_channels, n_substeps) recorded voltage in mV for one interval's spikes.
    - `def dependence(self, spikes) -> dict` — THE GUARD.  Does the recording actually depend on which neurons fired?
- **`class InLoopStimulus`**
    Injects stimulation current into the SAME tier the electrode records from.
    - `def safety(self) -> dict`
    - `def current_nA(self, t_ms, n_neurons) -> np.ndarray` — Injected current per tier neuron at time `t_ms`, nanoamperes.
    - `def capture_radius_sweep(cls, electrode, *, channel_index=0, radii_um=None) -> dict` — How many neurons each capture radius recruits, so the DECLARED 50 um can be judged.
    - `def mechanism(self, *, dt_ms=None, n_substeps=6) -> dict` — MEASURE the sampled waveform, so "the recovery phase is not sampled" is checkable.
    - `def dependence(self, make_tier, *, n_substeps=40, dt_ms=0.5, drive_nA=None) -> dict` — THE GUARD.  Does stimulating actually change what the tier does?

### `engine/embodied/seams.py`  (506 行)

THE FOUR SEAMS OF THE CLOSED LOOP, AS EXPLICIT INTERFACES.

- **`class SensoryDrive(Protocol)`**
    BODY -> BRAIN.  Turns an observation of the body into current injected into the tier.
    - `def transduce(self, obs) -> tuple[np.ndarray, dict]`
    - `def set_reference_pose(self, joint_angles_rad) -> None`
    - `def reset(self) -> None`
- **`class MotorCommand(Protocol)`**
    BRAIN -> BODY.  Whatever turns neural state into an actuation command.
    - `def step(self) -> Any`
- **`class NeuralRecorder(Protocol)`**
    BRAIN -> ELECTRODE.  Turns a spike matrix into a recorded trace.
    - `def record(self, spikes, *, t0_ms=0.0) -> np.ndarray`
    - `def dependence(self, spikes) -> dict`
- **`class NeuralStimulus(Protocol)`**
    ELECTRODE -> BRAIN.  Stimulation current injected into the tier.
    - `def current_nA(self, t_ms, n_neurons) -> np.ndarray`
- **`def survey() -> dict`**
    Which seams are implemented, and do the implementations actually satisfy the protocol?

### `engine/neural.py`  (434 行)

engine.neural -- spiking neural layer for the fly nervous system.

- **`def nt_sign(nt_type)`**
    +1 for excitatory, -1 for inhibitory, 0 if unknown.
- **`class NetworkParams`**
    Neuron parameters, each with a provenance tag.
    - `def provenance(self, k)`
    - `def to_dict(self)`
    - `def provenance_dict(self)`
- **`def lemplev_ziv(binary_string, max_len=None)`**
    Lempel-Ziv complexity (LZ76) of a binary sequence, normalised by length.
- **`def binarize_population(spikes, bin_steps=10, threshold=None)`**
    Binarise a spike raster into a 1-D population-activity string.
- **`def lemplev_ziv_info(binary_string, max_len=None)`**
- **`class LIFNetwork`**
    A vectorised LIF network with signed synapses and a fixed delay.
    - `def set_connectivity(self, W)` — Set the signed sparse weight matrix, shape (n_post, n_pre).
    - `def set_connectivity_from_edges(self, pre, post, syn_count, nt_of_presynaptic)` — Build signed weights from a FlyWire-style edge list.
    - `def step(self, i_ext=None, gain=None)` — Advance one dt.  Returns the boolean spike vector.
    - `def run(self, duration_ms, i_ext=None, gain=None, record=False, lz_mode='population', lz_bin_steps=10, lz_max_len=200000)` — Run and return a summary dict; optionally the full spike raster.
    - `def f_i_curve(self, currents, duration_ms=2000.0)` — Steady-state rate vs constant injected current (for the analytic test).
- **`def analytic_lif_rate(I, tau_m_ms, v_rest, v_th, v_reset, t_ref_ms)`**
    Closed-form steady-state rate of a noise-free LIF under constant drive.
- **`def build_synthetic_network(n, n_edges, exc_frac=0.6, seed=0, w_scale=0.5, syn_per_edge=6, params=None, nt_unknown=0.0)`**
    A FlyWire-SHAPED synthetic network, for testing the machinery before the

### `engine/cable.py`  (351 行)

Passive multicompartment cable approximation (not an active neuron backend).

- **`class Morphology`**
    One connected rooted tree, in micrometers; row order is never changed.
    - `def lengths_um(self)`
    - `def children(self)`
    - `def cylinder(cls, length_um, diameter_um, nseg=50)`
- **`def swc_to_morphology(path, keep=None, *, units=None)`**
    Read strict seven-column SWC; return (morphology, IDs in file row order).
- **`class CableNeuron`**
    Passive implicit cable: um, ms, mV, nF, uS, nA (positive inward).
    - `def cut_edges(self, child_indices)` — Disconnect parent axial edges by CHILD ROW INDEX, not SWC ID.
    - `def set_end_leak(self, node_indices, g_uS, E_rev_mV)` — Set (not increment) distinct end conductance/reversal at chosen rows.
    - `def step(self, i_syn=None, i_inject=None)` — Advance dt; currents are nA, POSITIVE INWARD, scalar or (n,) array.
    - `def run(self, duration_ms, i_inject=None)` — Return step-end voltages; duration rounds to nearest whole timestep.
    - `def analytic(self)` — Uniform-cylinder theory only; not a tapered/branched exact solution.
- **`class Synapse`**
    Exponential conductance, uS; reversal mV; currents nA.
    - `def step(self, spikes_or_events, dt_ms)`
    - `def outward_current(self, v_mV)`
    - `def current(self, v_mV)` — Backward-compatible alias of outward_current (positive OUTWARD).
    - `def inward_current(self, v_mV)`

### `engine/receptors.py`  (329 行)

engine.receptors -- sensory transduction: environment -> receptor current.

- **`class ConversionRefused(ValueError)`**
    Raised when an mV-equivalent receptor drive would be used as nanoamperes.
- **`def to_nA(drive_mV, nA_per_mV, provenance=None)`**
    Convert a receptor DRIVE (mV-equivalent) into nanoamperes, explicitly.
- **`class PhotoReceptor(_Receptor)`**
    Naka-Rushton with slow light adaptation.
    - `def instantaneous(self, I)`
    - `def steady_state(self, I)`
    - `def step(self, stimulus, dt_ms=None)`
- **`class MechanoReceptor(_Receptor)`**
    Two-state mechanotransduction channel.
    - `def steady_state(self, strain)`
    - `def step(self, stimulus, dt_ms=None)`
- **`class ChemoReceptor(_Receptor)`**
    Langmuir binding of an odorant, with a first-order approach to it.
    - `def steady_state(self, conc)`
    - `def step(self, stimulus, dt_ms=None)`
- **`class ThermoReceptor(_Receptor)`**
    Saturating sigmoid around a preferred temperature (normalised).
    - `def steady_state(self, temp_C)`
    - `def step(self, stimulus, dt_ms=None)`
- **`class HygroReceptor(ThermoReceptor)`**
    Humidity transduction; `t_pref` is the half-point humidity instead.
- **`class ReceptorBank`**
    Attach receptor populations to real sensory-neuron groups.
    - `def attach(self, modality, indices, model_cls, gain_mV=10.0, **model_kw)`
    - `def current(self, stimuli, dt_ms=None)` — `stimuli` maps modality -> stimulus (scalar, or per-receptor array).
    - `def current_nA(self, stimuli, nA_per_mV, provenance=None, dt_ms=None)` — Like ``current`` but returns NANOAMPERES for the conductance tiers.
    - `def describe(self)`

### `engine/layers.py`  (308 行)

engine.layers -- the layered, strictly feed-forward engine core.

项目内依赖：`.units`

- **`class FeedforwardViolation(RuntimeError)`**
- **`class WiringError(RuntimeError)`**
- **`class Port`**
- **`class Layer`**
    Base class.  Subclasses set name/requires/provides and implement step().
    - `def initialize(self, inputs)` — Called once before the run with the initial inputs.
    - `def step(self, dt, inputs)` — Advance this layer by `dt` (a Quantity).  Return {port: value}.
    - `def required_ports(self)`
    - `def provided_ports(self)`
    - `def describe(self)`
- **`class Pipeline`**
    An ordered, unit-checked, strictly feed-forward stack of layers.
    - `def add(self, layer)`
    - `def remove(self, name)`
    - `def by_name(self, name)`
    - `def wire(self)` — Validate the stack: feed-forward only, units compatible.
    - `def ports(self)`
    - `def describe(self)`
    - `def run(self, duration, dt_outer=None, state=None, on_step=None)` — Run the stack.
    - `def wiring_report(self)`
- **`class ConstantSource(Layer)`**
    Publishes a constant field.  The only layer allowed to be a root.
    - `def step(self, dt, inputs)`
- **`class Integrator(Layer)`**
    dX/dt = rate * input.  Used in tests to check wiring and dt handling.
    - `def initialize(self, inputs)`
    - `def step(self, dt, inputs)`
- **`class RatioMonitor(Layer)`**
    A monitoring layer: publishes a derived observable without feeding back.
    - `def step(self, dt, inputs)`

### `run_tactile_record.py`  (300 行)

TACTILE RECORDING: per-leg contact FORCE and contact AREA over a walking episode.

项目内依赖：`engine.embodied`

- **`def measure_tip_geometry(backend) -> dict`**
    The tarsal tip's real size, from the compiled model's geom AABB, in metres.
- **`def run_episode(seed, seconds, rig, dt_obs, gl) -> dict`**
- **`def main() -> int`**

### `engine/units.py`  (289 行)

engine.units -- explicit physical units with dimensional arithmetic.

- **`class UnitError(ValueError)`**
    Raised for unknown units, dimension mismatch, or bad unit syntax.
- **`class Dims`**
    A dimension signature: mapping base dimension -> exponent.
    - `def is_dimensionless(self)`
- **`def unit_of(text)`**
    Parse a unit string -> (scale_to_SI, Dims).  Raises UnitError.
- **`class Quantity`**
    A value with units.  Arithmetic keeps dimension bookkeeping exact.
    - `def dims(self)`
    - `def si(self)` — Value in SI base units.
    - `def is_dimensionless(self)`
    - `def from_si(cls, si_value, unit)`
    - `def to(self, unit)`
    - `def in_unit(self, unit)`
- **`def Q(value, unit='1')`**
    Shorthand constructor.
- **`def convert(value, frm, to)`**
    Convert a bare number between compatible units.  Raises on mismatch.

### `engine/sensorimotor.py`  (281 行)

engine.sensorimotor -- the sensorimotor interface to a virtual environment.

- **`class SensorimotorInterface`**
    Named sensory-input and motor-output ports, built from real annotations.
    - `def all_input_indices(self)`
    - `def all_output_indices(self)`
    - `def make_drive(self, signals, shape_n=None, default=0.0)` — Build a per-neuron external-current array in mV.
    - `def read_activity(self, spikes, dt_ms=1.0)` — Read motor output from a spike vector.  Returns {name: rate in Hz}.
    - `def read_population(self, spikes)` — Also report the intrinsic population rate.
    - `def verify(self)` — Check the groups against the annotation table, exactly.
    - `def describe(self)`
    - `def to_dict(self)`
    - `def save_groups(self, path)`

### `engine/local_tissue.py`  (279 行)

README — illustrative conservative local tissue / neuromodulation prototype.

- **`class FinitePool`**
    Finite well-mixed pool: volume um^3, initial nM, grid conductance um^3/s.
- **`class PrescribedBath`**
    Infinite bath: fixed nM, exchange conductance to each voxel in um^3/s.
- **`class LocalTissue`**
    Finite ligand grid with reversible receptors and a mass-flow ledger.
    - `def occupancy(self)`
    - `def total_amount(self)` — Grid free+bound and finite pools, in nM*um^3 (excludes infinite baths).
    - `def mass_balance_error(self)`
    - `def step(self, dt_s)` — Advance one positive-duration split step; return mass residual in nM*um^3.
- **`class NonPhysiologicalSupportProxy`**
    Opt-in dimensionless availability, NOT oxygen/ATP or physiological energy.
    - `def step(self, dt_s, supply_s=0.0, demand_s=0.0)`

### `engine/neural_active.py`  (276 行)

Bounded, idealized active NEURON cables -- NOT fly-calibrated physiology.

- **`class IdealCableSpec`**
    - `def validate(self)`
    - `def provenance(self)`
- **`class ActiveCable`**
    Construct via ActiveRuntime.add_cable; probe and drive at segment centers.
    - `def voltage_mV(self)` — Transmembrane voltage at every segment center (copy).
    - `def segment_xyz_um(self)` — Ideal straight cable along +x, origin (0,0,0), segment-center sites.
    - `def extracellular_mV(self)` — Solved extracellular layer-0 voltage, distinct from command.
    - `def set_inward_current_nA(self, values)` — Set held current per segment; positive INWARD. Scalar drives EACH segment.
    - `def set_extracellular_voltage_mV(self, values)` — Held prescribed field command e_extracellular per segment.
    - `def add_conductance_uS(self, receptor, increment)` — Immediate START-of-interval ExpSyn state increment, in uS.
    - `def cut_axial(self, child_section)` — Disconnect section's parent junction; new ends are sealed, irreversible.
- **`class ActiveRuntime`**
    Exclusive fixed-dt global NEURON owner, <=8 cables / <=1000 total segments.
    - `def add_cable(self, gid, spec=None)`
    - `def t_ms(self)`
    - `def reset(self, voltage_mV=-65.0)`
    - `def step(self)`
    - `def close(self)`

### `engine/neural_hist.py`  (263 行)

Dedicated histamine conductance channel on top of ConductanceNetwork.

项目内依赖：`.neural_cond`

- **`class HistamineParams(ConductanceParams)`**
    ConductanceParams extended with one dedicated histamine channel.
    - `def validate(self)`
    - `def provenance(self)`
- **`class HistamineNetwork(ConductanceNetwork)`**
    ConductanceNetwork + independent `g_hist` state (parent source untouched).
    - `def set_histamine_connectivity(self, hist_uS)` — Set the recurrent histamine matrix; must not overlap We/Wi (no double count).
    - `def step(self, current_nA=0.0, exc_events_uS=0.0, inh_events_uS=0.0, hist_events_uS=0.0, enabled=None)` — One backward-Euler interval with three conductance channels.
    - `def extra_state_hash(self)` — Stable digest of the histamine-specific state (channel + matrices).
- **`class GradedHistamineDriver`**
    Thin adapter making GradedVision's apply_to contract usable here.
    - `def apply_to(self, grad, light, dt_ms=None)` — Run grad.step() and route the increment into the histamine channel.
- **`def histamine_channel_matches(grad, params)`**
    True when a GradedVision instance's E_hist/tau equal a HistamineParams.
- **`def decay_factors(params, dt_ms)`**
    Exact per-step decay factors (diagnostics; not used in the hot path).
- **`def zero_histamine_equivalence_check(n_neurons, params, dt_ms, steps, seed=0, exc_uS=None, inh_uS=None)`**
    Run a parent and a HistamineNetwork side by side with zero histamine.
- **`def parameter_provenance()`**
    Module-level provenance block for reports.

### `engine/project_layers.py`  (227 行)

engine.project_layers -- wrap the project's existing modules as engine layers.

项目内依赖：`.layers`, `.units`

- **`class TrachealGasLayer(Layer)`**
    O2 delivery by the tracheal system, not by blood vessels.
    - `def step(self, dt, inputs)` — Steady radial diffusion from a tracheole of radius r_t out to the
- **`class CellStateLayer(Layer)`**
    cellstate.py -- per-cell volume / ATP / membrane integrity.
    - `def step(self, dt, inputs)`
- **`class GrowthLayer(Layer)`**
    growth.py -- colony growth and division, limited by a substrate.
    - `def initialize(self, inputs)`
    - `def step(self, dt, inputs)`
- **`class HormoneLayer(Layer)`**
    hormone.py -- receptor binding and a delayed nuclear response.
    - `def step(self, dt, inputs)`
- **`class MechanicsLayer(Layer)`**
    cpm.py -- the project's own Cellular Potts engine, as a tissue layer.
    - `def step(self, dt, inputs)`

### `engine/params.py`  (217 行)

engine.params -- parameter registry with ENFORCED provenance.

项目内依赖：`.units`

- **`class ProvenanceError(ValueError)`**
- **`class Param`**
    - `def to_dict(self)`
- **`class ParamRegistry`**
    A named, validated collection of parameters.
    - `def add(self, param)`
    - `def define(self, name, value, unit='1', provenance=ILLUSTRATIVE, **kw)`
    - `def unit(self, name)`
    - `def quantity(self, name)`
    - `def get(self, name, default=None)`
    - `def names(self)`
    - `def validate(self)` — Every `derived` parameter must depend only on registered names.
    - `def check_species(self, model_species, allow_cross=False)` — Flag measured parameters from a different species.
    - `def check_stage(self, model_stage)` — Flag measured parameters from a different developmental stage.
    - `def summary(self)`
    - `def measured_names(self)`
    - `def unmeasured_names(self)`
    - `def require_measured(self, names)` — Refuse to make a validated claim about parameters that are not measured.
    - `def snapshot(self)`
    - `def to_json(self, indent=2)`
    - `def markdown_table(self)`

### `engine/profile.py`  (217 行)

engine.profile -- organism/scale profiles that encode verified constraints.

- **`class ProfileViolation(RuntimeError)`**
- **`class OrganismProfile`**
    - `def forbidden_ports(self)`
    - `def validate_pipeline(self, pipeline)` — Refuse a pipeline that contradicts this organism's physiology.
    - `def required_caveats(self)`
    - `def to_dict(self)`

### `engine/rules.py`  (215 行)

engine.rules -- a small declarative rule grammar for cell behaviour.

项目内依赖：`.units`

- **`class RuleError(ValueError)`**
- **`class Rule`**
    - `def test(self, state)`
    - `def as_text(self)`
    - `def to_dict(self)`
- **`class RuleSet`**
    - `def add(self, condition, target, effect, value, **kw)`
    - `def compile(self, ports)` — Validate every rule against the pipeline's declared ports.
    - `def fires(self, state)`
    - `def evaluate(self, state)` — Return {target: (effect, value, unit)} for every satisfied rule.
    - `def apply(self, state, target_values)` — Apply fired rules to a dict of target names -> current numbers.
    - `def provenance_summary(self)`
    - `def annotate(self)` — Auto-generated documentation (the PhysiCell 'annotation' lesson).
    - `def to_json(self)`

### `engine/embodied/force_channel.py`  (210 行)

THE CONTACT-FORCE CHANNEL IS A NON-PHYSICAL LOAD.  ONE DECLARATION, ENFORCED.

- **`def measured_residual(episode_dir) -> dict[str, Any]`**
    Recompute THIS episode's residual from its own array, so no number here is copied.
- **`def propagation() -> dict`**
    WHERE THE DOWNGRADE GOES NEXT.  Written down because the easy mistake is to stop at the array.
- **`def contract(residual=None) -> dict`**
    The four contract fields for any output derived from the contact-force channel.

### `engine/injury_tissue.py`  (208 行)

Illustrative injury reduction, NOT measured fly injury/viability physiology.

项目内依赖：`.local_tissue`

- **`def nernst_k_mV(inside_mM, outside_mM, temperature_K=298.15)`**
    Ideal activity approximation, monovalent K; no mM-to-M factor in ratio.
- **`class PotassiumConfig`**
- **`class InjuryPotassium`**
    Two finite free pools + finite bound pool, with explicit reservoir ledger.
    - `def total_amount(self)`
    - `def mass_balance_error(self)`
    - `def ek_mV(self)`
    - `def apply_injury(self, permeability_um_s, reservoir_conductance_um3_s=None)`
    - `def step(self, dt_s)`
- **`class InjuryLigand`**
    Piecewise LocalTissue, using fresh public construction at an injury event.
    - `def ledger(self)`
    - `def total_amount(self)`
    - `def mass_balance_error(self)`
    - `def switch_barrier(self, barrier_edges)` — Replace full barrier map; unlisted edges revert to D*A/dx, no mass jump.
    - `def step(self, dt_s)`

### `engine/embodied/visual_drive.py`  (202 行)

VISUAL DRIVE: the body -> brain channel the platform declared but never wired.

项目内依赖：`engine.embodied.adapters`, `engine.embodied.vision`

- **`class VisualDrive`**
    Body -> brain, the VISUAL channel.  Satisfies the same `SensoryDrive` seam as the receptors.
    - `def bandwidth(self) -> dict` — The number a reader must see before trusting anything this channel produces.
    - `def set_reference_pose(self, joint_angles_rad) -> None` — No-op BY DESIGN: a visual channel has no reference POSE.  Present only to satisfy
    - `def reset(self) -> None`
    - `def transduce(self, obs)` — (current_nA over the whole tier, drive report) for one observation.
    - `def bind_sim(self, sim) -> 'VisualDrive'`
    - `def dependence(self, obs, *, blank_luminance=0.0) -> dict` — THE GUARD.  A blinded scene must give a DIFFERENT drive, or the channel is not coupled.

### `engine/isolation_scenario.py`  (191 行)

Integrated IDEALIZED graph experiment, hand implemented, not fly physiology.

项目内依赖：`.electrode`, `.local_tissue`, `.neural_active`, `.neural_cond`, `.neural_hybrid`

- **`class IsolationConfig`**
    - `def validate(self)`
- **`def run_isolation_scenario(config=None)`**
    Return arrays and JSON-compatible metadata; never write files.

### `engine/electrode_recording.py`  (176 行)

Fixed-geometry recording diagnostic. One-sided PSDs in V²/Hz. Analog anti-aliasing precedes ideal sampling; causal digital SOS is NOT anti-alias. No measured amplifier parameters or biological-current claims.

项目内依赖：`.electrode_frontend`

- **`def positive(name, value, zero=False)`**
- **`def shared_reference_covariance(channel_psds, reference_psd)`**
    Independent channel minus common reference: diag(Si)+Sref*11T.
- **`class RecordingConfig`**
- **`class RecordingChain`**
    - `def analog_response(self, f)`
    - `def digital_response(self, f)`
    - `def input_psds(self, f)`
    - `def budget(self, points_per_decade=1200, f_min=1e-06, f_max=3000000.0)` — Integrate analog tails and digital-folded aliases using one-sided PSD.
    - `def demo(self, current_nA=0.1, duration_s=2.0, oversample=8)` — Charge-balanced source/return toy; causal zero-state signal.

### `engine/neural_cond.py`  (176 行)

Conductance point tier, explicit units and illustrative parameters.

- **`class ConductanceParams`**
    - `def validate(self)`
    - `def provenance(self)`
- **`class ConductanceNetwork`**
    - `def set_connectivity(self, exc_uS, inh_uS)`
    - `def step(self, current_nA=0.0, exc_events_uS=0.0, inh_events_uS=0.0, enabled=None)` — Events arrive at interval START, returned spikes at interval END.
    - `def run(self, duration_ms, current_nA=0.0, probe_indices=())`

### `engine/embodied/bridge_body.py`  (170 行)

Independent engineered intent -> shipped FlyGym hybrid turning controller.

- **`class BridgeBodyConfig`**
- **`class BridgeBody`**
    Use ``step(forward=1, yaw=0, duration_s=.01)`` or named commands.
    - `def dt(self) -> float`
    - `def snapshot(self) -> dict`
    - `def step(self, forward=0.0, yaw=0.0, *, duration_s=None, command=None, turn=None) -> dict`
    - `def observe(self) -> dict`
    - `def render(self) -> np.ndarray`
    - `def provenance(self) -> dict`
    - `def close(self)`

### `engine/checks.py`  (166 行)

engine.checks -- the verification harness.

- **`class ChecksFailed(RuntimeError)`**
- **`class Check`**
- **`class CheckResult`**
    - `def to_dict(self)`
- **`class CheckSuite`**
    - `def add(self, name, kind, fn, description='', tolerance=None)`
    - `def analytic(self, name, fn, description='', tolerance=1e-06)`
    - `def conservation(self, name, fn, description='', tolerance=1e-12)`
    - `def determinism(self, name, fn, description='', tolerance=0.0)`
    - `def invariant(self, name, fn, description='', tolerance=0.0)`
    - `def run(self, stop_on_error=False)`
    - `def summary(self, results=None)`
    - `def require_all_passed(self, results=None)`
    - `def markdown(self, results=None)`
    - `def to_json(self, results=None)`

### `engine/neural_hybrid.py`  (162 行)

Fixed-owner bounded hybrid coordinator, NOT a whole-brain implementation.

项目内依赖：`.neural_active`, `.neural_cond`

- **`class ConductanceEdge`**
- **`class Delivery`**
- **`class HybridNetwork`**
    Construct with explicit global IDs, point IDs and detailed ID->spec dict.
    - `def t_ms(self)`
    - `def step(self, point_current_nA=0.0)`
    - `def close(self)`

### `engine/record.py`  (156 行)

engine.record -- self-describing run records.

- **`class RunRecord`**
    - `def set_profile(self, profile)`
    - `def set_parameters(self, registry)`
    - `def set_wiring(self, pipeline)`
    - `def set_checks(self, suite, results=None)`
    - `def add_output(self, path, what='')`
    - `def add_note(self, text)`
    - `def add_caveat(self, text)`
    - `def set_metric(self, key, value)`
    - `def hash(self)`
    - `def write(self, path)`
    - `def markdown_claim_boundary(self)`

### `engine/electrode.py`  (121 行)

Finite spherical volume contacts in infinite homogeneous ohmic medium.

- **`def points(value)`**
- **`class Contact`**
- **`def transfer_mV_per_nA(xyz_um, contacts, conductivity_S_m)`**
    Green function for uniform spherical VOLUME sources; units mV/nA.
- **`class BipolarField`**
    - `def potential_mV(self, xyz_um, currents_nA)`
    - `def inward_axial_drive_nA(self, cable, currents_nA)` — For Vm=Vi-Ve, add sum_j g_ij*(Ve_j-Ve_i) to passive RHS.
- **`class VoltageRecorder`**
    Point voltage sampling + first-order bandwidth + noise and artifact.
    - `def step(self, dt_ms, source_contacts=(), outward_currents_nA=(), stimulus_field=None, stimulus_currents_nA=())`

### `engine/embodied/neural_motor_bridge.py`  (92 行)

Engineered, literature-informed spike-to-intent bridge (not anatomical reconstruction).

- **`class BridgeConfig`**
    - `def validate(self)`
- **`class SpikeIntentBridge`**
    Consume chronological binary spikes; output bounded engineering intent.
    - `def reset(self)`
    - `def advance(self, spikes, *, spike_dt_s, locomotion_permission=0.0, connected=True)`

### `engine/graded_vision.py`  (74 行)

Continuous graded photoreceptor release with an explicit chloride contract.

项目内依赖：`.receptors`

- **`class GradedVision`**
    - `def step(self, light, dt_ms=None)`
    - `def apply_to(self, net, light, dt_ms=None)` — Apply one step, requiring the network's inhibitory channel to match E_hist/tau.
    - `def describe(self)`
- **`def pure_histamine_mask(verified, pre)`**

### `engine/embodied/neural_command_audit.py`  (52 行)

Executable checks of neural-only command origin, not biological validation.

- **`def translate_motor_spikes(spikes, groups, window_s, reference_hz=100.0)`**
- **`def audit_interval(*, spikes, groups, window_s, reference_hz, actual_ctrl, sensory_current, sensory_indices, background_current, readout_connected=True)`**

### `engine/embodied/feco_identity.py`  (43 行)

Resolve BANC FeCO subtype by exact CSV subclass, never name similarity.

- **`def resolve_feco(csv_path, built)`**

### `engine/embodied/feco_translation.py`  (36 行)

Fail-closed FeCO input translation prerequisites; no movement decisions.

- **`def assess_feco_channel(channel, sample_interval_s)`**
    Explain why a candidate receptor cannot yet inject current automatically.

### `engine/embodied/autonomy_contract.py`  (25 行)

Fail-closed reporting contract; not proof that a runtime has no hidden controls.

- **`def assess_autonomy(manifest)`**

---

## 5. 这份文档刻意**不**声称的东西

- 不声称签名正确、不声称实现符合契约。它只把**签名与作者写在第一行的契约**集中起来，让你不必通读代码就能定位改动面。
- 不覆盖 `engine/**` 与顶层测量脚本之外的代码（测试、交付脚本、场景前端不在内）。
- 不替代运行时的行为检查：契约是否被遵守由 `run_all_checks.py` 的 37 项与不变量/注入回归负责。

