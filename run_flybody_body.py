"""BODY-level measurement of the published FlyBody model as a candidate body for this closure.

MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT.  Nothing in this file is a behaviour claim, and
nothing in it is credited to the neural circuit.

This branch deliberately does NOT touch the neural tier, the connectome, the spikes or the
clocks.  It asks questions about the BODY alone:

    Q1  What is FlyBody made of, in this project's terms -- joints per leg, actuators per leg,
        adhesion, muscles, activation states, unit system?
    Q2  With its own published actuators and its own published neutral command (every leg
        actuator is a position servo whose ctrl=0 is the neutral posture), and with no neural
        drive at all, does the fly carry its own weight on a flat floor for 200 ms?
    Q3  What changes when the servo term is removed (passive joint springs alone), and what
        changes when the published claw adhesion actuators are switched on?

Arms (all experimenter-set, none neural):
  published_command          every actuator at its published neutral ctrl=0, adhesion off
  published_plus_adhesion    neutral ctrl=0, six published claw adhesion actuators at 1
  springs_only               actuator force removed at the XML level, passive springs alone

Height criterion for THIS body: FlyBody's root body (`thorax`) sits at the free joint, so the
root origin is near the top of the animal and is NOT a standing height.  The reported height is
therefore the distance from the fly's own centre of mass (`data.subtree_com`) to the floor
plane, in mm.  Escape is the first time that height falls below 90% of its seated value, the
same 10% rule this project uses for its canonical body.

Unit reading: FlyBody is CGS (cm, g, s).  The evidence is asserted at runtime in
`unit_reading()` from the model's own compiled values -- gravity 981, total mass 9.8462e-4,
air density 1.28e-3 -- and every reported length is converted to mm and every force to uN
through the constants declared once in CONVERSION.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine.memory_guard as mg
import engine.canonical_physics as cp

OUT = HERE / 'outputs/model_branches/flybody_body'
FLYBODY = HERE / 'upstream/flybody'
XML = FLYBODY / 'floor.xml'          # published arena: <include fruitfly.xml> + floor plane
XML_SHA256 = '1490751ff170d089b8882aa4972778b64bcf56f6d39b0e87029156dcf0452019'
FRUITFLY_SHA256 = 'd14946fd0311025ecca70c8eeb5de80e1fe18700d3072be37ecbb18d33d80fd8'
DT_BODY = .0001                      # the model's own published timestep, asserted below
DURATION = .2

# FlyBody is CGS: length cm, mass g, time s, force g*cm/s^2 = 1e-5 N.  Declared once.
CONVERSION = {'mm_per_model_length': 10.0, 'mg_per_model_mass': 1000.0,
              'uN_per_model_force': 10.0,
              'note': '1 model length = 1 cm; 1 model mass = 1 g; 1 model force = 1 g*cm/s^2 = 1e-5 N'}
LEG_OF_TAG = {'T1': 'F', 'T2': 'M', 'T3': 'H'}
#: Inverse map, used by anything that has to name a FlyBody element for one of this project's legs.
LEG_TAG = {'LF': ('T1', 'left'), 'RF': ('T1', 'right'), 'LM': ('T2', 'left'),
           'RM': ('T2', 'right'), 'LH': ('T3', 'left'), 'RH': ('T3', 'right')}
CANONICAL_LEGS = ('LF', 'RF', 'LM', 'RM', 'LH', 'RH')


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model():
    import mujoco
    if sha256(XML) != XML_SHA256 or sha256(FLYBODY / 'fruitfly.xml') != FRUITFLY_SHA256:
        raise ValueError('published FlyBody asset changed; refusing to run')
    model = mujoco.MjModel.from_xml_path(str(XML))
    if not np.isclose(model.opt.timestep, DT_BODY, rtol=0, atol=1e-15):
        raise ValueError('FlyBody timestep differs from the clock this project benchmarks against')
    return model


def names(model, objtype):
    import mujoco
    n = {mujoco.mjtObj.mjOBJ_GEOM: model.ngeom, mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
         mujoco.mjtObj.mjOBJ_JOINT: model.njnt, mujoco.mjtObj.mjOBJ_BODY: model.nbody}[objtype]
    return [(i, mujoco.mj_id2name(model, objtype, i) or '') for i in range(n)]


def leg_of(name):
    """Map a FlyBody leg element name to this project's leg label (LF/RF/LM/RM/LH/RH).

    FlyBody names its collision geoms with a `_collision` suffix, so the suffix is stripped
    before the leg tag is read; a name with no leg tag returns None and is never guessed.
    """
    if name.endswith('_collision'):
        name = name[:-len('_collision')]
    for tag, letter in LEG_OF_TAG.items():
        for side in ('left', 'right'):
            if name.endswith(f'_{tag}_{side}'):
                return ('L' if side == 'left' else 'R') + letter
    return None


def unit_reading(model):
    """Read the model's unit system off its own compiled numbers; never assume it."""
    gravity, mass, density = (abs(float(model.opt.gravity[2])), float(model.body_mass.sum()),
                              float(model.opt.density))
    checks = {
        'gravity_model_units_per_s2': gravity,
        'gravity_is_cm_s2_for_9p81_m_s2': bool(np.isclose(gravity, 981.0, rtol=1e-9)),
        'total_mass_model_units': mass,
        'mass_in_mg_if_unit_is_gram': float(mass*CONVERSION['mg_per_model_mass']),
        'air_density_model_units_per_length3': density,
        'density_is_g_per_cm3_for_air': bool(np.isclose(density, 1.28e-3, rtol=.02)),
        'timestep_s': float(model.opt.timestep),
        'length_unit': 'cm', 'mass_unit': 'g', 'time_unit': 's',
        'force_unit': 'g*cm/s^2 = 1e-5 N',
    }
    if not (checks['gravity_is_cm_s2_for_9p81_m_s2'] and checks['density_is_g_per_cm3_for_air']):
        raise ValueError('FlyBody unit reading does not hold; the CGS conversion would be wrong')
    checks['body_weight_model_force'] = mass*gravity
    checks['body_weight_uN'] = float(mass*gravity*CONVERSION['uN_per_model_force'])
    if not 5.0 < checks['body_weight_uN'] < 20.0:
        raise ValueError(f"body weight {checks['body_weight_uN']:.3f} uN is not a fly's weight, so "
                         'the unit reading would be wrong')
    return checks


def body_census(model):
    """What this body has that the canonical body lacks, and what it lacks that the other has."""
    import mujoco
    per_leg, claw_geoms, tarsus_geoms = {}, {}, {}
    for i, n in names(model, mujoco.mjtObj.mjOBJ_GEOM):
        leg = leg_of(n)
        if not (leg and 'collision' in n):
            continue
        if n.startswith('tarsal_claw'):
            claw_geoms[n] = leg
        elif n.startswith('tarsus'):
            tarsus_geoms[n] = leg
    for leg in CANONICAL_LEGS:
        joints = [(i, n) for i, n in names(model, mujoco.mjtObj.mjOBJ_JOINT) if leg_of(n) == leg]
        acts = [(i, n) for i, n in names(model, mujoco.mjtObj.mjOBJ_ACTUATOR) if leg_of(n) == leg]
        per_leg[leg] = dict(
            joints=[n for _, n in joints], n_joints=len(joints),
            joint_ranges_rad={n: [float(model.jnt_range[i][0]), float(model.jnt_range[i][1])]
                              for i, n in joints},
            joint_springref_rad={n: float(model.qpos_spring[int(model.jnt_qposadr[i])])
                                 for i, n in joints},
            joint_stiffness={n: float(model.jnt_stiffness[i]) for i, n in joints},
            joint_damping={n: float(model.dof_damping[int(model.jnt_dofadr[i])]) for i, n in joints},
            joint_actuated={n: [a for a, an in names(model, mujoco.mjtObj.mjOBJ_ACTUATOR)
                                if an == n] for i, n in joints},
            actuators=[n for _, n in acts], n_actuators=len(acts),
            actuator_force_range={n: [float(model.actuator_forcerange[i][0]),
                                      float(model.actuator_forcerange[i][1])] for i, n in acts},
            actuator_ctrl_range={n: [float(model.actuator_ctrlrange[i][0]),
                                     float(model.actuator_ctrlrange[i][1])] for i, n in acts})
    adhesion = []
    for i, n in names(model, mujoco.mjtObj.mjOBJ_ACTUATOR):
        if n.startswith('adhere_'):
            adhesion.append(dict(actuator=n, index=i,
                                 kind='claw' if n.startswith('adhere_claw') else 'labrum',
                                 ctrlrange=[float(model.actuator_ctrlrange[i][0]),
                                            float(model.actuator_ctrlrange[i][1])],
                                 attached_body=mujoco.mj_id2name(
                                     model, mujoco.mjtObj.mjOBJ_BODY, int(model.actuator_trnid[i][0])),
                                 leg=leg_of(n)))
    return dict(nq=int(model.nq), nv=int(model.nv), nu=int(model.nu), njnt=int(model.njnt),
                nbody=int(model.nbody), ngeom=int(model.ngeom), na=int(model.na),
                per_leg=per_leg, n_joints_per_leg=sorted({v['n_joints'] for v in per_leg.values()}),
                n_actuators_per_leg=sorted({v['n_actuators'] for v in per_leg.values()}),
                claw_geom_names=sorted(claw_geoms), claw_legs={n: claw_geoms[n] for n in sorted(claw_geoms)},
                n_claw_collision_geoms=len(claw_geoms),
                claw_collision_geom_names=sorted(claw_geoms),
                n_tarsus_collision_geoms=len(tarsus_geoms),
                adhesion_actuators=adhesion,
                n_adhesion_actuators=len(adhesion),
                actuator_kind='published joint position servo: force = gain*(ctrl - qpos), '
                              'gain 0.8 (coxa, femur), 0.4 (tibia, tarsus), so ctrl=0 IS the '
                              'neutral posture command',
                passive_joint_spring='stiffness 0.01, damping 0.01 (tibia damping 0.004; tarsomeres '
                                     'stiffness 0.1, damping 0.002); springref per joint encodes the '
                                     'middle/hind-leg neutral offsets',
                coupling='the tarsus2 actuator drives tarsomeres 2..5 through a fixed tendon',
                no_muscle='the published model has no muscle, no attachment site and no activation '
                          'state (na=0), so muscle-level motor resolution is LOST by this body')


def floor_geom(model):
    import mujoco  # noqa: F401  (kept local: this module imports mujoco only where it is used)
    return next(i for i, n in names(model, mujoco.mjtObj.mjOBJ_GEOM) if n == 'floor')


def claw_geoms(model):
    import mujoco  # noqa: F401
    return [i for i, n in names(model, mujoco.mjtObj.mjOBJ_GEOM) if n.startswith('tarsal_claw')]


def seat(model, data, gap_model=0.0, iters=60):
    """Put the free root where the lowest published claw is exactly `gap_model` from the floor.

    Distance is measured with `mj_geomDistance` on the published claw collision geoms against the
    published floor plane -- the same instrument this project uses on its canonical body.
    """
    import mujoco
    claws, floor = claw_geoms(model), floor_geom(model)
    if len(claws) != 12:
        raise ValueError(f'expected 12 claw collision geoms, found {len(claws)}')

    def lowest(z):
        data.qpos[:] = 0.0
        data.qpos[2] = z
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        return min(float(mujoco.mj_geomDistance(model, data, g, floor, 1e3, None)) for g in claws)

    lo, hi = -1.0, 1.0
    if not lowest(lo) < gap_model < lowest(hi):
        raise ValueError('cannot bracket a root height for the published neutral pose')
    for _ in range(iters):
        mid = .5*(lo+hi)
        if lowest(mid) < gap_model:
            lo = mid
        else:
            hi = mid
    z = .5*(lo+hi)
    lowest(z)
    return float(z)


def contacts(model, data, claw_leg):
    """Floor contacts split into per-claw-leg support and everything else (body touching down)."""
    import mujoco
    floor = floor_geom(model)
    per_leg = {leg: 0.0 for leg in CANONICAL_LEGS}
    n_claw, nonfoot, total_claw, total = 0, [], 0.0, 0.0
    force = np.zeros(6)
    for ci in range(data.ncon):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        if floor not in (g1, g2):
            continue
        other = g2 if g1 == floor else g1
        mujoco.mj_contactForce(model, data, ci, force)
        magnitude = float(np.linalg.norm(force[:3]))
        total += magnitude
        leg = claw_leg.get(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or '')
        if leg is None:
            nonfoot.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or f'geom{other}')
        else:
            per_leg[leg] += magnitude
            total_claw += magnitude
            n_claw += 1
    return per_leg, total_claw, total, n_claw, sorted(set(nonfoot))


def upright_angle_deg(model, data):
    """Angle between the thorax body's own +z axis and world +z: 0 deg is upright."""
    import mujoco
    rot = data.xmat[model.body('thorax').id].reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip(rot[2, 2], -1.0, 1.0))))


def episode(model, arm, duration=DURATION, stride=10):
    import mujoco
    claw_leg = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g):
                leg_of(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)) for g in claw_geoms(model)}
    claw_act = [i for i, n in names(model, mujoco.mjtObj.mjOBJ_ACTUATOR) if n.startswith('adhere_claw')]
    data = mujoco.MjData(model)
    z_seated = seat(model, data, gap_model=0.0)
    seated = data.qpos.copy()
    mujoco.mj_forward(model, data)
    thorax = model.body('thorax').id
    floor_z = float(data.geom_xpos[floor_geom(model)][2])
    weight = float(model.body_mass.sum()*abs(model.opt.gravity[2]))
    # springs_only removes the servo term entirely.  This is an XML-level variant of the published
    # model, written into the compiled model before any step, NOT a control input.
    if arm == 'springs_only':
        model.actuator_gainprm[:, 0] = 0.0
        model.actuator_biasprm[:] = 0.0
    rows, escape, actuator_force_max = [], None, 0.0
    h0 = float(data.subtree_com[thorax][2]) - floor_z
    n_steps = int(round(duration/DT_BODY))
    for step in range(n_steps+1):
        data.ctrl[:] = 0.0
        if arm == 'published_plus_adhesion':
            data.ctrl[claw_act] = 1.0
        if step % stride == 0:
            per_leg, total_claw, total, n_claw, nonfoot = contacts(model, data, claw_leg)
            com = float(data.subtree_com[thorax][2]) - floor_z
            clearance = min(float(mujoco.mj_geomDistance(model, data, g, floor_geom(model), 1e3, None))
                            for g in claw_geoms(model))
            rows.append(dict(time_s=float(data.time), com_height_mm=com*CONVERSION['mm_per_model_length'],
                             root_z_mm=float(data.qpos[2])*CONVERSION['mm_per_model_length'],
                             claw_clearance_mm=clearance*CONVERSION['mm_per_model_length'],
                             upright_deg=upright_angle_deg(model, data),
                             claw_support_fraction=total_claw/weight,
                             total_support_fraction=total/weight,
                             n_claw_contacts=n_claw, n_contacts=int(data.ncon),
                             nonfoot_geoms=nonfoot,
                             per_leg_claw_support={k: v/weight for k, v in per_leg.items()}))
            if escape is None and com < .9*h0:
                escape = float(data.time)
        actuator_force_max = max(actuator_force_max, float(np.max(np.abs(data.actuator_force))))
        if step < n_steps:
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
    if arm == 'springs_only' and actuator_force_max != 0.0:
        raise AssertionError('springs_only arm still has actuator force; the variant did not apply')
    com_h = np.array([r['com_height_mm'] for r in rows])
    t = np.array([r['time_s'] for r in rows])
    claw_sup = np.array([r['claw_support_fraction'] for r in rows])
    total_sup = np.array([r['total_support_fraction'] for r in rows])
    tail = t >= duration - .05
    nonfoot_ever = sorted({g for r in rows for g in r['nonfoot_geoms']})
    summary = dict(
        arm=arm, duration_s=duration, seated_root_z_mm=z_seated*CONVERSION['mm_per_model_length'],
        seated_com_height_mm=h0*CONVERSION['mm_per_model_length'], escape_time_s=escape,
        com_height_min_mm=float(com_h.min()), com_height_final_mm=float(com_h[-1]),
        com_height_drop_fraction=float(1-com_h[-1]/h0),
        upright_deg_max=float(max(r['upright_deg'] for r in rows)),
        upright_deg_final=float(rows[-1]['upright_deg']),
        claw_clearance_min_mm=float(min(r['claw_clearance_mm'] for r in rows)),
        claw_support_fraction_at_t0=float(claw_sup[0]),
        claw_support_fraction_mean_last_50ms=float(claw_sup[tail].mean()),
        total_support_fraction_mean_last_50ms=float(total_sup[tail].mean()),
        n_claw_contacts_at_t0=int(rows[0]['n_claw_contacts']),
        n_claw_contacts_final=int(rows[-1]['n_claw_contacts']),
        per_leg_claw_support_final={k: float(v) for k, v in rows[-1]['per_leg_claw_support'].items()},
        per_leg_claw_support_final_uN={k: float(v*weight*CONVERSION['uN_per_model_force'])
                                       for k, v in rows[-1]['per_leg_claw_support'].items()},
        body_floor_geoms_contacted=nonfoot_ever,
        actuator_force_max_model_units=actuator_force_max,
        actuator_force_max_uN=actuator_force_max*CONVERSION['uN_per_model_force'],
        note='claw_support_fraction = summed claw-to-floor contact force / (total mass * |gravity|); '
             'total_support_fraction also counts any non-claw body geom resting on the floor')
    return summary, dict(rows=rows, seated_qpos=seated,
                         floor_z_model=floor_z, body_weight_model=weight)


def canonical_comparison():
    """Quote (never re-measure) the published canonical-body numbers used for comparison."""
    path = HERE / 'outputs/model_branches/passive_vs_actuated'
    out = dict(source_dir=str(path), files=[])
    if path.exists():
        for f in sorted(path.glob('*.json')):
            out['files'].append(f.name)
            out[f.stem] = json.loads(f.read_text())
    else:
        out['note'] = 'canonical comparison artifacts not found in this checkout'
    return out


def figure(result, traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    for arm in result['arms']:
        rows = traces[arm]['rows']
        t = [r['time_s']*1e3 for r in rows]
        axes[0].plot(t, [r['com_height_mm'] for r in rows], label=arm)
        axes[1].plot(t, [r['claw_support_fraction'] for r in rows], label=arm)
        axes[2].plot(t, [r['upright_deg'] for r in rows], label=arm)
        axes[3].plot(t, [r['n_claw_contacts'] for r in rows], label=arm)
    axes[0].set(xlabel='time (ms)', ylabel='centre-of-mass height (mm)',
                title='FlyBody standing height, zero neural drive')
    axes[1].set(xlabel='time (ms)', ylabel='claw support / body weight', ylim=(0, 2),
                title='does the published body carry its own weight?')
    axes[1].axhline(1.0, color='k', ls=':', lw=1)
    axes[2].set(xlabel='time (ms)', ylabel='thorax tilt (deg)', title='uprightness (0 deg = upright)')
    axes[3].set(xlabel='time (ms)', ylabel='claw contacts', title='feet on the floor')
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.suptitle('FlyBody body-level measurement, no neural circuit involved '
                 '(manual research prototype, not a product)', fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'flybody_body.png'
    fig.savefig(path, dpi=130)
    return path


def main(guard_report=None):
    import mujoco
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = load_model()
    units, census = unit_reading(model), body_census(model)
    arms, traces = {}, {}
    for arm in ('published_command', 'published_plus_adhesion', 'springs_only'):
        model = load_model()          # fresh compiled model per arm: the variant never leaks
        summary, arrays = episode(model, arm)
        arms[arm] = summary
        traces[arm] = arrays
        print(arm, 'escape', summary['escape_time_s'],
              'com final mm', round(summary['com_height_final_mm'], 4),
              'claw support last50ms', round(summary['claw_support_fraction_mean_last_50ms'], 3),
              'body contacts', summary['body_floor_geoms_contacted'], flush=True)
    question = ('With its own published actuators and neutral command and no neural drive, does the '
                'published FlyBody model carry its own weight on a floor, and what does its body '
                'provide that this project\'s canonical body does not?')
    provenance = dict(
        code=str(Path(__file__).resolve()), interpreter=sys.executable,
        xml=str(XML), xml_sha256=sha256(XML),
        fruitfly_xml_sha256=sha256(FLYBODY/'fruitfly.xml'),
        asset_commit=json.loads((FLYBODY/'PROVENANCE.json').read_text())['commit'],
        repository='https://github.com/TuragaLab/flybody', license='Apache-2.0',
        asset_edits='none: the published XML and meshes are used verbatim; the springs_only arm is a '
                    'runtime variant of the compiled model and is reported as one',
        mujoco_version=mujoco.__version__,
        clock=dict(body_dt_s=DT_BODY, duration_s=DURATION, samples_per_ms=10),
        memory_guard=guard_report('completed') if guard_report else None,
        wall_s=time.time()-started)
    not_supported = [
        'No neuron, spike, synapse, connectome or clock appears in this file at all: it is a body '
        'measurement, and no result here is credited to a neural circuit.',
        'The posture that holds the fly up comes from the PUBLISHED MODEL\'S OWN per-joint position '
        'servo (force = gain*(ctrl - qpos)) with the experimenter setting ctrl=0.  That is a '
        'model-internal posture hold, not a neural one, and a connectome-driven version would have '
        'to supply the target itself.',
        'The adhesion arm is the experimenter setting ctrl=1 on the published adhesion actuators.  '
        'This project has no adhesion motor pool and claims none.',
        'The springs_only arm removes actuator force at the XML level; it is a declared variant, not '
        'a biological condition, and it is what shows whether the servo is load-bearing.',
        'Nothing here is walking, gait, escape or any behaviour, and this project\'s standing '
        'criteria were not applied to FlyBody in this file.',
        'Being published does not make it calibrated for this loop: FlyBody has no muscle, no '
        'attachment site and no activation state, so muscle-level motor resolution is lost, not '
        'gained, by adopting this body.',
    ]
    cp.write_result(OUT/'flybody_body.json', question=question, status='PASS',
                    measured=dict(arms=arms, body_census=census, unit_reading_evidence=units,
                                  canonical_comparison=canonical_comparison(),
                                  seated_pose=dict(
                                      definition='root pose with every joint at 0, free root moved so '
                                                 'the lowest published claw geom is exactly on the '
                                                 'published floor plane',
                                      root_z_mm=arms['published_command']['seated_root_z_mm'],
                                      com_height_mm=arms['published_command']['seated_com_height_mm'],
                                      why_zero_is_the_right_neutral='FlyBody defines every range and '
                                          'springref around the neutral posture, so qpos=0 IS its '
                                          'published neutral pose; the per-leg springrefs carry the '
                                          'middle/hind-leg offsets')),
                    provenance=provenance, units={**units, **CONVERSION, 'length_reported': 'mm',
                                                  'force_reported': 'uN', 'time': 's', 'angle': 'deg'},
                    label='MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT; body-level measurement only',
                    status_meaning='PASS means the measurement ran and its unit reading held; it is '
                                   'not a behaviour result and not a neural result',
                    illustrative_vs_measured='Simulator output measured; the neutral-command choice, '
                                             'the adhesion arm and the springs_only variant are '
                                             'DECLARED experimenter settings',
                    what_this_does_not_support=not_supported)
    np.savez_compressed(OUT/'flybody_body_traces.npz',
                        **{f'{arm}_{k}': v for arm, row in traces.items() for k, v in row.items()
                           if k != 'rows'},
                        **{f'{arm}_series': json.dumps(row['rows'])
                           for arm, row in traces.items()})
    print('figure', figure({'arms': arms}, traces), flush=True)
    return dict(arms=arms, census=census, units=units)


if __name__ == '__main__':
    with mg.guarded_run('flybody_body', need_mb=1600.0, cap_mb=4608.0) as _rep:
        main(guard_report=_rep)
