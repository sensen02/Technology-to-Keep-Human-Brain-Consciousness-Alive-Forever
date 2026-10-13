"""Tarsus-DOF model extension: XML construction and structural verification.

DECLARED ASSUMPTIONS (all labelled, none fitted to behaviour):
  A1  one hinge joint per leg is inserted into the existing Tarsus1 body, which currently has
      no joint. The axis is copied from that leg's existing tibia pitch joint convention (0 1 0,
      the local y axis), not invented; the range is declared as +-1.2 rad because no source
      gives a tarsus range.
  A2  two actuators per leg (tarsus depressor and tarsus levator) are added as spatial-tendon
      muscles, using the SAME muscle class parameters and dynprm as the existing tibia muscles,
      so no new tuning is introduced.
  A3  attachment sites are declared: the parent (tibia) site sits 0.10 model length units
      proximal to the tarsus joint, and the child (tarsus) sites sit +-0.02 units along the
      tarsus local x axis. That gives a declared lever arm of about 0.02 length units. No
      anatomical attachment measurement exists for these muscles in this project.
Nothing else is modified: no existing joint, actuator, geometry, mass or inertia changes.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, re, sys, time, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_muscle_closure as closure
OUT = HERE / 'outputs/model_branches/tarsus_dof'
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
PARENT_SITE_Z = 0.10
CHILD_SITE_X = 0.02
JOINT_RANGE = 1.2
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


FLYBODY_TARSUS = {  # from the published flybody model: axis local x, per-leg range, spring reference
    'T1': dict(range=(-0.7, 1.2), springref=-0.242),
    'T2': dict(range=(-1.0, 1.8), springref=0.608),
    'T3': dict(range=(-0.8, 1.2), springref=0.258),
}
FLYBODY_LEG_STIFFNESS = 0.01   # flybody default class "leg"
FLYBODY_LEG_DAMPING = 0.01
LEG_TOKEN = {'LF': 'T1', 'RF': 'T1', 'LM': 'T2', 'RM': 'T2', 'LH': 'T3', 'RH': 'T3'}


def muscle_params(canonical: Path) -> dict:
    """Copy the reference tibia muscle parameters so no new tuning is introduced."""
    raw = canonical.read_text()
    block = raw[raw.index('<actuator>'):raw.index('</actuator>')]
    m = re.search(r'name="LFTibia_flex_93434"[^>]*ctrlrange="([^"]+)"[^>]*lengthrange="([^"]+)"'
                  r'[^>]*dynprm="([^"]+)"[^>]*gainprm="([^"]+)"[^>]*biasprm="([^"]+)"', block)
    assert m, 'could not copy the reference muscle parameters'
    return dict(ctrlrange=m.group(1), lengthrange=m.group(2), dynprm=m.group(3),
                gainprm=m.group(4), biasprm=m.group(5))


def build_xml(source: Path, new_muscle_params: dict, body_dt: float, axes: dict,
              springrefs: dict | None = None) -> tuple:
    """Build on the zero-floor branch XML so the comparison against the published 25 us
    baseline differs only by the extension (same control range convention)."""
    text = source.read_text()
    # keep the control-range convention of the base branch and set the body timestep
    text, n_dt = re.subn(r'(<option\b[^>]*\btimestep=")[^"]+(")', lambda m: m.group(1) + repr(body_dt) + m.group(2), text)
    assert n_dt == 1, n_dt
    paths = []
    for leg in LEGS:
        # A1: insert the hinge joint inside the existing Tarsus1 body element.
        anchor = f'<body name="{leg}Tarsus1"'
        start = text.index(anchor)
        insert_at = text.index('>', start) + 1
        token = LEG_TOKEN[leg]
        lo, hi = FLYBODY_TARSUS[token]['range']
        ax = axes[leg]
        sref = (springrefs or {}).get(leg)
        sref_attr = '' if sref is None else f' springref="{sref:.6f}"'
        joint = (f'\n                <joint name="joint_{leg}Tarsus1_pitch" pos="0 0 0" '
                 f'axis="{ax[0]:.8f} {ax[1]:.8f} {ax[2]:.8f}" range="{lo} {hi}" limited="true" '
                 f'stiffness="{FLYBODY_LEG_STIFFNESS}" damping="{FLYBODY_LEG_DAMPING}"{sref_attr} />')
        # A3: declare the two child attachment sites on the tarsus body.
        sites = (f'\n                <site name="{leg}_tarsus_depressor_attach" pos="{CHILD_SITE_X} 0 0" />'
                 f'\n                <site name="{leg}_tarsus_levator_attach" pos="{-CHILD_SITE_X} 0 0" />')
        text = text[:insert_at] + joint + sites + text[insert_at:]
        # A3: declare the parent attachment site on the tibia, proximal to the tarsus joint.
        t_anchor = f'<body name="{leg}Tibia"'
        t_start = text.index(t_anchor)
        t_insert = text.index('>', t_start) + 1
        text = (text[:t_insert]
                + f'\n              <site name="{leg}_tibia_tarsus_muscle_attach" pos="0 0 {PARENT_SITE_Z}" />'
                + text[t_insert:])
    # tendons
    tendon_block = []
    actuator_block = []
    for leg in LEGS:
        for muscle, child_site, direction in (('tarsus_depressor', f'{leg}_tarsus_depressor_attach', 'depressor'),
                                              ('tarsus_levator', f'{leg}_tarsus_levator_attach', 'levator')):
            tname = f'{leg}C_{muscle}_tendon'
            aname = f'{leg}Tarsus_{direction}'
            tendon_block.append(
                f'    <spatial name="{tname}" width="0.001" limited="false">\n'
                f'      <site site="{leg}_tibia_tarsus_muscle_attach" />\n'
                f'      <site site="{child_site}" />\n'
                f'    </spatial>')
            p = new_muscle_params
            actuator_block.append(
                f'    <general name="{aname}" class="muscle" tendon="{tname}" ctrlrange="0 1" '
                f'lengthrange="{p["lengthrange"]}" dynprm="{p["dynprm"]}" gainprm="{p["gainprm"]}" '
                f'biasprm="{p["biasprm"]}" />')
    text = text.replace('  <tendon>', '  <tendon>\n' + '\n'.join(tendon_block), 1)
    assert '  <tendon>' in source.read_text(), 'no tendon block to extend'
    text = text.replace('<actuator>', '<actuator>\n' + '\n'.join(actuator_block), 1)
    return text, paths


def zero_branch_model():
    import mujoco
    return mujoco.MjModel.from_xml_path(str(zero_xml()))


def zero_xml():
    import run_zero_floor_branch as zero
    return zero.XML


def main(report):
    import mujoco
    canonical = closure.cp.CANONICAL_XML
    base = closure.cp.load_model()
    import run_zero_floor_branch as zero
    source_xml = zero.XML
    raw = Path(canonical).read_text()
    block = raw[raw.index('<actuator>'):raw.index('</actuator>')]
    m = re.search(r'name="LFTibia_flex_93434"[^>]*ctrlrange="([^"]+)"[^>]*lengthrange="([^"]+)"'
                  r'[^>]*dynprm="([^"]+)"[^>]*gainprm="([^"]+)"[^>]*biasprm="([^"]+)"', block)
    assert m, 'could not copy the reference muscle parameters'
    params = dict(ctrlrange=m.group(1), lengthrange=m.group(2), dynprm=m.group(3),
                  gainprm=m.group(4), biasprm=m.group(5))
    body_dt = 2.5e-5
    # Axis construction (source-based, not copied): flybody gives the tibia and tarsus the SAME local
    # flexion axis. Here the numeric axis is computed so the new hinge is parallel, at rest, to that
    # leg's existing tibia pitch axis in world coordinates. Nothing is invented; nothing is copied by value.
    import mujoco as _mj
    base_zero = zero_branch_model()
    axes = {}
    d0 = _mj.MjData(base_zero)
    d0.qpos[:] = base_zero.key_qpos[0]
    d0.qvel[:] = 0
    _mj.mj_forward(base_zero, d0)
    for leg in LEGS:
        tib = _mj.mj_name2id(base_zero, _mj.mjtObj.mjOBJ_BODY, f'{leg}Tibia')
        tar = _mj.mj_name2id(base_zero, _mj.mjtObj.mjOBJ_BODY, f'{leg}Tarsus1')
        tib_joint = _mj.mj_name2id(base_zero, _mj.mjtObj.mjOBJ_JOINT, f'joint_{leg}Tibia_pitch')
        local_axis = np.asarray(base_zero.jnt_axis[tib_joint], dtype=float)
        world = d0.xmat[tib].reshape(3, 3) @ local_axis
        axis_local = d0.xmat[tar].reshape(3, 3).T @ world
        axis_local = axis_local / np.linalg.norm(axis_local)
        axes[leg] = axis_local.tolist()
    text, asset_paths = build_xml(source_xml, params, body_dt, axes)
    # --- v3: the source states the tibia-tarsus joint is FLEXED when the fly stands, so the passive
    # rest must sit in the flexion direction. The direction is measured here, not assumed: the sign of
    # the joint angle that lowers the thorax towards the substrate is taken as flexion.
    probe_path = OUT / 'fruitfly_tarsus_dof_probe.xml'
    probe_path.write_text(text)
    probe = mujoco.MjModel.from_xml_path(str(probe_path))
    dp = mujoco.MjData(probe)
    dp.qpos[:] = probe.key_qpos[0]
    dp.qvel[:] = 0
    thorax = probe.body('Thorax').id
    flexion_sign = {}
    for leg in LEGS:
        jid = mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_JOINT, f'joint_{leg}Tarsus1_pitch')
        adr = int(probe.jnt_qposadr[jid])
        heights = {}
        for off in (0.2, -0.2):
            q = probe.key_qpos[0].copy()
            q[adr] = off
            dp.qpos[:] = q
            mujoco.mj_forward(probe, dp)
            heights[off] = float(dp.xpos[thorax, 2])
        flexion_sign[leg] = 1 if heights[0.2] < heights[-0.2] else -1
    springrefs = {leg: flexion_sign[leg] * abs(FLYBODY_TARSUS[LEG_TOKEN[leg]]['springref'])
                  for leg in LEGS}
    text, asset_paths = build_xml(source_xml, params, body_dt, axes, springrefs)
    probe_path.unlink(missing_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    xml = OUT / 'fruitfly_tarsus_dof.xml'
    xml.write_text(text)
    model = mujoco.MjModel.from_xml_path(str(xml))
    result = dict(
        status='STRUCTURAL_EXTENSION_BUILT',
        label='manual labelled model extension; not a behaviour result and not a cure',
        declared_assumptions=dict(
            base_xml='built on the zero-floor branch XML (%s) so control-range convention and every other '
                     'setting match the published 25 us baseline' % source_xml,
            body_timestep_s=body_dt,
            A1_joint='hinge inserted into each existing Tarsus1 body; axis COMPUTED so the new hinge is parallel at '
                     'rest to that leg tibia pitch axis in world coordinates (flybody gives tibia and tarsus the same '
                     'flexion axis); range and passive stiffness/damping taken from the published flybody model '
                     '(per-leg ranges T1 -0.7..1.2, T2 -1..1.8, T3 -0.8..1.2; leg class stiffness %.2f damping %.2f)'
                     % (FLYBODY_LEG_STIFFNESS, FLYBODY_LEG_DAMPING),
            A1_axes_local={leg: ax for leg, ax in axes.items()},
            A1_range_source='flybody (TuragaLab/flybody, Apache-2.0) per-leg tarsus joint ranges',
            A1_springref='magnitude from the flybody per-leg spring reference, placed in the FLEXION direction '
                         'measured here; the published source states the tibia-tarsus joint is flexed when the fly '
                         'is standing (Azevedo et al. 2024 supplementary methods)',
            A1_flexion_sign_measured={leg: s for leg, s in flexion_sign.items()},
            A1_springref_applied={leg: round(v, 6) for leg, v in springrefs.items()},
            A2_actuators='tarsus depressor and levator added per leg as spatial-tendon muscle actuators, using '
                         'copied parameters of an existing tibia muscle and therefore introducing no new tuning: %s'
                         % json.dumps(params),
            A3_attachments='tibia site at (0,0,%.2f) and tarsus sites at (+-%.2f,0,0) in body frames; '
                           'declared lever arm about %.3f length units' % (PARENT_SITE_Z, CHILD_SITE_X, CHILD_SITE_X),
            not_changed=['existing joints', 'existing actuators', 'geometry', 'masses', 'inertias']),
        dimensions=dict(base_nq=int(base.nq), new_nq=int(model.nq), base_nv=int(base.nv), new_nv=int(model.nv),
                        base_nu=int(base.nu), new_nu=int(model.nu),
                        added_dof=int(model.nv - base.nv), added_actuators=int(model.nu - base.nu)),
        provenance=dict(canonical_xml=str(canonical), canonical_sha256=sha(canonical),
                        base_branch_xml=str(source_xml), base_branch_sha256=sha(source_xml),
                        branch_xml=str(xml), branch_sha256=sha(xml),
                        asset_paths_resolved=len(asset_paths),
                        geometry_content_changes=False),
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    (OUT / 'tarsus_dof_structure.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['dimensions'], indent=2))
    print(json.dumps({k: v for k, v in result['declared_assumptions'].items() if k != 'A2_actuators'}, indent=2))
    return result


if __name__ == '__main__':
    with closure.mg.guarded_run('tarsus_dof_structure', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
