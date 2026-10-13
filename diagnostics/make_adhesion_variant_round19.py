"""Generate adhesion model variants for round19 by TEXT insertion into a COPY of the asset.

WHY THIS FILE EXISTS.
    The canonical muscle asset this project runs
    (outputs/muscles_six_legs/fruitfly_six_leg_standing.xml, SHA256
    65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6) comes from the
    FlyMimic musculoskeletal branch, which documents "FlyMimic has no adhesion"
    (flygym/compose/fly/musculoskeletal.py: leg_to_adhesionactuator is always empty).
    The FlyBody branch of the same installed package does have adhesion, and its API
    (Fly.add_leg_adhesion, flygym/compose/fly/base_fly.py) adds one actuator per leg on
    the distal tarsus5 body with a gain and ctrlrange (0, 1).  MuJoCo implements that
    element as the native <adhesion> shortcut, documented as:

        transmission target is a BODY; the actuator injects forces AT CONTACTS in the
        NORMAL direction; the force is divided equally between the contacts of that
        body's geoms; with no geom gap set it requires ACTIVE CONTACTS; control range
        must be non-negative; gain is in force units and total force = ctrl * gain.
        Underlying general attributes: dyntype none, dynprm "1 0 0", gaintype fixed,
        gainprm "gain 0 0", biastype none, biasprm "0 0 0", trntype body,
        ctrllimited true.

    So adding it to THIS model is a real model change and must live in a variant.

HOW THE EDIT IS DONE, AND WHY THAT WAY.
    The variant is produced by inserting text immediately before the single
    '</actuator>' tag of a byte copy of the canonical file.  No XML round-trip, no
    re-serialisation: every byte of the canonical document other than the six inserted
    actuator elements is preserved exactly, so the 90 muscle actuators and all tendon
    routes cannot drift.  The script asserts that.

WHAT THIS IS NOT.
    A candidate physical correction based on the engine's own documented default inside
    the measured biological range.  NOT a tuned parameter, NOT a controller, NOT
    autonomous standing.  The canonical asset is deliberately left unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
CANONICAL_SHA = '65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6'
VARIANT_DIR = ROOT / 'outputs/muscles_six_leg_standing_adhesion'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
# gain 40.0 is FlyGym's own documented default (Fly.add_leg_adhesion(gain=40.0)); in this
# model's force units (1 unit = 1 uN) that is 40 uN / 24.4464 uN = 1.64 body weights per leg.
GAIN_ENGINE_DEFAULT = 40.0
# 2.0 body weights per leg is the TOP of the measured range reported by the earlier
# adhesion work (Ramdya et al. 2017 Nat Commun 8:14494).  It is a stated sensitivity
# endpoint, NOT a value tuned to any outcome.
GAIN_MEASURED_TOP_BW = 2.0

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build_variant(gain: float, out_path: Path) -> dict:
    text = CANONICAL.read_text()
    assert text.count('</actuator>') == 1, 'expected exactly one actuator section'
    assert text.count('<actuator>') == 1
    # the six adhesion actuators: one per leg, on that leg's distal tarsus (tarsus5) body,
    # named for the leg, exactly reproducing MuJoCo's documented adhesion shortcut.
    block = '  <!-- round19 candidate physical correction: engine-documented leg adhesion -->\n'
    for leg in LEGS:
        block += (f'  <adhesion name="{leg}Tarsus5_adhesion" body="{leg}Tarsus5" '
                  f'gain="{gain:g}" ctrlrange="0 1" />\n')
    out_text = text.replace('</actuator>', block + '</actuator>')
    # every byte before the insertion point must be untouched
    insert_at = text.index('</actuator>')
    assert out_text[:insert_at] == text[:insert_at], 'canonical prefix changed'
    assert out_text[insert_at + len(block):] == text[insert_at:], 'canonical suffix changed'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_text)
    return {'file': str(out_path), 'gain': gain, 'sha256': sha256(out_path),
            'inserted_elements': [f'{leg}Tarsus5_adhesion' for leg in LEGS]}

def validate(variant: Path, canonical_model, variant_model) -> dict:
    import mujoco as mj
    out = {'variant': str(variant),
           'nq': int(variant_model.nq), 'nv': int(variant_model.nv),
           'njnt': int(variant_model.njnt), 'nu': int(variant_model.nu),
           'ntendon': int(variant_model.ntendon)}
    assert variant_model.nq == canonical_model.nq
    assert variant_model.nv == canonical_model.nv
    assert variant_model.njnt == canonical_model.njnt
    assert variant_model.ntendon == canonical_model.ntendon
    assert variant_model.nu == canonical_model.nu + 6, 'nu must grow by exactly 6'
    # the 90 muscle actuators must be identical, in order
    n0 = canonical_model.nu
    for i in range(n0):
        a, b = canonical_model.actuator(i), variant_model.actuator(i)
        assert a.name == b.name
        for f in ('gaintype', 'biastype', 'dyntype', 'trntype', 'gainprm', 'biasprm',
                  'dynprm', 'gear', 'ctrlrange', 'trnid', 'actadr', 'actlimited'):
            va, vb = getattr(a, f), getattr(b, f)
            assert (va == vb).all() if hasattr(va, 'all') else va == vb, f'{a.name}:{f}'
    # tendon lengths identical
    d0, d1 = mj.MjData(canonical_model), mj.MjData(variant_model)
    mj.mj_forward(canonical_model, d0)
    mj.mj_forward(variant_model, d1)
    tl0 = d0.ten_length.copy()
    tl1 = d1.ten_length[:canonical_model.ntendon].copy()
    assert tl0.shape == tl1.shape
    max_tendon_delta = float(abs(tl0 - tl1).max())
    out['max_tendon_length_delta'] = max_tendon_delta
    out['muscle_actuator_fields_identical'] = True
    # the six added actuators must match MuJoCo's documented adhesion expansion
    rows = []
    for i in range(n0, variant_model.nu):
        a = variant_model.actuator(i)
        assert int(a.trntype[0]) == int(mj.mjtTrn.mjTRN_BODY), 'adhesion must transmit to a body'
        assert int(a.dyntype[0]) == 0, 'dyntype must be none'
        assert int(a.gaintype[0]) == int(mj.mjtGain.mjGAIN_FIXED)
        assert int(a.biastype[0]) == int(mj.mjtBias.mjBIAS_NONE)
        assert a.ctrllimited == 1 and a.ctrlrange[0] == 0.0 and a.ctrlrange[1] == 1.0
        body = mj.mj_id2name(variant_model, mj.mjtObj.mjOBJ_BODY, int(a.trnid[0]))
        rows.append({'name': a.name, 'body': body, 'gain': float(a.gainprm[0]),
                     'trntype': int(a.trntype[0]), 'dyntype': int(a.dyntype[0]),
                     'gaintype': int(a.gaintype[0]), 'biastype': int(a.biastype[0]),
                     'ctrlrange': a.ctrlrange.tolist(),
                     'dynprm_first': float(a.dynprm[0])})
    out['adhesion_actuators'] = rows
    assert {r['body'] for r in rows} == {f'{leg}Tarsus5' for leg in LEGS}
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=VARIANT_DIR)
    args = ap.parse_args()
    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    assert sha256(CANONICAL) == CANONICAL_SHA, 'canonical asset hash changed'
    # Unit check performed here rather than assumed: the model documents neither the mass
    # nor the force unit, so body weight in model force units is measured directly, and every
    # headline number in the round19 report is ALSO given in unit-free body-weight fractions.
    bw = float(mj.mj_getTotalmass(_load_mjcf(str(CANONICAL)).compile())
               * abs(float(_load_mjcf(str(CANONICAL)).compile().opt.gravity[2])))
    gains = {'adhesion_engine_default': GAIN_ENGINE_DEFAULT,
             'adhesion_measured_top': GAIN_MEASURED_TOP_BW * bw}

    canonical_model = _load_mjcf(str(CANONICAL)).compile()
    report = {'status': 'ADHESION_VARIANT_GENERATION',
              'canonical': {'file': str(CANONICAL), 'sha256_before': sha256(CANONICAL)},
              'mechanism_source': {
                  'api': 'flygym/compose/fly/base_fly.py Fly.add_leg_adhesion -> '
                         'add_actuator(kind="adhesion", body=tarsus5, gain=..., ctrlrange=(0,1))',
                  'shipped_element': 'flygym/assets/model/flybody/fruitfly.xml uses the native '
                                     '<adhesion group="3" ctrlrange="0 1" gain="0.985"/> shortcut',
                  'mujoco_semantics': 'body-transmission actuator injecting forces at contacts in '
                                      'the contact-normal direction, divided equally among the '
                                      'contacts of the target body; requires active contacts '
                                      'unless the geoms declare a gap',
                  'this_model_branch': 'FlyMimic musculoskeletal documents "FlyMimic has no '
                                       'adhesion"; leg_to_adhesionactuator is always empty'},
              'body_weight_force_units': bw,
              'variants': [], 'validation': []}
    for label, gain in gains.items():
        path = args.out_dir / f'fruitfly_six_leg_standing_{label}.xml'
        info = build_variant(gain, path)
        info['label'] = label
        info['body_weights_per_leg_at_ctrl1'] = gain / bw
        vm = _load_mjcf(str(path)).compile()
        report['variants'].append(info)
        report['validation'].append(validate(path, canonical_model, vm))
    report['canonical']['sha256_after'] = sha256(CANONICAL)
    assert report['canonical']['sha256_after'] == CANONICAL_SHA
    out_json = args.out_dir / 'variant_generation.json'
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps({'body_weight_force_units': bw,
                      'variants': [{k: v for k, v in x.items() if k != 'inserted_elements'}
                                   for x in report['variants']],
                      'validation': [{'variant': v['variant'], 'nu': v['nu'],
                                      'max_tendon_length_delta': v['max_tendon_length_delta'],
                                      'adhesion_name_body_gain': [(r['name'], r['body'], r['gain'])
                                                                  for r in v['adhesion_actuators']]}
                                     for v in report['validation']]}, indent=2))
    print(out_json)

if __name__ == '__main__':
    main()
