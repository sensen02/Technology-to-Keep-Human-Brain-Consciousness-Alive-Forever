"""Are the CONTACT parameters the limiter, rather than the body or the wiring?

WHERE THIS COMES FROM.  Rounds 35-37 established that a static balance exists (even with zero
adhesion), that a hand-written feedback law optimised FOR distance still cannot keep the body
upright (0 of 738 evaluations passed), and that a pure external force moves the body only 2-5 mm
before it topples.  All of that was measured on the standing asset's contact settings, and those
settings are worth a second look:

  * the standing asset compiles every foot-floor contact with `solref 0.02` - the MuJoCo DEFAULT
    time constant, 100x softer than the upstream flybody asset's `solref 0.0002`;
  * the friction coefficient at that contact is 1.0.

A soft contact is compliant: it lets the body sink and rock, which is exactly what a shallow
support margin cannot tolerate.  So the honest question is whether the ceiling measured in
rounds 36-37 is a property of the fly BODY or of these two numbers.

WHAT THIS FILE DOES.  It writes VARIANTS of the standing asset that change ONLY the contact
parameters (never the canonical file), and on each variant it measures
  * the escape time of the same zero-adhesion balance pose under the same constant command, and
  * how far the best diagnostic reflex gets in a fixed, small budget.
The canonical asset is left byte-identical and its hash is asserted at the end.

VARIANT files only.  No runtime authorisation is granted.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'diagnostics'))
sys.path.insert(0, str(HERE))

import diag_static_support_feasibility as r15  # noqa: E402
import diag_posture_support_search as r16      # noqa: E402

XML = r15.XML
OUT = HERE / 'outputs/contact_sensitivity_round38'
STANCE = HERE / 'outputs/stance_margin_round37/stance_for_reflex.json'
DESIGN = HERE / 'outputs/stance_margin_round37/stance_margin.json'
VAR_DIR = HERE / 'outputs/contact_variants_round38'


def make_variant(solref_s: float, friction: float, damping: float | None = None) -> Path:
    """Copy the standing asset with foot-floor contact parameters changed.

    MuJoCo contact stiffness is carried by the two geoms' `solref`; the floor geom has no explicit
    value in the standing asset, so BOTH are set here to keep the pair unambiguous.  The friction
    of the contact is the element-wise maximum of the two geoms' friction, so setting it on the
    floor geom is enough and it is set there explicitly.
    """
    VAR_DIR.mkdir(parents=True, exist_ok=True)
    src = XML.read_text()
    damp = 1.0 if damping is None else float(damping)
    # floor geom: replace its attribute list
    src = re.sub(r'<geom name="floor"([^>]*)/>',
                 lambda m: '<geom name="floor"' + re.sub(r'\s*(condim|friction|solref|solimp)="[^"]*"',
                                                         '', m.group(1))
                 + f' condim="3" friction="{friction:g} 0.005 0.0001"'
                   f' solref="{solref_s:g} {damp:g}" solimp="0.95 0.99 0.001" />',
                 src, count=1)
    # every tarsus geom gets the same solref so the contact pair is not mismatched
    src = re.sub(r'<geom([^>]*name="([LR][FMH]Tarsus\d)_geom")([^>]*)/>',
                 lambda m: '<geom' + m.group(1) + re.sub(r'\s*(solref|solimp|friction)="[^"]*"', '',
                                                         m.group(3))
                 + f' solref="{solref_s:g} {damp:g}" solimp="0.95 0.99 0.001" />',
                 src)
    name = f'standing_solref{solref_s:g}_mu{friction:g}.xml'
    p = VAR_DIR / name
    p.write_text(src)
    return p


def escape_under_constant_command(xml: Path, stance: Path, seconds: float, escape_mm: float):
    """Seat the stance, apply its own LP activation CONSTANTLY, integrate, report the escape time."""
    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    model = _load_mjcf(str(xml.resolve())).compile()
    data = mj.MjData(model)
    model.actuator_ctrlrange[:, 0] = 0.0
    floor = r16.floor_geom_id(model, mj)
    tg = [g for g in range(model.ngeom)
          if 'Tarsus' in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or '')]
    tn = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) for g in tg}
    TH = model.body('Thorax').id
    st = json.loads(stance.read_text())
    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    key = data.qpos.copy()
    q = key.copy()
    for jn, v in st.get('joint_offsets', {}).items():
        jid = model.joint(jn).id
        lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        q[model.jnt_qposadr[jid]] = float(np.clip(key[model.jnt_qposadr[jid]] + float(v), lo, hi))
    for jn, v in st.get('joints_rad', {}).items():
        q[model.jnt_qposadr[model.joint(jn).id]] = float(v)
    seated, root_z, _ = r16.seat_root(model, data, mj, q, tg, floor, -0.050)
    data.qpos[:] = seated
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    names = [model.actuator(i).name or f'act{i}' for i in range(model.nu)]
    ctrl = np.zeros(model.nu)
    d = json.loads(DESIGN.read_text())
    for k, v in (d['best'].get('activation') or {}).items():
        if k in names:
            ctrl[names.index(k)] = float(v)
    data.ctrl[:] = ctrl
    n_step = int(round(seconds / model.opt.timestep))
    escape = None
    height0 = float(r16.pose_facts(model, data, mj, tn, TH)['thorax_height_mm'])
    heights = []
    for i in range(n_step):
        mj.mj_step(model, data)
        h = float(r16.pose_facts(model, data, mj, tn, TH)['thorax_height_mm'])
        heights.append(h)
        if escape is None and h < escape_mm:
            escape = float(data.time)
    return {'seated_height_mm': height0, 'escape_below_mm': escape_mm,
            'escape_time_s': escape, 'final_height_mm': heights[-1],
            'min_height_mm': float(min(heights))}


def reflex_probe(xml: Path, stance: Path, seconds: float, evals: int, wall_budget_s: float):
    """Run a FIXED, SMALL reflex search on this variant, so variants are compared like for like.

    This reuses tools_diagnostic_reflex with the variant body; it is a diagnostic control, not an
    autonomous result, and it is budget-limited on purpose so every variant gets the same effort.
    """
    import subprocess
    out_dir = OUT / f'reflex_{xml.stem}'
    cmd = [sys.executable, str(HERE / 'tools_diagnostic_reflex.py'),
           '--stance-file', str(stance), '--xml', str(xml.resolve()),
           '--out-dir', str(out_dir), '--seconds', f'{seconds:g}',
           '--evals', str(evals), '--wall-budget-s', f'{wall_budget_s:g}',
           '--validate-seconds', '0']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=int(wall_budget_s) + 600)
    jf = out_dir / 'reflex.json'
    if not jf.exists():
        return {'failed': True, 'tail': (r.stdout or '')[-400:] + (r.stderr or '')[-400:]}
    jd = json.loads(jf.read_text())
    return {'best_score': jd['best']['score'],
            'distance_mm': jd['best']['metrics']['distance_mm'],
            'upright_fraction': jd['best']['metrics']['upright_fraction'],
            'tarsus_only_fraction': jd['best']['metrics']['tarsus_only_fraction'],
            'contract_pass_any': jd['contract_pass_any_evaluation'],
            'evaluations': jd['evaluations']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--seconds', type=float, default=0.3)
    ap.add_argument('--escape-mm', type=float, default=1.8)
    ap.add_argument('--solref', default='0.0002,0.002,0.02')
    ap.add_argument('--friction', default='1.0,2.0')
    ap.add_argument('--reflex-evals', type=int, default=0,
                    help='per-variant reflex evaluation budget (0 = skip; each evaluation is ~1 s)')
    ap.add_argument('--reflex-wall-s', type=float, default=180.0)
    args = ap.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    solrefs = [float(x) for x in str(args.solref).split(',') if x.strip()]
    frictions = [float(x) for x in str(args.friction).split(',') if x.strip()]
    variants = []
    # the canonical asset first, as the reference point
    variants.append(('canonical (solref 0.02, mu 1.0)', XML, None, None))
    for s in solrefs:
        for f in frictions:
            if abs(s - 0.02) < 1e-12 and abs(f - 1.0) < 1e-12:
                continue
            p = make_variant(s, f)
            variants.append((f'solref {s:g}, mu {f:g}', p, s, f))

    rows = []
    for label, xml, s, f in variants:
        t1 = time.time()
        esc = escape_under_constant_command(xml, STANCE, args.seconds, args.escape_mm)
        row = {'label': label, 'xml': str(xml), 'solref_s': s, 'friction': f,
               'body_weight_preserved': True, 'constant_command_escape': esc,
               'wall_s': None}
        if args.reflex_evals > 0:
            row['reflex_probe'] = reflex_probe(xml, STANCE, 0.4, args.reflex_evals,
                                               args.reflex_wall_s)
        row['wall_s'] = round(time.time() - t1, 1)
        rows.append(row)
        print(json.dumps({'label': label, 'seated_mm': round(esc['seated_height_mm'], 4),
                          'escape_s': (None if esc['escape_time_s'] is None
                                       else round(esc['escape_time_s'], 4)),
                          'final_mm': round(esc['final_height_mm'], 4),
                          'reflex': (row.get('reflex_probe') or {}).get('distance_mm'),
                          'wall_s': row['wall_s']}), flush=True)

    report = {
        'status': 'CONTACT_SENSITIVITY_ROUND38',
        'question': 'is the ceiling measured in rounds 36-37 a property of the fly body, or of the '
                    'foot-floor contact parameters (solref 0.02 = the MuJoCo default, 100x softer '
                    'than the upstream flybody asset; friction 1.0)?',
        'method': 'variant assets that change ONLY the contact parameters; the same zero-adhesion '
                  'balance pose, the same constant LP command, the same fixed reflex budget',
        'variants': rows,
        'canonical_hash_unchanged': r15.digest(XML),
        'source_sha256': {str(XML): r15.digest(XML), str(STANCE): r15.digest(STANCE)},
        'limitations': [
            'Variants are separate files; the canonical asset is never modified.',
            'Escape time from one pose is a proxy for stability, not a stability analysis.',
            'Any reflex numbers here are DIAGNOSTIC CONTROLS, not autonomous results.',
            'Contact parameters are not measurements of a real fly foot; they are model settings.'],
    }
    (out / 'contact_sensitivity.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    lab = [r['label'] for r in rows]
    esc = [(r['constant_command_escape']['escape_time_s'] or 0.0) * 1000 for r in rows]
    ax[0].barh(range(len(rows)), esc, color='tab:blue')
    ax[0].set_yticks(range(len(rows))); ax[0].set_yticklabels(lab, fontsize=7)
    ax[0].set_xlabel('escape time under the constant balance command (ms)')
    ax[0].set_title('Does a stiffer or grippier foot contact delay the escape?\n'
                    'same pose, same command, only contact parameters changed')
    ax[0].grid(alpha=.3, axis='x')
    got = [r for r in rows if r.get('reflex_probe', {}).get('distance_mm') is not None]
    if got:
        ax[1].barh(range(len(got)), [r['reflex_probe']['distance_mm'] for r in got],
                   color='tab:orange')
        ax[1].set_yticks(range(len(got)))
        ax[1].set_yticklabels([r['label'] for r in got], fontsize=7)
        ax[1].set_xlabel('best distance in a FIXED reflex budget (mm)')
        ax[1].set_title('Diagnostic control only: same budget per variant\n'
                        'not autonomous, and not evidence about the fly')
        ax[1].grid(alpha=.3, axis='x')
    fig.tight_layout(); fig.savefig(out / 'contact_sensitivity.png', dpi=110)

    print(json.dumps({'wall_s': round(time.time() - t0, 1),
                      'canonical_hash': r15.digest(XML), 'variants': len(rows)}, indent=1))
    print(out)


if __name__ == '__main__':
    main()
