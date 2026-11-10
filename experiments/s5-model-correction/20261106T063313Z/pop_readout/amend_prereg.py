"""Declared amendment of the frozen design, made AFTER the offline fit and BEFORE any physics.

Two defects were found in the OFFLINE stage only (no release had been run):

1. alpha scale error.  The pre-registered raw alpha grid topped out at 1e5 while the ridge
   kernel's own scale is s = trace(X X^T)/n = 3.16e8 for the 12000-cell features.  Every
   grid point was therefore numerically inert (gamma = alpha/s <= 3e-4, i.e. no
   regularisation at all) and every CV curve was monotone up to the grid edge, so "alpha
   chosen on the fit window" was not actually being chosen.  Fix: parameterise
   alpha = gamma * s over the fixed grid common.GAMMAS.  This is a REPARAMETERISATION of
   the same ridge family - no model, feature or target changes.

2. P2 estimator degeneracy.  The pre-registered P2 tied its offset to the same shrunk
   weights (b2 = -W2 @ m_fit) while leaving the target uncentred, so its prediction
   averages to ~0 while the target averages to 2.43.  That objective cannot represent the
   mean torque at all: the fit-window CV curve was monotone toward total shrinkage
   (-57.1 at every grid point) and the held-out R^2 was -0.991.  Fix: P2 de-means each
   cell's rate by its fit-window mean AND supplies the per-joint fit-window mean torque as
   the one fixed offset (b2 = target_mean_fit - W2 @ m_fit), fitting on the fit-window
   centred target.  This is exactly the standard ridge-with-intercept shape, and it is the
   only form in which "remove the offset, then decode" is a well-posed question.  The
   degenerate form is retained as P2z and reported, never released.

   Consequence: the pre-registered "P1i" (free-intercept ridge) is now IDENTICAL to P2, so
   P1i is dropped as a separate entry to avoid double counting; P4 (100 ms trend removal)
   stays as an interface-only diagnostic.

Both amendments were decided from FIT-WINDOW evidence (the CV curves) plus the
mathematics.  No physical release had run.  The pre-amendment fit artifacts, including the
held-out numbers quoted above, are retained unchanged under PREAMENDMENT/ and are also
reported in the final report.  gamma is still chosen by blocked CV inside the fit window
only, so the held-out window still drives no choice.

Run once, before run.py.  Writes PREREGISTRATION.json, SOURCE_FREEZE.json and
CHANGE_LEDGER.json with old and new hashes for every file it changes.
"""
import json
from pathlib import Path

import common as C

AMENDMENTS = [
    dict(id='A1',
         target='fit_protocol.alpha_selection / readouts',
         old='alpha chosen on the fit window from the raw grid '
             '[1e-2 ... 1e5] (common.ALPHAS)',
         new='alpha = gamma * s with s = mean squared feature norm of the fit window '
             '(trace(X X^T)/n); gamma chosen on the fit window from the fixed grid '
             '[1e-6 ... 1e3] (common.GAMMAS). Both gamma and the raw alpha are reported.',
         reason='the raw grid was numerically inert at this feature scale (s = 3.16e8, so the '
                'largest pre-registered alpha was gamma = 3e-4): every CV curve was monotone to '
                'the grid edge and alpha was not in fact being chosen. A reparameterisation of '
                'the same ridge family, not a change of model.',
         evidence='PREAMENDMENT/READOUT_STATIC.json: every cv_curve monotone in alpha, chosen '
                  'alpha = 1e5 = grid maximum for all six fitted variants; '
                  'PREAMENDMENT/static_PREAMENDMENT.log',
         physics_run_before=False),
    dict(id='A2',
         target='readouts.P2',
         old='command = clip(W2 @ (r_12000 - m_fit), +/- limits) with b2 = -W2 @ m_fit tied to '
             'the same shrunk weights and the target left uncentred',
         new='command = clip(W2 @ (r_12000 - m_fit) + b2, +/- limits) with '
             'b2 = target_mean_fit - W2 @ m_fit, fitted against the fit-window centred target '
             '(the standard ridge-with-intercept shape). The degenerate tied-offset form is '
             'kept as P2z, interface-only, NOT released.',
         reason='the pre-registered form cannot represent the mean torque (prediction averages '
                'to ~0, target averages to 2.43), so it measured nothing about slow drift; it '
                'was a mis-specified estimator, not a result.',
         evidence='PREAMENDMENT/READOUT_STATIC.json: P2 cv_curve = -57.1 at every alpha '
                  '(monotone toward total shrinkage), r2_held = -0.991; the degeneracy is '
                  'visible in the FIT-window CV alone. Both the degenerate held-out number and '
                  'this amendment are reported.',
         physics_run_before=False),
    dict(id='A3',
         target='readouts.P1i',
         old='P1i = ridge over all 12000 with a free per-joint intercept, reported as an '
             'interface-only diagnostic',
         new='P1i removed: with the A2 offset supplied, P2 IS the free-intercept ridge, so '
             'keeping P1i would double count one readout',
         reason='algebraic identity; no measurement is lost',
         evidence='static.py: P2 fits the fit-window centred target on the de-meaned features '
                  'with b2 = target_mean_fit - W2 @ m_fit',
         physics_run_before=False),
    dict(id='A4',
         target='source',
         old='static.py / common.py as frozen by prereg.py',
         new='static.py and common.py corrected: (a) the interface statistics used the wrong '
             'window bound and raised ZeroDivisionError on the first run; (b) the scale-free '
             'gamma grid and the P2 offset above; (c) dead code removed from common.py',
         reason='a crashed stage and a mis-scaled grid cannot be carried into a heavy job',
         evidence='the ZeroDivisionError was on the FIRST static.py run and is described here; '
                  'the second (pre-amendment) run is archived in PREAMENDMENT/',
         physics_run_before=False),
]


def main():
    if (C.O / 'AMENDMENT_APPLIED.json').exists():
        raise SystemExit('amendment already applied; refusing to run twice')
    pre = C.js(C.O / 'PREREGISTRATION.json')
    freeze_before = C.js(C.O / 'SOURCE_FREEZE.json')
    led = C.js(C.O / 'CHANGE_LEDGER.json')
    old_prereg = C.sha(C.O / 'PREREGISTRATION.json')
    old_src = {str(p): C.sha(p) for p in sorted(C.SRC.glob('*.py'))}

    # ---- rewrite the two amended design blocks
    cfg = json.loads((C.O / 'PREREGISTRATION.json').read_text())
    cfg['readouts']['P2'] = (
        'the same 12000 features with every cell\'s FIT-WINDOW mean removed before the ridge '
        '(explicit de-meaning), and with the per-joint fit-window mean torque supplied as THE one '
        'fixed offset: command = clip(W2 @ (r_12000 - m_fit) + b2, +/- limits), '
        'b2 = target_mean_fit - W2 @ m_fit. This is the standard ridge-with-intercept shape and '
        'is the declared test of whether slow drift dominates: if P2 reconstructs the torque as '
        'well as P1, the fast fluctuation carries the information; if it collapses, the absolute '
        'rate level does.')
    cfg['readouts']['P2z'] = (
        'interface-only diagnostic, NOT released: the same de-meaned features with the offset '
        'tied to the same shrunk weights (b2 = -W2 @ m_fit) and the target uncentred. Its '
        'prediction cannot represent the mean torque, so it shows why an offset must be '
        'supplied; it is reported, never put on the body.')
    cfg.pop('why_P1_has_no_constant_term', None)
    cfg['P1_has_no_constant_term_because'] = (
        'with a free per-joint intercept the readout would be P2; P1 therefore keeps the same '
        'structure as the current 204-cell readout (no constant term), so that every part of the '
        'torque, including its DC level, must come from the neural rate levels themselves.')
    cfg['fit_protocol']['alpha_selection'] = (
        'alpha is parameterised scale-free: alpha = gamma * s, s = mean squared feature norm of '
        'the fit window (trace(X X^T)/n, the mean eigenvalue scale of the ridge kernel). gamma is '
        'chosen INSIDE the fit window only by %d contiguous blocked folds: gamma = argmax of the '
        'mean held-fold R^2 over the fixed grid %r. The held-out 0.8-1.4 s window never enters '
        'the choice. Both gamma and the raw alpha are reported, and the whole CV curve.'
        % (C.CV_FOLDS, C.GAMMAS))
    cfg['fit_protocol']['method'] = (
        'ridge regression, dual form (n=600 fit intervals << d=12000 features); the only feature '
        'transform is the declared per-cell fit-window de-meaning of P2/P3, plus the '
        'interface-only 100 ms trend removal of P4')
    cfg['amendments'] = AMENDMENTS
    cfg['amendment_note'] = ('These amendments were applied AFTER the offline fit and BEFORE any '
                             'physical release. Every pre-amendment artifact is retained under '
                             'PREAMENDMENT/. No reading-rule threshold, no phase list, no '
                             'timestep list and no release grid changed.')
    C.dump(C.O / 'PREREGISTRATION.json', cfg)
    new_prereg = C.sha(C.O / 'PREREGISTRATION.json')

    # ---- refresh the source freeze against the amended source
    now_src = {str(p): C.sha(p) for p in sorted(C.SRC.glob('*.py'))}
    freeze_after = C.js(C.O / 'SOURCE_FREEZE.json')
    freeze_after['files'] = now_src
    freeze_after['amended_after'] = 'A1/A2/A3/A4 (see PREREGISTRATION.json amendments); no '
    freeze_after['amended_note'] = ('refreshed by amend_prereg.py before the heavy job; the '
                                    'pre-amendment hashes are in CHANGE_LEDGER.tool_fixes')
    old_freeze = C.sha(C.O / 'SOURCE_FREEZE.json')
    C.dump(C.O / 'SOURCE_FREEZE.json', freeze_after)

    # ---- ledger
    fixes = led.setdefault('tool_fixes', [])
    for p in sorted(now_src):
        if old_src.get(p) != now_src[p]:
            fixes.append(dict(path=p, old_sha256=old_src.get(p), new_sha256=now_src[p],
                              reason='declared amendment A1/A2/A3/A4 of the offline stage; no '
                                     'physics had been run',
                              timestamp='before_heavy_job'))
    fixes.append(dict(path=str(C.O / 'PREREGISTRATION.json'), old_sha256=old_prereg,
                      new_sha256=new_prereg,
                      reason='amendments A1 (scale-free alpha) and A2 (well-posed P2 offset) '
                             'recorded in-file', timestamp='before_heavy_job'))
    fixes.append(dict(path=str(C.O / 'SOURCE_FREEZE.json'), old_sha256=old_freeze,
                      new_sha256=C.sha(C.O / 'SOURCE_FREEZE.json'),
                      reason='refreshed to the amended source hashes', timestamp='before_heavy_job'))
    led['amendments'] = AMENDMENTS
    led['preamendment_evidence'] = {
        'dir': str(C.O / 'PREAMENDMENT'),
        'files': sorted(p.name for p in (C.O / 'PREAMENDMENT').iterdir()),
        'note': 'the pre-amendment fit, including the held-out numbers quoted in the amendment '
                'reasons, is kept unchanged; nothing was overwritten'}
    C.dump(C.O / 'CHANGE_LEDGER.json', led)

    C.dump(C.O / 'AMENDMENT_APPLIED.json', dict(
        applied=True, when='after the offline fit, before any physical release',
        prereg_old_sha256=old_prereg, prereg_new_sha256=new_prereg,
        source_old=old_src, source_new=now_src,
        amendments=[a['id'] for a in AMENDMENTS],
        physics_releases_run_before=0,
        heldout_used_for_any_choice=False,
        note='Reading rule, phases, timesteps and the release grid are untouched.'))
    print('AMENDED', [a['id'] for a in AMENDMENTS], flush=True)


if __name__ == '__main__':
    main()
