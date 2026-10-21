"""Record the remaining corrections into v3 (my own, unverified batch) and re-freeze it:
  * the true-protocol ms table from the corrected batch;
  * the one-step ms error in the F-A sentence (step 3055 = 76.375 ms, not 76.400 ms);
  * the INVALID -> WITHDRAWN change of the integrity record, which was ordered in turn 18 (explaining the verifier's M5).
"""

import hashlib
import json
import pathlib
import time

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
V3 = DESK / 'S2-equilibrium-A-corrected-reissued-v3-20261015T170319'
CORR = DESK / 'S2-accepted-equilibrium-CORRECTED-PROTOCOL-25us-20261015T171055'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')

doc = dict(
    status='PASS_MS_LABEL_CORRECTION',
    wrong_labels_in_this_batch=('every millisecond label in this batch was inherited from runs that silently used the XML timestep of 100 us; '
                               'the TRUE values are 4x larger (see the corrected batch)'),
    true_protocol_ms=dict(
        run='declared 100-frame archive, m.opt.timestep = 25e-6 explicitly set, 20000 steps = 0.5 s, ctrl every 200 steps = 5 ms',
        first_non_foot_contact='step 2811 = 70.275 ms (pair A6_geom | floor)',
        legs_in_contact_below_6='step 538 = 13.45 ms',
        zero_tarsus_contacts='step 5082 = 127.05 ms',
        first_max_u_diamond_ge_1='step 2 = 0.05 ms (pair LHTarsus2_geom | floor)',
        root_z='initial 1.9808396350942905, min 0.5803232830656021 at step 3814, final 0.7954749634889989',
        six_leg_steps='901 / 20001 = 0.04504774761261937',
        energy='44.50790486183263 -> 14.530504940083375, non-monotone, 9435 rising steps of 20000',
        tilt='max 52.790359184586556 deg, final 21.46477733769685 deg, never inverted',
        act='never exceeds 1 (max 0.9999999999999997) — the out-of-domain warning of the 100 us runs does not apply at 25 us',
        per_step_max_qacc='step 0 2.5865734771464476e-11, final 1505.3124904361139, window max 24303.06890766346 at step 3011, '
                          'first above 1e-3 and 1.0 mm/s^2 at step 2 = 0.05 ms',
        comparison_with_pose_A='A (s=0.95941) first non-foot contact step 2398 = 59.950 ms under the same 25 us protocol; the accepted '
                              'equilibrium is therefore LATER by about 10.3 ms, not earlier'),
    old_100us_ms_are_4x_too_small=True,
    f_a_sentence_ms_fix=dict(step=3055, wrong_ms=76.400, correct_ms=76.375,
                             note='3055 x 25 us = 76.375 ms; 76.400 ms corresponds to step 3056. The step number and the geom identity were '
                                  'correct; only the ms conversion was off by one step. Corrected here and to be used in any later text.'),
    integrity_record_status_change=dict(observed_by='verification S2-openloop-frozen-check-20261015T235925Z (finding M5)',
                                        what='the file went from INVALID (as read at 16:59) to WITHDRAWN (17:01:26) after that verification froze',
                                        why='the owner ordered exactly that in turn 18: either rewrite it factually or mark it withdrawn and state the '
                                            'reason in the v1 README; WITHDRAWN was chosen and the README section 17 was added. The factual event '
                                            'record (before/after sha256, 16:14:08, replaced file, bytes gone) is inside the same file.'),
    corrected_batch=str(CORR), corrected_batch_report_sha256=sha(CORR / 'report.json'), recorded_at=now())
(V3 / 'ms_label_correction.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1))
files = sorted(p for p in V3.iterdir() if p.is_file() and p.name not in ('delivery-hashes.json', 'FROZEN_DELIVERY_HASHES.json'))
(V3 / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))
(V3 / 'FROZEN_DELIVERY_HASHES.json').write_text(json.dumps(dict(status='FROZEN_AT_WRITE', frozen_at=now(),
                                                                files={p.name: sha(p) for p in files}), ensure_ascii=False, indent=1))
print('v3 files:', len(files))
print('corrected batch report sha:', sha(CORR / 'report.json'))
