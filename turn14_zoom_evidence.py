"""Re-render the two zoom-evidence figures with BOTH the detailed annotations (numbers read from the arrays) and the
required title (which drive archive, the declared window 0.5 s / 20000 steps / dt 25 us, "manual research, NOT standing").
Numbers come from the stored report.json / time_series.npz; nothing is re-simulated.
"""

import hashlib
import json
import pathlib
import time

import numpy as np

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
ARTS = [(DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559',
         'the DECLARED 100-frame platform archive (20261015_135150)'),
        (DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164943-from-accepted-pose-archive',
         'an archive EXPORTED FROM THE ACCEPTED STATE ITSELF (20261015_164914, same pipeline/branch, only the initial pose changed)')]
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
HAVE_MIN, HAVE_MAX = 0.10, None
edits = []


def render(d, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = json.loads((d / 'report.json').read_text())
    res = r['result']
    ts = np.load(d / 'time_series.npz')
    dt_ms = 25e-6 * 1000.0
    t = np.asarray(ts['time_s']) * 1000.0
    rz = np.asarray(ts['root_z'])
    legs = np.asarray(ts['legs_in_contact'])
    nf = np.asarray(ts['n_nonfoot_contacts'])
    nff = np.asarray(ts['n_nonfoot_floor_contacts'])
    tilt = np.asarray(ts['tilt_deg'])
    en = np.asarray(ts['energy'])

    fig, ax = plt.subplots(2, 2, figsize=(15.0, 9.6))
    a = ax[0][0]
    n = min(1200, t.size)
    a.plot(t[:n], rz[:n], lw=1.7, color='#1f77b4')
    if res['first_nonfoot_contact_step'] is not None:
        s = res['first_nonfoot_contact_step']
        a.axvline(s * dt_ms, color='red', ls=':', lw=1.8)
        a.annotate('step %d = %.3f ms\nfirst NON-FOOT contact\n%s' % (s, s * dt_ms, ' | '.join(res['first_nonfoot_contact_geom_pair'] or [])),
                   (s * dt_ms, float(rz[s])), textcoords='offset points', xytext=(26, 34), fontsize=9,
                   arrowprops=dict(arrowstyle='->', lw=1.0), bbox=dict(boxstyle='round', fc='white', alpha=0.92))
    else:
        a.annotate('no non-foot contact in the whole window', (0.03, 0.85), xycoords='axes fraction', fontsize=9,
                   bbox=dict(boxstyle='round', fc='white', alpha=0.92))
    if res['first_legs_in_contact_below_6_step'] is not None:
        a.axvline(res['first_legs_in_contact_below_6_step'] * dt_ms, color='orange', ls=':', lw=1.4)
    a.set_xlim(0, n * dt_ms)
    a.set_xlabel('time [ms] (first %d ms)' % int(n * dt_ms)); a.set_ylabel('root z [mm]')
    a.set_title('zoom A: root z (init %.6f, min %.6f at step %s, final %.6f)'
                % (res['root_z_initial'], res['root_z_min'], res['root_z_min_step'], res['root_z_final']), fontsize=9)

    a = ax[0][1]
    a.plot(t, legs, lw=1.4, color='#2ca02c', label='legs with tarsus contact')
    a.plot(t, nf, lw=1.2, color='#d62728', label='non-foot contacts (any)')
    a.plot(t, nff, lw=1.2, color='#ff7f0e', ls='--', label='of which with the floor')
    if res['first_legs_in_contact_below_6_step'] is not None:
        a.axvline(res['first_legs_in_contact_below_6_step'] * dt_ms, color='orange', ls=':', lw=1.4)
        a.annotate('step %d = %.3f ms: legs < 6' % (res['first_legs_in_contact_below_6_step'], res['first_legs_in_contact_below_6_ms']),
                   (res['first_legs_in_contact_below_6_step'] * dt_ms, 6.0), textcoords='offset points', xytext=(18, -46),
                   fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.0), bbox=dict(boxstyle='round', fc='white', alpha=0.92))
    if res['first_step_with_zero_tarsus_contacts'] is not None:
        a.axvline(res['first_step_with_zero_tarsus_contacts'] * dt_ms, color='purple', ls=':', lw=1.4)
        a.annotate('step %d = %.3f ms: zero tarsus contacts'
                   % (res['first_step_with_zero_tarsus_contacts'], res['first_step_with_zero_tarsus_contacts_ms']),
                   (res['first_step_with_zero_tarsus_contacts'] * dt_ms, 24.0), textcoords='offset points', xytext=(-150, 18),
                   fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.0), bbox=dict(boxstyle='round', fc='white', alpha=0.92))
    else:
        a.annotate('no step with zero tarsus contacts in the whole window', (0.03, 0.42), xycoords='axes fraction', fontsize=9,
                   bbox=dict(boxstyle='round', fc='white', alpha=0.92))
    a.set_xlabel('time [ms]'); a.set_ylabel('count')
    a.set_title('zoom B: contacts (six-leg fraction %.6f = %d/%d)'
                % (res['six_legs_in_contact_fraction'], res['six_legs_in_contact_steps'], int(ts['time_s'].size)), fontsize=9)
    a.legend(fontsize=8)

    a = ax[1][0]
    m = min(4000, t.size)
    a.plot(t[:m], tilt[:m], lw=1.6, color='#9467bd')
    a.axhline(90.0, color='red', ls='--', lw=1.2, label='90 deg = body axis horizontal (never reached)')
    k = int(np.nanargmax(tilt[:m]))
    a.annotate('peak %.4f deg at step %d (whole window max %.4f deg)'
               % (float(tilt[:m].max()), k, res['tilt_deg_max']),
               (k * dt_ms, float(tilt[:m].max())), textcoords='offset points', xytext=(22, 4), fontsize=9,
               bbox=dict(boxstyle='round', fc='white', alpha=0.92))
    a.set_xlabel('time [ms] (first 100 ms)'); a.set_ylabel('tilt [deg]')
    a.set_title('zoom C: tilt (flipped %s)' % res['flipped'], fontsize=9)
    a.legend(fontsize=8)

    a = ax[1][1]
    a.plot(t, en, lw=1.6, color='#8c564b')
    a.annotate('energy %.6f -> %.6f\nmonotone decreasing %s; rising steps %d/%d'
               % (res['energy_initial'], res['energy_final'], res['energy_monotone_decreasing'],
                  res['energy_rising_steps'], res['energy_steps_compared']),
               (0.03, 0.06), xycoords='axes fraction', fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.92))
    a.set_xlabel('time [ms]'); a.set_ylabel('energy [model units]')
    a.set_title('zoom D: energy', fontsize=9)

    for x in ax.ravel():
        x.grid(alpha=0.3)
    fig.suptitle('zoom evidence, accepted declared-cone static equilibrium open loop under %s\n'
                 'DECLARED WINDOW: 0.5 s = 20000 steps at dt = 25 us, qpos/qvel/act written once, only ctrl in the loop\n'
                 'static necessary condition != maintainability; manual research, NOT standing / NOT walking; not a product'
                 % label, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = d / 'openloop-zoom-evidence.png'
    before = sha(out)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    edits.append(dict(file=str(out), before=before, after=sha(out),
                      reason='zoom evidence restored to carry every annotated number, together with the required title'))


def main():
    for d, label in ARTS:
        render(d, label)
    table = DESK / 'S2-equilibrium-hash-reanchor-20261015T165026' / 'hash_reanchor_table.json'
    doc = json.loads(table.read_text())
    doc.setdefault('this_pass_edits', []).extend(edits)
    doc['recorded_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    table.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    batch = DESK / 'S2-openloop-pair-batch-20261015T165026' / 'hash_reanchor_table.json'
    batch.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    for d in [t[0] for t in ARTS] + [table.parent, batch.parent]:
        files = sorted(x for x in d.iterdir() if x.is_file() and x.name != 'delivery-hashes.json')
        mf = d / 'delivery-hashes.json'
        old = json.loads(mf.read_text()) if mf.exists() else {}
        delivery = old.get('delivery_time_hashes', {k: v for k, v in old.items() if isinstance(v, str)})
        nowh = {x.name: sha(x) for x in files}
        for k, v in nowh.items():
            delivery.setdefault(k, v)
        mf.write_text(json.dumps(dict(format='two columns: delivery_time_hashes (preserved) + now_hashes (current)',
                                      delivery_time_hashes=delivery, now_hashes=nowh, recorded_at=time.strftime('%Y-%m-%d %H:%M:%S')),
                                  ensure_ascii=False, indent=1))
    for e in edits:
        print(pathlib.Path(e['file']).parent.name, e['before'][:8], '->', e['after'][:8])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
