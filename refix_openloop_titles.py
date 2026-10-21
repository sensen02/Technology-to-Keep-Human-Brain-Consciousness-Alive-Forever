"""Re-render the two figures of both open-loop artifacts so the title names WHICH drive archive was used (the declared
platform archive vs the archive exported from the accepted state), then append the before/after hashes to the reanchor table.
Numbers are taken from the stored report.json / time_series.npz; nothing is re-simulated.
"""

import hashlib
import json
import pathlib
import time

import numpy as np

DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
ART_OLD = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'
ART_NEW = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164943-from-accepted-pose-archive'
REANCHOR = DESK / 'S2-equilibrium-hash-reanchor-20261015T165026'
BATCH = DESK / 'S2-openloop-pair-batch-20261015T165026'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
edits = []


def render(d, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = json.loads((d / 'report.json').read_text())
    res = r['result']
    ts = np.load(d / 'time_series.npz')
    t = np.asarray(ts['time_s']) * 1000.0
    rz = np.asarray(ts['root_z'])
    legs = np.asarray(ts['legs_in_contact'])
    nf = np.asarray(ts['n_nonfoot_contacts'])
    nff = np.asarray(ts['n_nonfoot_floor_contacts'])
    tilt = np.asarray(ts['tilt_deg'])
    en = np.asarray(ts['energy'])
    dt_ms = 25e-6 * 1000.0
    drive = r['inputs']['drive']
    drive_short = drive.split('/')[-2] + '/' + drive.split('/')[-1]

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    ax = axes[0][0]
    ax.plot(t, rz, lw=1.2, color='#1f77b4')
    ax.axhline(rz[0], color='gray', ls='--', lw=1.0, label='initial root z')
    if res['first_nonfoot_contact_step'] is not None:
        s = res['first_nonfoot_contact_step']
        ax.axvline(s * dt_ms, color='red', ls=':', lw=1.6, label='first non-foot contact (step %d = %.3f ms)' % (s, s * dt_ms))
    if res['first_legs_in_contact_below_6_step'] is not None:
        ax.axvline(res['first_legs_in_contact_below_6_step'] * dt_ms, color='orange', ls=':', lw=1.4,
                   label='first <6 legs in contact (step %d)' % res['first_legs_in_contact_below_6_step'])
    ax.set_xlabel('time [ms]'); ax.set_ylabel('root z [mm]')
    ax.set_title('root z (min %.6f at step %s, final %.6f)' % (res['root_z_min'], res['root_z_min_step'], res['root_z_final']),
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(t, legs, lw=1.2, color='#2ca02c', label='legs with tarsus contact')
    ax.plot(t, nf, lw=1.0, color='#d62728', label='non-foot contacts (any)')
    ax.plot(t, nff, lw=1.0, color='#ff7f0e', ls='--', label='of which with the floor')
    ax.set_xlabel('time [ms]'); ax.set_ylabel('count')
    ax.set_title('contact bookkeeping (all tarsus lost at step %s; six-leg fraction %.6f)'
                 % (res['first_step_with_zero_tarsus_contacts'], res['six_legs_in_contact_fraction']), fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot(t, tilt, lw=1.2, color='#9467bd')
    ax.axhline(90.0, color='red', ls='--', lw=1.2, label='90 deg = body axis horizontal (never reached)')
    ax.set_xlabel('time [ms]'); ax.set_ylabel('tilt [deg]')
    ax.set_title('tilt (max %.4f deg; flipped %s)' % (res['tilt_deg_max'], res['flipped']), fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.plot(t, en, lw=1.2, color='#8c564b')
    ax.set_xlabel('time [ms]'); ax.set_ylabel('energy [model units]')
    ax.set_title('energy (monotone decreasing %s; rising steps %d/%d)'
                 % (res['energy_monotone_decreasing'], res['energy_rising_steps'], res['energy_steps_compared']), fontsize=10)
    ax.grid(alpha=0.3)
    fig.suptitle('open-loop dynamics of the accepted declared-cone static equilibrium under %s\n'
                 'drive archive %s (sha256 %s)   |   25 us, 20000 steps = 0.5 s, state written once, only ctrl in the loop\n'
                 'static necessary condition != maintainability; no standing claim; manual research implementation, not a product'
                 % (label, drive_short, drive_short and r['inputs']['drive_sha256'][:16]), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = d / 'accepted-equilibrium-openloop.png'
    before = sha(out)
    fig.savefig(out, dpi=135)
    plt.close(fig)
    edits.append(dict(file=str(out), before=before, after=sha(out),
                      reason='title now names WHICH drive archive was used (the run itself and every number are unchanged)'))


def main():
    render(ART_OLD, 'the DECLARED platform archive exported on 2026-10-15 13:51:50 (initial pose = projected historical pose)')
    render(ART_NEW, 'an archive EXPORTED FROM THE ACCEPTED STATE ITSELF (same pipeline, same branch/policy, only the initial pose changed)')
    p = REANCHOR / 'hash_reanchor_table.json'
    doc = json.loads(p.read_text())
    doc.setdefault('this_pass_edits', []).extend(edits)
    doc['recorded_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    (BATCH / 'hash_reanchor_table.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    for d in (ART_OLD, ART_NEW, REANCHOR, BATCH):
        files = sorted(x for x in d.iterdir() if x.is_file() and x.name != 'delivery-hashes.json')
        (d / 'delivery-hashes.json').write_text(json.dumps({x.name: sha(x) for x in files}, ensure_ascii=False, indent=1))
    print('figures re-rendered:', len(edits))
    for e in edits:
        print('  ', e['file'].split('/')[-2], e['before'][:8], '->', e['after'][:8])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
