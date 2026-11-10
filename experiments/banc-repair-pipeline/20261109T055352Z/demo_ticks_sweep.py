"""How long should the demonstration lift before it starts pressing down?

Teacher self-check only: the same state dependent press as demo_press_probe.py, with the lift
duration swept. A configuration is useful only if the demonstration itself satisfies the task.
"""
from pathlib import Path
import json, sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import demo_press_probe as P


def main():
    rows = {}
    for lt in (90, 110, 130, 150, 170, 200):
        res = {}
        for leg in range(rc.NL):
            r = P.run(leg, lift_ticks=lt)
            res[rc.LEG_ORDER[leg]] = dict(
                final_padz=round(r['final_padz'], 4), final_load=round(r['final_load'], 3),
                max_padz=round(r['max_padz'], 3), support_min=round(r['support_min'], 3),
                com_min=round(r['com_min'], 3), passed=r['passed'],
                crit=''.join(k[0].upper() for k, v in r['criteria'].items() if v))
        n = sum(v['passed'] for v in res.values())
        rows['lift_ticks=%d' % lt] = dict(n_passed=n, per_leg=res)
        print('lift_ticks %3d -> %d/6  %s' % (lt, n, {L: v['crit'] for L, v in res.items()}),
              flush=True)
    best = max(rows.items(), key=lambda kv: kv[1]['n_passed'])
    out = dict(label='demo lift-duration sweep with state dependent press, no learning',
               best=best[0], best_n=best[1]['n_passed'], per_config=rows)
    (HERE / 'DEMO_TICKS_SWEEP.json').write_text(json.dumps(out, indent=2) + '\n')
    print('best:', best[0], best[1]['n_passed'])


if __name__ == '__main__':
    main()
