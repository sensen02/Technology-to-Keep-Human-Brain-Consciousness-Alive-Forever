"""Sweep the clock-free teacher's height threshold so the demonstration itself passes."""
from pathlib import Path
import json, sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import demo_state_teacher as S


def main():
    rows = {}
    for h in (0.03, 0.05, 0.08, 0.10, 0.14, 0.20):
        res = {}
        for leg in range(rc.NL):
            r = S.run(leg, height_thr=h)
            res[rc.LEG_ORDER[leg]] = dict(max_padz=round(r['max_padz'], 3),
                                          final_padz=round(r['final_padz'], 4),
                                          final_load=round(r['final_load'], 3),
                                          support_min=round(r['support_min'], 3),
                                          com_min=round(r['com_min'], 3),
                                          passed=r['passed'],
                                          crit=''.join(k[0].upper() for k, v in r['criteria'].items() if v))
        n = sum(v['passed'] for v in res.values())
        rows['height_thr=%.2f' % h] = dict(n_passed=n, per_leg=res)
        print('height_thr %.2f -> %d/6  %s' % (h, n, {L: v['crit'] for L, v in res.items()}),
              flush=True)
    best = max(rows.items(), key=lambda kv: kv[1]['n_passed'])
    out = dict(label='clock-free state teacher height-threshold sweep, no learning',
               best=best[0], best_n=best[1]['n_passed'], per_config=rows)
    (HERE / 'DEMO_STATE_SWEEP.json').write_text(json.dumps(out, indent=2) + '\n')
    print('best:', best[0], best[1]['n_passed'])
    print('passed legs:', [L for L, v in best[1]['per_leg'].items() if v['passed']])


if __name__ == '__main__':
    main()
