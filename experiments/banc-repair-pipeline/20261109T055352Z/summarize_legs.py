"""Summarise the per-leg and sequential-curriculum runs."""
import json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
tags = sys.argv[1:] or ['lf96', 'legLM', 'legLH', 'legRF', 'legRM', 'legRH', 'seq']
for tag in tags:
    f = HERE / ('run_demo_%s.log' % tag)
    if not f.exists():
        print(tag, 'missing'); continue
    lines = f.read_text().splitlines()
    ev = [json.loads(l) for l in lines if l.startswith('{"EVAL"')]
    up = [json.loads(l) for l in lines if l.startswith('{"update"')]
    print('%-7s updates=%-3s evals=%d' % (tag, up[-1]['update'] if up else '?', len(ev)))
    for d in ev[-1:]:
        s = d['summary']
        print('   EVAL %2d -> %d/6 passing: %s' % (d['EVAL'], s['n_passed'],
                                                   ','.join(s['passed_legs']) or '-'))
        for L in ('LF', 'LM', 'LH', 'RF', 'RM', 'RH'):
            v = d['per_leg'][L]
            print('      %s lift %+.4f adv %.4f load %.3f padz %+.4f com %.3f  %s' % (
                L, v['max_lift'], v['best_advance'], v['final_load_over_body_weight'],
                v['final_padz'], v['com_min'],
                ''.join(k[0].upper() for k, x in v['criteria'].items() if x)))
