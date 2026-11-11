"""Concise matched actual deceleration and front cycle event evidence."""
import json
from api import HERE,CONDITIONS

def main():
    reports={c['name']:json.loads((HERE/c['name']/'RESULT.json').read_text()) for c in CONDITIONS}
    control=reports['straight_control_100'];summary={}
    for name,r in reports.items():
        speed=r['forward_signed_native_per_s'];base=control['forward_signed_native_per_s']
        summary[name]=dict(is_control=r['is_matched_control'],all_original_five=r['original_all5_gates'],all_functional_gates=r['all_functional_gates'],
            forward=speed,forward_fraction_of_control=speed/base,forward_reduction_fraction=1-speed/base,
            matched_control_forward=base,maximum_candidate_forward=.95*base,slip=r['slip_ratio_max'],per_leg_slip=r['per_leg_slip_ratio'],
            support=r['foot_support_mean'],com_min=r['com_height_min'],nonfoot=r['nonfoot_force_max_native'],cycles=r['cycles']['complete_per_leg'],
            front_events={leg:[e for e in r['cycles']['events'] if e['leg']==leg] for leg in ('LF','RF')},
            maximum_completed_front_event_age_s={leg:max((e['duration'] for e in r['cycles']['events'] if e['leg']==leg and e['type']=='complete'),default=None) for leg in ('LF','RF')},
            wall_s=r['wall_s'])
    out=dict(cases=summary,scope='Matchedactualsame9.6window teacherfunctionaldeceldiagnostic; control is baseline not deceleration. Alloriginalgates and4s cycle criteria unchanged, no neuralautonomy/training/food claim.')
    (HERE/'CASE_SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
