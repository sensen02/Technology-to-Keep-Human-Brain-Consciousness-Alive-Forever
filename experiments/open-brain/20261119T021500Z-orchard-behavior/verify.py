"""Summarize observed behavior and verify matched inputs and physical ledgers."""
import argparse
import json
from pathlib import Path

import numpy as np

CASES = ('FRESH','EMPTY_GROUND','GUIDED')
GROUND = {'ground_near','ground_far'}
LAUNCH_DELAYS = (13,13,14,14)


def check(root,reference):
    facts, traces, eyes = {}, {}, {}
    paths = {case:(reference if case == 'GUIDED' else root)/case for case in CASES}
    for case in CASES:
        p = paths[case]
        facts[case] = json.loads((p/'RESULT.json').read_text())
        traces[case] = np.load(p/'TRACE.npy')
        eyes[case] = [json.loads(line) for line in (p/'EYES.jsonl').read_text().splitlines()]
    initial = {case: np.load(paths[case]/'INITIAL.npy') for case in CASES}
    trace_a,trace_b = (traces[case] for case in CASES[:2])
    different = np.flatnonzero(np.any(trace_a[:,1:4]!=trace_b[:,1:4],axis=1))
    first_different = float(trace_a[different[0],0]) if len(different) else None
    contact = next((i for i,(a,b) in enumerate(zip(*(eyes[case] for case in CASES[:2]),strict=True))
                    if a['contact']!=b['contact']),None)
    prefix = len(eyes['FRESH']) if contact is None else contact
    sensory_match = eyes['FRESH'][:prefix] == eyes['EMPTY_GROUND'][:prefix]
    quality = dict(same_initial=all(np.array_equal(initial['FRESH'],q) for q in initial.values()),
                   stock_and_contact_prefix_match=sensory_match,
                   inventory_conservation=all(r['mass_error'] < 1e-11 for r in facts.values()),
                   no_external_forces=all(r['max_external'] == 0. for r in facts.values()),
                   sample_count=all(len(traces[case])==facts[case]['native_steps'] for case in CASES))
    results={}
    for case, fact in facts.items():
        with np.load(paths[case]/'NEURAL.npz') as n:
            emitted,arrived=n['emitted'],n['arrived']
            for i,delay in enumerate(LAUNCH_DELAYS):
                assert np.array_equal(np.flatnonzero(emitted[:,i])+delay,
                                      np.flatnonzero(arrived[:,i]))
            active = n['head'][:,2]>0.
            active_high_samples=int(active.sum())
            front_frames=int(np.count_nonzero(n['head'][:,4]))
        rows = eyes[case]
        high = [row['time_s'] for row in rows if row['height_probability']>.8]
        has_high = any(any(item['high'] for item in row['candidates']) for row in rows)
        targets = GROUND if case=='FRESH' else (set(fact['fruit_stable_s']) if case=='EMPTY_GROUND'
                                                   else {'branch_outer'})
        success=any(fact['intake'][name] >= .01 and
                    (case=='FRESH' or (fact['fruit_stable_s'][name]>=.07 and fact['first_air_s'] is not None))
                    for name in targets)
        trace=traces[case]
        pre = trace[:,7]==0
        results[case] = dict(behavior_pass=success, any_high_candidate=has_high,
                             first_high_selection_s=high[0] if high else None,
                             high_selected_frames=len(high), high_drive_frames=active_high_samples,
                             ground_drive_frames=front_frames,
                             eye_frames=len(rows), first_air_s=fact['first_air_s'],
                             first_intake=fact['first_intake'], intake=fact['intake'],
                             prelaunch_nonfoot_s=float(np.count_nonzero(pre&(trace[:,6]>0))*fact['native_dt_s']),
                             prelaunch_wing_error_deg=float(np.rad2deg(trace[pre,12].max())),
                             distance_to_branch_outer_mm=fact['min_xy_to_fruit_mm']['branch_outer'],
                             max_height_mm=fact['max_height_mm'], arrived=fact['arrived'])
    support=json.loads((reference/'SUPPORT/RESULT.json').read_text())
    s0=np.load(reference/'SUPPORT/orchard_neutral_INITIAL.npy')
    quality['initial_support_probe_match']=all(np.array_equal(np.load(f),s0)
                                               for f in (reference/'SUPPORT').glob('*_INITIAL.npy'))
    quality['flight_axon_delays']=True
    quality['wings_held_on_ground']=all(r['prelaunch_wing_error_deg']<3. for r in results.values())
    return dict(quality=quality,behavior=results,paired_control=dict(
        first_position_divergence_s=first_different,first_taste_difference_frame=contact,
        identical_spawn_pixels=all((root/CASES[0]/f'eye_start_{side}.png').read_bytes()==
                                   (root/CASES[1]/f'eye_start_{side}.png').read_bytes() for side in ('left','right'))),
        support=support)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--reference',type=Path,
                        help='Use this original folder for the guided and mechanical controls')
    args=parser.parse_args()
    root=args.root.resolve()
    result=check(root,(args.reference or root).resolve())
    (root/'VALIDATION.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'quality':result['quality'],'behavior':result['behavior']},indent=2))
