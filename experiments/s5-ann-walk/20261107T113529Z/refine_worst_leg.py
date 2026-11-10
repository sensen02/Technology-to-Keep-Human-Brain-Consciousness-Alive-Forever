"""One preregistered local physical refinement, prioritizing the worst foot.
Search1mean objective improved fourlegs, but bothhindlegs stillfail. This second
bounded stage uses one measured center; no inferred coordinate gains are combined.
"""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T112003Z/optimize_stance.py')
spec=importlib.util.spec_from_file_location('bounded_original_physical_search',SOURCE)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T113529Z')
PREVIOUS=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T112003Z')
s.OUT=OUT
s.PARAM.update(parameter_bounds_rad=[-.03,.03],tested_offset_rad=.01,maximum_trials=13)
original_score=s.score_trial


def score_trial(data,reason):
    r=original_score(data,reason)
    if 'slip_to_progress_ratios' not in r:return r
    ratios=np.asarray(r['slip_to_progress_ratios'])
    penalty=r['objective']-float(ratios.mean())
    r['prior_mean_objective']=r['objective']
    r['objective']=float(ratios.max()+.1*ratios.mean()+penalty)
    r['worst_leg_ratio']=float(ratios.max())
    return r

s.score_trial=score_trial


def candidates(center):
    result=[('measured_center',center.copy())]
    for i,name in enumerate(s.JOINTS):
        for value,label in [(-.01,'minus'),(.01,'plus')]:
            x=center.copy();x[i]+=value
            assert abs(x).max()<=.03+1e-15
            result.append((name+'_'+label,x))
    return result


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if not a.run:ap.error('parent must launch --run after source review')
    record=json.loads((PREVIOUS/'BEST_PARAMETERS.json').read_text())
    center=np.asarray(record['selected']['parameters_rad'])
    assert np.array_equal(center,np.array([0.,.02,0.,0.,0.,0.]))
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=OUT/'PREREGISTRATION.json'
    if protocol.exists():raise FileExistsError('preserve prior local refinement')
    hashes={str(p):s.n.sha(p) for p in [Path(__file__),SOURCE,PREVIOUS/'BEST_PARAMETERS.json',
        PREVIOUS/'AUDIT_SEARCH_RESULTS.json',PREVIOUS/'selected/teacher_selected_4s/VERIFICATION.json',s.GAIT]}
    s.n.dump(protocol,dict(importance='核心必需',stage='S5 second bounded physical coordinate refinement, worst-foot objective',
        previous=record['selected'],previous_strict_result='fullD/steps/headingpass, materialratios [.3296,.2271,.7220,.2947,.4151,.7395]; bothhindlegsfailed. No finalwalkingclaim.',
        budget=dict(new_trials=13,prior_search_trials=13,aggregate_declared_trials=26,max_steps_new=3120000,
                    seconds_per_trial=3.,scored_seconds=2.),
        center=center,candidates=[dict(tag=t,parameters=x) for t,x in candidates(center)],
        bounds=[-.03,.03],coordinate_delta=.01,joints=s.JOINTS,
        selection='min(max6slipratio +.1mean6slipratio +UNCHANGEDcontact/fall/no-progress/height/lateral/yawpenalty); all13actualtrials, no unmeasuredcoordinatecombination',
        rationale='Whole-legacceptancecannotbemetbyimprovingmeanonly; makeworstleg explicit. ThischangesSEARCHobjective notphysicalacceptance.',
        unchanged='same originalfullbody71mesh49/48/42neq0/free root/gravity/contact/friction/nativecaps,PD128,30rad/s,12.5usnativeevery-stepPD+forward,referencegeometry/phase',
        diagnostic='5mscontact/100msmeanprogress searchmeasureonly; strictnative4s+independentreplay+eachlegslip<=.5+ANN>=10s requiredseparately',
        source_hashes=hashes,units='native; absoluteSIforce/originalbrainnecessity/biologicalvalidity NOT_ESTABLISHED'))
    search=s.Search();results=[]
    for tag,parameters in candidates(center):
        results.append(search.trial(tag,parameters))
        s.n.dump(OUT/'SEARCH_RESULTS.json',dict(results=results,trials_started=search.trials_started,
            actual_physics_steps=search.physics_steps,maximum_trials=13))
    valid=[r for r in results if not r['score']['failed'] and r['score'].get('completed_full_duration') and r['finite'] and not r['warnings']]
    best=min(valid,key=lambda r:r['score']['objective']) if valid else None
    s.n.dump(OUT/'BEST_PARAMETERS.json',dict(selected=best,baseline=results[0],trials_attempted=search.trials_started,
        actual_physics_steps=search.physics_steps,source_hashes_unchanged={p:s.n.sha(p)==h for p,h in hashes.items()},
        status='PENDING_STRICT_NATIVE_VALIDATION' if best else 'NO_VALID_CANDIDATE',neural_success=False))

if __name__=='__main__':main()
