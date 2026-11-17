"""Repeat the admitted far replant in the same response selector as the local peg reflex."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from compose import HERE, GAP, STUDENT, CHECKPOINT, STOCK_SPEED, Response, checksum, load, read, source_at, source_imports, sources, write

LANE=HERE.parent/'gap'


def run(output,reference,stop_at=None,ablate='none'):
    original=read(reference/'RESULT.json')
    short=stop_at is not None
    pin=read(STUDENT/'STUDENT_SUITE.json')
    if not pin['main_qualified'] or checksum(CHECKPOINT)!=pin['checkpoint_sha256']:
        raise ValueError('Unqualified stumble assembly weight')
    if not short and (original.get('native_iterations',0) != 1380000 or original.get('reload_ms') is None or
                      not original.get('before_renewal_stand',{}).get('passed')):
        raise ValueError('Full gap run source lacks admitted local replant and stop')
    stumble,gap,_=sources()
    with source_imports(GAP,gap):
        crossing=source_at('composed_gap_native_bench',LANE/'crossing.py')
    routes=[]
    positive=load(CHECKPOINT)

    def factory(sensor,plan,shape,mode,reach,gain,retain):
        gap_actor=crossing.TerrainReflex(sensor,plan,shape,mode,reach,gain,retain)
        local_stumble=stumble.Local(sensor.body,crossing.RESUMED_WALK)
        motor=Response(positive,gap_actor,gap.frozen_policy(),ablate)

        class IntegratedPolicy:
            def __getattr__(self,key):
                return getattr(gap_actor,key)

            def __call__(self,t,x,intent,local,nominal,measured):
                now=local_stumble.sense(measured)
                action,route=motor.act(x,[intent*STOCK_SPEED,0.],now,local,measured,t)
                routes.append((t,route))
                return action

        return IntegratedPolicy()

    parent=crossing.stan.identity.capture
    def pin_sources(core,scope,*,extra_paths):
        return parent(core,{**scope,'same_motion_response_entry':True,'ablate':ablate},
                      extra_paths=[*extra_paths,HERE/'compose.py',Path(__file__),CHECKPOINT,
                                   STUDENT/'stumble_policy.py',Path(stumble.__file__),reference/'RESULT.json'])

    crossing.stan.identity.capture=pin_sources
    try:
        duration=stop_at if short else crossing.END
        crossing.run(output,'far',.036,20.,400,duration,True,branch_factory=factory)
    finally:
        crossing.stan.identity.capture=parent
    result=read(output/'RESULT.json')
    selected={key:[t for t,kind in routes if kind==key] for key in ('gap','stumble','nominal')}
    names=('TRACE.npz','DECISIONS.npz','NATIVE.npz','GEOMETRY_NATIVE.npz')
    if short:
        with np.load(output/'TRACE.npz',allow_pickle=False) as new, np.load(reference/'TRACE.npz',allow_pickle=False) as old:
            exact={name:np.array_equal(new[name],old[name][:len(new[name])]) for name in new.files}
    else:
        exact={name:checksum(output/name)==checksum(reference/name) for name in names}
    complete=result['actual_ms']==duration and all(exact.values()) and bool(selected['gap']) and not selected['stumble']
    report=dict(source=str(reference),source_qualified_by_full_physical_replant=not short,actual_ms=result['actual_ms'],
                selected={key:dict(count=len(times),first_ms=times[0] if times else None) for key,times in selected.items()},
                geometry_aware_gap_teacher=True,ash_gap_reload_ms=result['reload_ms'],matched_physical_replay=exact,
                response_entry=str(HERE/'compose.py'),root_path_writes=False,time_dependent_local_gap_only=True,
                passed=bool(complete and not short and ablate=='none'),short_prefix_compares=bool(complete and short),
                no_upper_network=True,ablate=ablate)
    write(output/'ASSEMBLY_AUDIT.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('reference',type=Path)
    p.add_argument('--stop-at',type=int)
    p.add_argument('--ablate',choices=('none','stumble','gap'),default='none')
    a=p.parse_args()
    run(a.output.resolve(),a.reference.resolve(),a.stop_at,a.ablate)
