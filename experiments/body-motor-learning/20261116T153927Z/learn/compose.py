"""Select admitted local stumble and loss responses in a common physical body entry."""
import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
HISTORY = HERE.parents[1]
STUMBLE = HISTORY/'20261116T060654Z'
GAP = HISTORY/'20261116T063132Z'/'teacher'
STUDENT = STUMBLE/'student'
CHECKPOINT = STUDENT/'FIT_RF_A'/'step_0240.pt'
STANCE = GAP/'SAFE_STOP_WEIGHT_EVIDENCE.json'
BRAKE = GAP/'PLAN_BRAKE_RF_ONLY.json'
STOCK_SPEED = .07171300063848401
if str(STUDENT) not in sys.path:
    sys.path.insert(0,str(STUDENT))
from stumble_policy import checksum, load, read, write


@contextmanager
def source_imports(folder, physical=None):
    reserved=('physical','geometry','gap_stop','gap_servo')
    previous={name:sys.modules.pop(name) for name in reserved if name in sys.modules}
    sys.path.insert(0,str(folder))
    if physical is not None:
        sys.modules['physical']=physical
    try:
        yield
    finally:
        sys.path.remove(str(folder))
        for name in reserved:
            sys.modules.pop(name,None)
        sys.modules.update(previous)


def source_at(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sources():
    with source_imports(STUMBLE/'teacher'):
        stumble=source_at('historical_stumble_bench',STUMBLE/'teacher'/'physical.py')
    with source_imports(GAP):
        gap=source_at('historical_gap_bench',GAP/'physical.py')
    with source_imports(GAP,gap):
        brake=source_at('historical_gap_brake',GAP/'gap_weight.py')
    return stumble,gap,brake.Weight


class Response:
    """The caller supplies current measurements. Only the engineering brake is stateful."""
    def __init__(self, stumble, gap, stock, ablate):
        self.stumble=stumble
        self.gap=gap
        self.stock=stock
        self.ablate=ablate

    def act(self, x, command, local_stumble, local_gap, measured, elapsed_ms):
        v,w=np.asarray(command)
        if w != 0. or v not in (0.,STOCK_SPEED):
            raise ValueError('This terrain assembly only admits nominal forward and stop')
        intent=int(v>0.)
        nominal=self.stock(x,intent)
        if self.ablate != 'gap' and self.gap.onset is not None:
            return self.gap(elapsed_ms,x,intent,local_gap,nominal,measured),'gap'
        if self.ablate != 'stumble':
            candidate=self.stumble(x,intent,local_stumble)
            if not np.array_equal(candidate,nominal):
                return candidate,'stumble'
        if self.ablate != 'gap':
            candidate=self.gap(elapsed_ms,x,intent,local_gap,nominal,measured)
            if self.gap.onset is not None:
                return candidate,'gap'
        return nominal,'nominal'


def check_runtime(path, qualified, sensor_width, match, selected, ablate, flat):
    fresh=read(path/'RESULT.json')
    expected=read(qualified/'RESULT.json')
    fresh_native=read(path/'NATIVE_AUDIT.json')
    expected_native=read(qualified/'NATIVE_AUDIT.json')
    actual_pins=dict(TRACE=checksum(path/'TRACE.npz'),NATIVE=checksum(path/'NATIVE.npz'))
    if actual_pins != dict(TRACE=fresh['trace_sha256'],NATIVE=fresh_native['native_sha256']):
        raise ValueError('Composition source file failed runtime fingerprint')
    exact=dict(trace=actual_pins['TRACE']==expected['trace_sha256'],
               native=actual_pins['NATIVE']==expected_native['native_sha256'])
    with np.load(path/'DECISIONS.npz',allow_pickle=False) as comp, np.load(qualified/'DECISIONS.npz',allow_pickle=False) as known:
        primary=comp['local'][:,:sensor_width] if selected=='stumble' else comp['local'][:,19:19+sensor_width]
        exact.update(current_primary=np.array_equal(primary,known['local']),
                     current_observations=np.array_equal(comp['x'],known['x']),
                     chosen_action=np.array_equal(comp['action'],known['action']))
        audits=(comp['action'],comp['base'],comp['ms'])
    changed=np.flatnonzero(np.any(audits[0]!=audits[1],axis=1))
    actual_changed=[int(audits[2][i]) for i in changed]
    times={mode:[ms for ms,choice in match if choice==mode] for mode in ('stumble','gap','nominal')}
    intended=selected if ablate=='none' else 'nominal'
    other='gap' if selected=='stumble' else 'stumble'
    # The gap retains safe stop as its task-specific qualification. The ordinary continued gait gate is retained separately.
    fixed=read(STANCE)['full']
    safe=bool(fixed['safe_stance_pass'] and fixed['physical_zero_external_force']) if selected=='gap' else True
    expected_behavior=bool(fresh['native_joint_pass'] if selected=='stumble' else
                           safe and fresh['event']['observed_gap_event'])
    replay=all(exact.values()) if ablate=='none' else False
    qualified_now=bool(not flat and replay and not times[other] and times[intended] and expected_behavior)
    comp_gate=dict(task=selected,ablation=ablate,run=str(path),source=str(qualified),
                   extent=fresh['actual_ms'],native_frozen_readings=fresh_native['native_integrations'],
                   actual_trigger_ms={mode:milliseconds[0] if milliseconds else None for mode,milliseconds in times.items()},
                   selected_decisions={mode:len(ms) for mode,ms in times.items()},
                   physical_modified_action_count=len(actual_changed),
                   first_modified_action_ms=actual_changed[0] if len(changed) else None,
                   source_physical_exact=exact,passed=qualified_now,includes_upper_network=False,
                   unlimited_continued_walk_gate=fresh['native_joint_pass'],current_contact_selects_response=True,
                   engineering_memory={'gap':'timed lift/retract/hold phases with physical touchdown feedback'},
                   flat_no_reflex=bool(flat and fresh['native_joint_pass'] and not times['stumble'] and not times['gap']),
                   confirms_gap_nearside_stable_only=bool(qualified_now and selected=='gap'))
    if selected=='gap':
        comp_gate['origin_gap_qualified_record']=fixed
    return comp_gate


def run(output, selected, ablation='none', flat=False):
    stumble,gap,weight=sources()
    suite=read(STUDENT/'STUDENT_SUITE.json')
    locked=read(STANCE)
    if not suite['main_qualified'] or checksum(CHECKPOINT)!=suite['checkpoint_sha256']:
        raise ValueError('Stumble policy failed prior case qualification')
    if not locked['full']['safe_stance_pass']:
        raise ValueError('Gap teacher has no qualified physical event')
    previous=stumble if selected=='stumble' else gap
    primary_local=previous.Local
    stumble_local=stumble.Local
    gap_local=gap.Local
    s_fields=tuple(stumble.LOCAL_FIELDS)
    g_fields=tuple(gap.LOCAL_FIELDS)
    if len(s_fields)!=19 or len(g_fields)!=17:
        raise ValueError('Historical named observation layout changed')
    collision_fields=tuple('stumble_'+key for key in s_fields)+tuple('gap_'+key for key in g_fields)
    correction=load(CHECKPOINT)
    stock=previous.frozen_policy()
    qualified=STUDENT/'STUDENT_PEG_A' if selected=='stumble' else Path(locked['full']['path'])
    case=STUMBLE/'teacher'/'PEG_A.json' if selected=='stumble' else GAP/'AS.json'
    probe=None if selected=='stumble' else 6450
    record=[]
    bound=[]

    class BothLocal:
        def __init__(self, body, walk_ms):
            self.stumble=stumble_local(body,walk_ms)
            self.gap=gap_local(body,walk_ms)
            self.data=self.stumble if selected=='stumble' else self.gap
            self.origin=body.d.time
            self.measure=None
            self.controller=Response(correction,weight(self.gap,read(BRAKE)),stock,ablation)
            bound.append(self)

        def __getattr__(self, key):
            return getattr(self.data,key)

        def sense(self, measured):
            self.measure=measured
            return np.r_[self.stumble.sense(measured),self.gap.sense(measured)]

    def actor(x, intent, local):
        sensor,=bound
        ms=int(round((sensor.body.d.time-sensor.origin)*1000))
        if ms%5:
            raise RuntimeError('Engineering time and actuator schedule diverged')
        action,route=sensor.controller.act(x,[intent*STOCK_SPEED,0.],local[:19],local[19:],sensor.measure,ms)
        record.append((ms,route))
        return action

    previous.Local=BothLocal
    previous.LOCAL_FIELDS=collision_fields
    try:
        previous.run(output,'student',None if flat else case,policy=actor,native=True,
                     probe_ms=probe,policy_source=Path(__file__))
    finally:
        previous.Local=primary_local
        previous.LOCAL_FIELDS=s_fields if selected=='stumble' else g_fields
    primary_width=len(s_fields) if selected=='stumble' else len(g_fields)
    report=check_runtime(output,qualified,primary_width,record,selected,ablation,flat)
    report['flat_control']=flat
    report['pins']={str(path):checksum(path) for path in (CHECKPOINT,STUDENT/'STUDENT_SUITE.json',STANCE,BRAKE,
             GAP/'gap_weight.py',GAP/'gap_stop.py',STUMBLE/'teacher'/'physical.py',GAP/'physical.py',case,Path(__file__))}
    write(output/'ASSEMBLY_AUDIT.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case',choices=('stumble','gap'))
    p.add_argument('output',type=Path)
    p.add_argument('--ablate',choices=('none','stumble','gap'),default='none')
    p.add_argument('--flat',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1)
    run(args.output.resolve(),args.case,args.ablate,args.flat)
