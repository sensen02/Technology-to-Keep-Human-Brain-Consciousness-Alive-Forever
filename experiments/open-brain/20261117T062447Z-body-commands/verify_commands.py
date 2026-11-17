"""Read independent replay records and verify physical and surgical outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from expansion import CUT, DEVELOP, FAMILIES, FIRST, HELD, HERE, MODEL_FILE, PARENT, TICKS, DT, sha, write
from cases import HELD as FIRST_HELD


def record(directory, name):
    result = json.loads((directory/name/'RESULT.json').read_text())
    arrays_path = directory/name/'TRACE.npz'
    assert result['trace_sha256'] == sha(arrays_path)
    assert result['body_snapshot_sha256'] == sha(MODEL_FILE)
    assert result['catalog_sha256'] == sha(PARENT/'CATALOG.json')
    with np.load(arrays_path, allow_pickle=False) as data:
        assert data['trace'].shape == (TICKS, 15)
        assert data['ctrl'].shape == (TICKS//10, 42)
        return result, dict(data)


def summarize(directory, names):
    return {name: json.loads((directory/name/'RESULT.json').read_text()) for name in names}


def paired_speed(records, pattern):
    count = 2 if pattern == 'held_{}_{}' else 1
    output = []
    for i in range(count):
        fast = records[pattern.format('fast', i)]['active_speed_mm_s']
        cruise = records[pattern.format('cruise', i)]['active_speed_mm_s']
        output.append(dict(cruise=cruise, fast=fast, pass_=cruise>0 and fast>=1.25*cruise))
    return output


def verify_replays(formal, replay, names, digest):
    output = {}
    for name in names:
        result, original = record(formal,name)
        rerun, reloaded = record(replay,name)
        assert result['weight_sha256'] == rerun['weight_sha256'] == digest
        assert result['gates'] == rerun['gates']
        same = {key: bool(np.array_equal(original[key], reloaded[key])) for key in original}
        assert all(same.values()), (name, same)
        output[name] = same
    return output


def verify_cut(directory):
    intact, control = record(directory,'held_arc_left_0')
    cut, operated = record(directory,CUT.name)
    entries = cut['arrival_roots']
    assert entries == intact['arrival_roots']
    affected = [entries.index(r) for r in cut['cut_roots']]
    assert affected
    catalog = json.loads((PARENT/'CATALOG.json').read_text())['ports']
    spared = [i for i,r in enumerate(entries) if catalog[r]['direction']=='DN' and i not in affected]
    checks = dict(same_brain_emissions=np.array_equal(control['emitted'][:,affected+spared],
                                                     operated['emitted'][:,affected+spared]),
                  cut_has_emission=bool(operated['emitted'][:,affected].any()),
                  intact_has_arrivals=bool(control['arrived'][:,affected].any()),
                  cut_has_zero_arrivals=not operated['arrived'][:,affected].any(),
                  spared_dn_arrivals=np.array_equal(control['arrived'][:,spared],operated['arrived'][:,spared]),
                  steer_signal_vanishes=not np.any(operated['trace'][:,6]!=operated['trace'][:,7]),
                  control_behavior=intact['pass_'], cut_behavior=cut['pass_'])
    return dict(gates={k:bool(v) for k,v in checks.items()}, roots=cut['cut_roots'],
                intact_arrivals=int(control['arrived'][:,affected].sum()),
                cut_sent=int(operated['emitted'][:,affected].sum()),
                cut_arrivals=int(operated['arrived'][:,affected].sum()),
                intact_yaw=intact['yaw_degrees'], cut_yaw=cut['yaw_degrees'])


def verify_brake(directory, name):
    result, arrays = record(directory,name)
    catalog = json.loads((PARENT/'CATALOG.json').read_text())['ports']
    roots = result['arrival_roots']
    forward = [i for i,r in enumerate(roots) if catalog[r]['group']=='forward']
    halt = [i for i,r in enumerate(roots) if catalog[r]['group']=='halt_web_a']
    tick = result['case']['start'] + 7100
    window = slice(tick+1000,result['case']['stop'])
    return dict(forward_arrivals=int(arrays['arrived'][window, forward].sum()),
                halt_arrivals=int(arrays['arrived'][window, halt].sum()))


def final(chosen):
    digest = sha(chosen)
    names = [case.name for case in (*HELD,CUT)]
    old_names = [case.name for case in FIRST_HELD]
    new = summarize(HERE/'HELD',names)
    old = summarize(HERE/'OLD',old_names)
    replay = verify_replays(HERE/'HELD',HERE/'REPLAY',names,digest)
    old_replay = verify_replays(HERE/'OLD',HERE/'OLD_REPLAY',old_names,digest)
    speed = paired_speed(new,'held_{}_{}')
    cut = verify_cut(HERE/'HELD')
    brake = {name: verify_brake(HERE/'HELD',name) for name in names if 'brake' in name}
    families = {family: all(new[f'held_{family}_{i}']['pass_'] for i in range(2)) for family in FAMILIES}
    joint_shape = torch.load(chosen,weights_only=True,map_location='cpu')['state']['layers.6.weight'].shape
    write(HERE/'FINAL.json', dict(checkpoint=str(chosen),sha256=digest,joint_shape=list(joint_shape),
          new=narrow(new),old=narrow(old),old_replay=old_replay,replay=replay,speed=speed,cut=cut,brake=brake,
          families=families, native_validation_steps=2*TICKS*(len(names)+len(old_names)),
          all_pass=all(families.values()) and all(r['pass_'] for r in old.values()) and
                   all(v['pass_'] for v in speed) and all(cut['gates'].values()) and
                   all(min(b.values())>0 for b in brake.values())))


def narrow(results):
    return {name:{key:value for key,value in r.items()
                  if key not in ('arrival_roots','probes','case','schedule')} for name,r in results.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen',type=Path)
    parser.add_argument('--dev',type=Path)
    args = parser.parse_args()
    if args.dev:
        records = summarize(args.dev,[case.name for case in DEVELOP])
        for name, r in records.items():
            failed=[k for k,v in r['gates'].items() if not v]
            print(name,round(r['active_speed_mm_s'],2),round(r['yaw_degrees'],2),round(r['peak_excursion_mm'],3),failed)
        print('passed',sum(r['pass_'] for r in records.values()),'of',len(records),'pairs',paired_speed(records,'dev_{}'))
    else:
        final(args.chosen.resolve())
