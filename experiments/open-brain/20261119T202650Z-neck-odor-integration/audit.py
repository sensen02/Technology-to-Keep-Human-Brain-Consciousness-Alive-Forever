"""Audit the static lesion at the actual source/receiver and the physical endpoints."""
import argparse
import hashlib
import math
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / '20261119T190005Z-odor-sac/LIVE_DETOUR_ODOR_L'
TARGET = '720575941510475536'
NEIGHBOR = '720575941535086424'


def read(path):
    return json.loads(path.read_text())


def check(condition, folder):
    result = read(folder / 'RESULT.json')
    reference = read(REFERENCE / 'RESULT.json')
    for key in ('seed', 'side', 'neural_seed', 'model_sha', 'geometry', 'max_duration_s'):
        assert result[key] == reference[key], (condition, key)
    receipt = read(folder / 'APPLIED.json')
    ports = {row['root']: row for row in receipt['applied']}
    expected_mode = {'intact':'intact','target_block':'target_only','target_partial':'target_only',
                     'shaft':'shaft'}[condition]
    fixture = HERE/'profiles'/('partial.json' if condition=='target_partial' else 'blocked.json')
    assert receipt['mode']==expected_mode and receipt['target']==TARGET
    assert receipt['profile_sha256']==hashlib.sha256(fixture.read_bytes()).hexdigest()
    routes = {r['root']:r for r in read(fixture)['routes']}
    assert len(ports)==23 and set(receipt['omitted_roots'])==routes.keys()-ports.keys()
    fixed=read(HERE.parent/'20261117T164207Z-fixed-neck/PARAMETERS.json')['ports']
    for root,row in ports.items():
        active=(expected_mode=='shaft' or (expected_mode=='target_only' and root==TARGET))
        source=routes[root]
        assert row['available']==(source['available'] if active else True)
        assert row['gain']==(source['gain'] if active else 1.0)
        if row['available']:
            ms=1000*source['delay_s'] if active else fixed[root]['propagation_ms']
            assert row['delay_ticks']==math.ceil(ms/receipt['dt_ms']-1e-12)
        else:
            assert row['delay_ticks'] is None
    with (np.load(folder/'NECK.npz') as neck, np.load(folder/'BODY.npz') as body,
          np.load(REFERENCE/'BODY.npz') as original, np.load(REFERENCE/'NECK.npz') as ref_neck,
          np.load(folder/'SENSE.npz') as sense, np.load(REFERENCE/'SENSE.npz') as ref_sense):
        trace, meal = body['trace'], body['meal']
        assert np.array_equal(body['initial'], original['initial'])
        assert np.array_equal(sense['samples'][0], ref_sense['samples'][0])
        for key in ('emitted','arrived','ascending','drive','feeding'):
            assert np.array_equal(neck[key][:1600], ref_neck[key][:1600])
        emitted, arrived = neck['emitted'], neck['arrived']
        assert emitted.shape == arrived.shape == (result['native_steps']//2,17)
        assert np.array_equal(neck['weights'], ref_neck['weights'])
        rows = {}
        for i, r in enumerate(neck['roots'].tolist()):
            route = ports[r]
            expected = np.zeros(len(emitted), dtype=bool)
            if route['available']:
                delay = route['delay_ticks']
                assert delay == neck['delays'][i] and delay > 0
                expected[delay:] = emitted[:-delay,i]
            assert np.array_equal(expected, arrived[:,i]), (condition,r)
            rows[r] = dict(source=int(emitted[:,i].sum()), arrived=int(arrived[:,i].sum()),
                           gain=route['gain'])
        assert len(rows)==17 and rows[TARGET]['source']>0
        assert result['native_extra_force']==0 and abs(result['stock_loss']-meal[-1,1])<1e-10
        assert result['intake']==meal[-1,1] and not body['observation'][:,1].any()
        g=result['geometry']
        crosses=(trace[:,1]>=g['far']) & ((trace[:,2]<g['low']) | (trace[:,2]>g['high']))
        stationary=trace[trace[:,0]>=result['duration_s']-.14]
        stand=bool(np.all((stationary[:,4]>=4) & (stationary[:,6]>.94)
                          & (np.linalg.norm(stationary[:,7:9],axis=1)<=.08)))
        assert stand==result['ended_standing']
        equivalence={}
        if condition=='intact':
            for observed, baseline in ((body,original),(neck,ref_neck),(sense,ref_sense)):
                for key in baseline.files:
                    equivalence[key]=bool(np.array_equal(observed[key],baseline[key]))
            assert all(equivalence.values())
        contacts=read(folder/'ARM.json')['wall_contact_changes']
        contact_s=sum(end-start for (start, ids),end in
                      zip(contacts, [entry[0] for entry in contacts[1:]]+[result['duration_s']])
                      if ids)
        return dict(roots=rows, wall_contact_s=contact_s, blocked_modeled_roots=[r for r,p in ports.items() if not p['available']],
                    source_arrival_verified=True, matched_initial_sense_and_1600_ticks=True,
                    original_arrays_identical=equivalence, wall_cleared=bool(crosses.any()),
                    wall_clear_s=float(trace[crosses,0][0]) if crosses.any() else None,
                    intake=result['intake'], first_intake_s=result['first_intake_s'],
                    meal_pass=result['meal_pass'], standing=stand, fall_s=result['first_fall_s'],
                    duration_s=result['duration_s'], native_steps=result['native_steps'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path,nargs='?',default=HERE)
    base=parser.parse_args().results
    cases={name:check(name,base/f'live_{name}_left')
           for name in ('intact','target_partial','target_block','shaft')}
    report=dict(task='P1 controlled odor detour, left, course 89217, neural seed 29',
                target_root=TARGET, neighbor_root=NEIGHBOR, matched_cases=cases)
    (base/'JOINT_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for name,case in cases.items():
        print(name,case['wall_cleared'],case['intake'],case['standing'],case['roots'][TARGET],
              case['roots'][NEIGHBOR])


if __name__=='__main__':
    main()
