"""Static, source-identified muscle/tendon/MN inventory; never advances physics."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from engine.embodied.adapters import TierConfig, build_neural_tier
from engine.embodied.motor_output import build_motor_manifest

ROOT = Path(__file__).resolve().parent
BODY = ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
POOL = ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'
OSIM = Path('/home/sensen/Desktop/engine修复验收/s5-external-muscle-source/20261028T120000Z/best_combined_full.osim')
EXPECTED = '65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def nums(s):
    return [float(v) for v in s.split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    out = ap.parse_args().out
    assert (out / 'PREREGISTRATION.json').exists()
    assert not (out / 'INVENTORY.json').exists(), 'do not overwrite an evidence run'
    inputs = [BODY, POOL, OSIM, Path(__file__), ROOT/'engine/embodied/adapters.py',
              ROOT/'engine/embodied/motor_output.py', ROOT/'engine/canonical_physics.py']
    hashes = {str(p): sha(p) for p in inputs}
    assert hashes[str(BODY)] == EXPECTED
    tree = ET.parse(BODY).getroot()
    assert tree.find('include') is None, 'includes need explicit resolution'
    # Scope intentionally limited to this exact known XML, not a general MJCF compiler.
    defaults = tree.find("default/default[@class='muscle']/general").attrib
    acts = list(tree.find('actuator'))
    assert len(acts) == 90 and len({a.get('name') for a in acts}) == 90
    assert all(a.tag == 'general' and a.get('class') == 'muscle' for a in acts)
    sites = {}
    meshes = []
    for b in tree.findall('.//worldbody//body'):
        for s in b.findall('site'):
            assert s.get('name') not in sites
            sites[s.get('name')] = dict(body=b.get('name'), attributes=dict(s.attrib))
        for g in b.findall('geom'):
            if g.get('mesh'):
                rgba = nums(g.get('rgba', tree.find('default/geom').get('rgba')))
                meshes.append(dict(body=b.get('name'), mesh=g.get('mesh'), rgba=rgba))
    tendons = {t.get('name'): t for t in tree.find('tendon')}
    cfg = TierConfig(background_mean_nA=0.0, background_sd_nA=0.0)
    built = build_neural_tier(cfg)
    roots = np.asarray(built['data']['root_ids'])[built['global_index']]
    manifest = build_motor_manifest(json.loads(POOL.read_text()), roots, [a.get('name') for a in acts])
    mrows = {r['actuator']: r for r in manifest['rows']}
    src = ET.parse(OSIM).getroot()
    sources = {m.get('name'): m for m in src.findall('.//Millard2012EquilibriumMuscle')}
    assert len(sources) == 15
    rows = []
    for a in acts:
        effective = {**defaults, **a.attrib}
        tendon = tendons[effective['tendon']]
        path = []
        for item in tendon:
            assert item.tag == 'site', 'audit wrapping geometry explicitly'
            path.append(dict(site=item.get('site'), **sites[item.get('site')]))
        row = dict(actuator=a.get('name'), explicit=dict(a.attrib), effective=effective,
                   tendon=dict(tendon.attrib), path=path, motor_pool=mrows[a.get('name')],
                   absolute_force_status='NOT_ESTABLISHED',
                   geometry_status=('LF source comparison below' if a.get('name').startswith('LF')
                                    else 'declared homologous derivation; not independent anatomical measurement'))
        if a.get('name') in sources:
            m = sources[a.get('name')]
            source_params = {k: m.findtext(k) for k in (
                'max_isometric_force','optimal_fiber_length','tendon_slack_length',
                'max_contraction_velocity','activation_time_constant','deactivation_time_constant',
                'minimum_activation','ignore_tendon_compliance')}
            pp = []
            for p in m.findall('.//PathPoint'):
                local_name = a.get('name') + '_' + p.get('name')
                s = sites.get(local_name)
                source_pos = nums(p.findtext('location'))
                ours = nums(s['attributes']['pos']) if s else None
                pp.append(dict(source_point=p.get('name'), source_body=p.findtext('socket_parent_frame'),
                               source_local_position=source_pos, target_site=local_name,
                               target_local_position=ours,
                               raw_coordinate_difference=(np.subtract(ours, source_pos).tolist() if ours else None),
                               interpretation='raw coordinate comparison only; frame/unit equivalence not established'))
            row['source_osim'] = dict(parameters=source_params, path_points=pp,
                raw_fmax_ratio=float(effective['gainprm'].split()[2])/float(source_params['max_isometric_force']))
        rows.append(row)
    assert sum('source_osim' in r for r in rows) == 15
    assert all(r['effective'][k] == 'muscle' for r in rows for k in ('dyntype','gaintype','biastype'))
    payload = dict(scope='static XML + current tier selection, no physics load/forward/step or neural advance',
                   input_sha256=hashes, tier_config=asdict(cfg), tier_fingerprint=built['fingerprint'],
                   tier_root_ids=[int(r) for r in roots],
                   tier_coverage=built['coverage'], rows=rows,
                   shared_motor_pools=manifest['shared_pools'], mesh_geoms=meshes,
                   summary=dict(n_actuators=len(rows), motor_status=dict(Counter(r['motor_pool']['status'] for r in rows)),
                                per_leg={leg:dict(Counter(r['motor_pool']['status'] for r in rows if r['actuator'].startswith(leg)))
                                         for leg in ('LF','LM','LH','RF','RM','RH')},
                                mesh_count=len(meshes), visible_mesh_count=sum(g['rgba'][3] > 0 for g in meshes),
                                dynamics_values=sorted(set(r['effective']['dynprm'] for r in rows)),
                                raw_fmax_ratios={r['actuator']:r['source_osim']['raw_fmax_ratio'] for r in rows if 'source_osim' in r},
                                min_activation_tau_s=min(float(r['effective']['dynprm'].split()[0]) for r in rows),
                                xml_timestep_s=float(tree.find('option').get('timestep'))))
    assert hashes == {str(p):sha(p) for p in inputs}
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    (out / 'INVENTORY.json').write_text(text)
    print(json.dumps(payload['summary'], ensure_ascii=False, indent=2))
    print('INPUT_HASHES_UNCHANGED; static assertions passed')


if __name__ == '__main__':
    main()
