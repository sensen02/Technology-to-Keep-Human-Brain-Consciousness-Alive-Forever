"""Stdlib source-only terrain derivative preparation; never imports a physical backend.

Parent may invoke --prepare after reviewing PREREG. No frozen file is written.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = PROJECT / 'experiments/banc-walking-recovery/20261110T152714Z'
XML = PROJECT / 'experiments/banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml'
POSE = PROJECT / 'experiments/banc-triad-stand-pose/20261110T030000Z/STAND_POSE_TRIAD_1.45.json'
SENSORY = PROJECT / 'experiments/neck-real-structure/20261113T064313Z/sensory_body_repair'
CACHE = Path('/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a')
VALIDATED_UTILITY = HERE.parent / 'terrain_contact_probe/terrain_contacts.py'
SOURCE_XML_SHA = 'd0839271100561cbd288cc662f0c94639755e6ed773a2b7a7d547f5fec2fe93a'
VALIDATED_UTILITY_SHA = '0b6afaed93ac444228d398f7ffdcc6c30ef6ec81c935f41783553e3db08ba3a6'
CASES = ('plane_control', 'flat_box_control', 'mild_blocks_seed17')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_prepared():
    frozen = json.loads((HERE/'PREPARED/FROZEN.json').read_text())
    for path, digest in frozen['source_hashes'].items():
        if sha(path) != digest:
            raise ValueError('Prepared source drift: ' + path)
    return frozen


def derivative(case):
    root = ET.parse(XML).getroot()
    assets = []
    for mesh in root.findall('./asset/mesh'):
        relative = mesh.attrib['file']
        local = (XML.parent/relative).resolve()
        resolved = local if local.is_file() else CACHE/Path(relative).name
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        mesh.set('file', str(resolved))
        assets.append(resolved)
    world = root.find('worldbody')
    floor = world.find("geom[@name='floor']")
    if floor is None or floor.attrib['type'] != 'plane':
        raise ValueError('Expected original world floor plane')
    for geom in world.iter('geom'):
        geom.set('group', '1')  # all fly geoms excluded from terrain-only ray group
    floor.set('group', '5')
    floor.set('name', 'terrain_plane')
    terrain = []
    if case == 'flat_box_control':
        floor.set('name', 'terrain_flat_box')
        floor.set('type', 'box')
        floor.set('pos', '0 0 -0.5')
        floor.set('size', '100 100 0.5')
        terrain.append(dict(name='terrain_flat_box', top_z=0., footprint=[-100.,100.,-100.,100.]))
    else:
        terrain.append(dict(name='terrain_plane', top_z=0., footprint='infinite'))
    if case == 'mild_blocks_seed17':
        rng = random.Random(17)
        for index in range(6):
            start = .8 + .38 * index
            height = rng.choice((.02, .04))
            name = 'terrain_block_' + str(index).zfill(2)
            ET.SubElement(world, 'geom', dict(name=name, type='box', group='5',
                pos=f'{start+.125:.17g} 0 {height/2:.17g}',
                size=f'0.125 5 {height/2:.17g}', contype='1', conaffinity='1',
                condim='3', solref='.002 1', rgba='.45 .40 .35 1'))
            terrain.append(dict(name=name, top_z=height, footprint=[start,start+.25,-5.,5.],
                                height_over_pad_radius=height/.008))
    return ET.tostring(root, encoding='unicode')+'\n', assets, terrain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if not args.prepare:
        print('SOURCE_ONLY; no files prepared, no model constructed')
        return
    if sha(XML) != SOURCE_XML_SHA or sha(VALIDATED_UTILITY) != VALIDATED_UTILITY_SHA:
        raise ValueError('Original XML or parent-validated utility drift')
    out = HERE/'PREPARED'
    out.mkdir(exist_ok=False)
    paths = [*HERE.glob('*.py'), HERE/'PREREG.json', XML, POSE, VALIDATED_UTILITY,
             SENSORY/'runtime.py', SENSORY/'adapter.py', SENSORY/'INPUT_BINDINGS.json',
             SENSORY/'FROZEN.json', *BASE.glob('*.py'), *(BASE/'sources').glob('*.py')]
    cases = {}
    for case in CASES:
        text, assets, terrain = derivative(case)
        output = out/(case+'.xml')
        with output.open('x') as stream:
            stream.write(text)
        paths.extend(assets)
        paths.append(output)
        cases[case] = dict(xml=str(output), xml_sha256=sha(output), terrain=terrain,
                           observation_version='TERRAIN_RELATIVE_PAD_CLEARANCE_238_V1',
                           legacy_plane_only_values_retained=True)
    source_hashes = {str(p.resolve()):sha(p) for p in sorted(set(paths))}
    ledger = dict(kind='FULL_BODY_TERRAIN_DERIVATIVE_SOURCE_ONLY_V1', cases=cases,
                  source_xml=str(XML), source_xml_sha256=SOURCE_XML_SHA,
                  original_scientific_sources_unchanged=True, physical_steps=0,
                  semantic_changes=['terrain registry mask', 'terrain-only ray groups',
                                    'padz=terrain-relative vertical clearance',
                                    'normal/tangent measurements added alongside legacy world-z/xy'],
                  old_identity_capture_must_not_be_used_as_actual_xml_identity=True)
    save(out/'MODEL_LEDGER.json', ledger)
    source_hashes[str(out/'MODEL_LEDGER.json')] = sha(out/'MODEL_LEDGER.json')
    save(out/'FROZEN.json', dict(kind='SOURCE_ONLY_TERRAIN_BODY_PREPARED_V1',
                               source_hashes=source_hashes, model_constructions=0, physical_steps=0))
    print(json.dumps(dict(prepared=str(out), cases=list(cases), physical_steps=0)))


if __name__ == '__main__':
    main()
