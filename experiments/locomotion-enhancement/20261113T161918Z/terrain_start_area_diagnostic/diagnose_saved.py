"""Parent-only short saved-array diagnostic; no model/NN/backend/physics.

Uses exact saved world sphere centers. No reconstruction from mean foot or
heading is needed; no scene is generated or edited. NumPy imports after flag.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = ROOT/'terrain_neural_baseline'
MODEL = ROOT/'terrain_body_baseline/PREPARED'
CASES = ('plane_control','mild_blocks_seed17')
LEGS = ('LF','LH','LM','RF','RH','RM')
RADIUS = .008
MARGIN = .05
TOL = 1e-10


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def source_paths():
    paths = [Path(__file__).resolve(),HERE/'PREREG.json',RUN/'FROZEN.json',RUN/'runtime.py',RUN/'metrics.py',
             MODEL/'MODEL_LEDGER.json',MODEL/'plane_control.xml',MODEL/'mild_blocks_seed17.xml']
    for case in CASES:
        folder = RUN/'trials'/case
        names = ['IDENTITY.json','INITIAL_STATE.json','INITIAL_TERRAIN.json','WARM_FLAT_CHECK.json','terrain_1ms.npz','trace_1ms.npz']
        names += ['RESULT.json','VERIFY.json'] if case=='plane_control' else ['FAILURE.json','PARTIAL_NATIVE.json']
        paths.extend(folder/name for name in names)
    paths.extend(RUN/'trials/flat_box_control'/name for name in ('RESULT.json','VERIFY.json'))
    return paths


def load_selected(np,folder):
    extra_keys = ('time','pad_center_world','pad_ground_height','com_world_xyz','com_ground_height',
        'loads_normal','legacy_plane_loads','registered_contact_count','legacy_plane_contact_count','control_source')
    trace_keys = ('time','qpos','loads','relativefoot','control_source')
    with np.load(folder/'terrain_1ms.npz',allow_pickle=False) as archive:
        extra = {key:archive[key].copy() for key in extra_keys}
    with np.load(folder/'trace_1ms.npz',allow_pickle=False) as archive:
        trace = {key:archive[key].copy() for key in trace_keys}
    n = len(extra['time'])
    if extra['pad_center_world'].shape != (n,6,3,3) or extra['pad_ground_height'].shape != (n,6,3):
        raise ValueError('Expected exact saved sphere centers and terrain heights')
    if trace['qpos'].shape != (n,49) or not np.array_equal(extra['time'],trace['time']) or not np.array_equal(extra['control_source'],trace['control_source']):
        raise ValueError('Original and terrain trace clocks/source mismatch')
    if not all(np.isfinite(value).all() for group in (extra,trace) for value in group.values()):
        raise ValueError('Nonfinite diagnostic array')
    expected = np.arange(1,n+1)*.001
    if np.max(np.abs(extra['time']-expected))>1e-8:
        raise ValueError('Unexpected1ms clocks')
    return extra,trace


def first(np,mask,times,*,kind):
    locations = np.argwhere(mask)
    if not len(locations):
        return dict(found=False,kind=kind)
    at = locations[0]
    result = dict(found=True,kind=kind,sample_index=int(at[0]),time_s=float(times[at[0]]))
    if len(at)>1:
        result.update(leg=LEGS[int(at[1])],leg_index=int(at[1]))
    if len(at)>2:
        result['sphere_index'] = int(at[2])
    return result


def envelope(np,points,com,root_xyz):
    flat = points.reshape(-1,3)
    low,high = flat.min(axis=0),flat.max(axis=0)
    perleg = {}
    for li,leg in enumerate(LEGS):
        values = points[:,li].reshape(-1,3)
        perleg[leg] = dict(center_min_xyz=values.min(axis=0).tolist(),center_max_xyz=values.max(axis=0).tolist())
    return dict(sphere_center_min_xyz=low.tolist(),sphere_center_max_xyz=high.tolist(),
        sphere_material_extent_min_xyz=(low-RADIUS).tolist(),sphere_material_extent_max_xyz=(high+RADIUS).tolist(),
        COM_min_xyz=com.min(axis=0).tolist(),COM_max_xyz=com.max(axis=0).tolist(),
        free_root_translation_min_xyz=root_xyz.min(axis=0).tolist(),free_root_translation_max_xyz=root_xyz.max(axis=0).tolist(),
        per_leg=perleg)


def box_gap(np,points,item):
    x0,x1,y0,y1 = item['footprint']
    low = np.array([x0,y0,0.])
    high = np.array([x1,y1,item['top_z']])
    outside = np.maximum(np.maximum(low-points,points-high),0.)
    return np.sqrt((outside*outside).sum(axis=-1))-RADIUS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-read-saved-arrays',action='store_true')
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    if not args.parent_read_saved_arrays:
        print('SOURCE_ONLY no arrays/NN/models/physics loaded')
        return
    if args.out is None:
        parser.error('--out required and must be new')
    started = time.monotonic()
    paths = source_paths()
    before = {str(path.resolve()):sha(path) for path in paths}
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=False)
    save(out/'SOURCE_HASHES_BEFORE.json',before)
    try:
        checks = {}
        plane_result = read(RUN/'trials/plane_control/RESULT.json')
        plane_verify = read(RUN/'trials/plane_control/VERIFY.json')
        flat_result = read(RUN/'trials/flat_box_control/RESULT.json')
        flat_verify = read(RUN/'trials/flat_box_control/VERIFY.json')
        for case,result,audit in (('plane_control',plane_result,plane_verify),('flat_box_control',flat_result,flat_verify)):
            folder = RUN/'trials'/case
            checks[case+'_archive_integrity_qualified'] = result['terrain_case']==case and result['physical_steps']==1232000 and audit['integrity_all_pass'] and all(audit['checks'].values()) and audit['result_sha256']==sha(folder/'RESULT.json')
        failure = read(RUN/'trials/mild_blocks_seed17/FAILURE.json')
        native = read(RUN/'trials/mild_blocks_seed17/PARTIAL_NATIVE.json')
        guard = read(RUN/'trials/mild_blocks_seed17/WARM_FLAT_CHECK.json')
        checks['mild_guard_refused_before_ANN'] = failure['physical_endpoints']==5800 and failure['neural_records']==1160 and 'ANN phase not admitted' in failure['error'] and native['actual_native_steps']==464000 and native['post_native_steps']==0 and guard['checked_before_ANN'] and not guard['all_passed']
        for case in CASES:
            folder = RUN/'trials'/case
            result = plane_result if case=='plane_control' else failure
            bindings = {'trace_1ms.npz':result['trace'] if case=='plane_control' else result['artifacts']['trace'],
                'terrain_1ms.npz':result['terrain_trace'] if case=='plane_control' else result['terrain_partial_artifacts']['terrain_1ms.npz'],
                'INITIAL_TERRAIN.json':result['initial_terrain'] if case=='plane_control' else result['terrain_partial_artifacts']['INITIAL_TERRAIN.json']}
            for name,record in bindings.items():
                checks[case+'_artifact_'+name] = record['sha256']==sha(folder/name)
        frozen = read(RUN/'FROZEN.json')['source_hashes']
        for path in (RUN/'runtime.py',RUN/'metrics.py',MODEL/'MODEL_LEDGER.json',MODEL/'plane_control.xml',MODEL/'mild_blocks_seed17.xml'):
            checks['frozen_source:'+str(path)] = frozen[str(path)]==sha(path)
        if not all(checks.values()):
            raise ValueError('Saved source/archive eligibility failed '+str([k for k,v in checks.items() if not v]))
        import numpy as np
        data = {case:load_selected(np,RUN/'trials'/case) for case in CASES}
        initial = {case:read(RUN/'trials'/case/'INITIAL_TERRAIN.json') for case in CASES}
        ledger = read(MODEL/'MODEL_LEDGER.json')['cases']['mild_blocks_seed17']
        blocks = [item for item in ledger['terrain'] if item['name'].startswith('terrain_block_')]
        report = {}
        for case,(extra,trace) in data.items():
            n = len(extra['time'])
            if n != (15400 if case=='plane_control' else 5800):
                raise ValueError('Unexpected stored endpoint count')
            init = initial[case]
            all_times = np.r_[0.,extra['time']]
            points = np.concatenate((np.asarray(init['pad_center_world'])[None],extra['pad_center_world']))
            com = np.concatenate((np.asarray(init['com_world_xyz'])[None],extra['com_world_xyz']))
            root = np.concatenate((np.asarray(read(RUN/'trials'/case/'INITIAL_STATE.json')['qpos'])[:3][None],trace['qpos'][:,:3]))
            prefix = np.arange(len(all_times))<=5800
            teacher = (np.arange(len(all_times))>=1000)&(np.arange(len(all_times))<=5800)
            ray_first = first(np,extra['pad_ground_height']>TOL,extra['time'],kind='first1ms ray-over-raised-terrain, not contact')
            if ray_first['found']:
                i,l,s = ray_first['sample_index'],ray_first['leg_index'],ray_first['sphere_index']
                point = extra['pad_center_world'][i,l,s]
                matches = [item['name'] for item in blocks if item['footprint'][0]<=point[0]<=item['footprint'][1] and item['footprint'][2]<=point[1]<=item['footprint'][3]]
                ray_first.update(actual_ground_height=float(extra['pad_ground_height'][i,l,s]),center_world=point.tolist(),ledger_terrain_under_ray=matches)
            nonplane_count = extra['registered_contact_count']-extra['legacy_plane_contact_count']
            normal_residual = extra['loads_normal']-extra['legacy_plane_loads']
            case_report = dict(saved_rows=n,initial_spheres=np.asarray(init['pad_center_world']).tolist(),
                prefix_0_to_5p8=envelope(np,points[prefix],com[prefix],root[prefix]),
                teacher_1_to_5p8=envelope(np,points[teacher],com[teacher],root[teacher]),
                first_ray_over_raised_terrain=ray_first,
                first_sampled_nonplane_contact_record=first(np,nonplane_count>0,extra['time'],kind='1ms contact record; may include inactive margin contacts'),
                first_positive_nonplane_normal_by_leg=first(np,normal_residual>TOL,extra['time'],kind='normal_loads minus horizontal-plane worldzloads; sampled force attribution'),
                max_nonplane_normal_by_leg={leg:float(normal_residual[:,li].max()) for li,leg in enumerate(LEGS)},
                normal_residual_min_native=float(normal_residual.min()),
                first_native_contact_time='UNKNOWN: native contacts are aggregate counts/peaks, not full per-contact time arrays',
                geometry_overlap_by_block=[])
            for block in blocks:
                gaps = box_gap(np,points,block)
                at = first(np,gaps<=TOL,all_times,kind='analytic sphere/AABB overlap; not a solved force')
                if at['found']:
                    i,l,s = at['sample_index'],at['leg_index'],at['sphere_index']
                    at.update(center_world=points[i,l,s].tolist(),surface_gap_native=float(gaps[i,l,s]))
                case_report['geometry_overlap_by_block'].append(dict(name=block['name'],top_z=block['top_z'],footprint=block['footprint'],first_overlap=at))
            if case=='plane_control':
                auto = np.arange(len(all_times))>=5800
                case_report['autonomous_5p8_to_15p4'] = envelope(np,points[auto],com[auto],root[auto])
                loaded = trace['loads']>TOL
                # A leg load cannot identify which of its three spheres is in contact.
                support_points = extra['pad_center_world'][:5800][loaded[:5800]]
                case_report['load_qualified_leg_sphere_envelope'] = dict(
                    status='conservative all3 spheres of a positive-loaded leg; not per-contact-point envelope',
                    min_xyz=(support_points.reshape(-1,3).min(axis=0)-RADIUS).tolist() if len(support_points) else None,
                    max_xyz=(support_points.reshape(-1,3).max(axis=0)+RADIUS).tolist() if len(support_points) else None)
                material_max = points[prefix].reshape(-1,3).max(axis=0)+RADIUS
                material_min = points[prefix].reshape(-1,3).min(axis=0)-RADIUS
                necessary_front = float(material_max[0]+MARGIN)
                auto_front = extra['pad_center_world'][5800:,:,:,0]+RADIUS
                necessary_mask = auto_front>=necessary_front
                case_report['flat_start_area_analysis_only'] = dict(
                    all_warm_sphere_material_xy_min=(material_min[:2]-MARGIN).tolist(),
                    all_warm_sphere_material_xy_max=(material_max[:2]+MARGIN).tolist(),
                    engineering_margin_native=MARGIN,necessary_first_forward_obstacle_x=necessary_front,
                    original_first_obstacle_x=float(blocks[0]['footprint'][0]),scene_written_or_moved=False,
                    name='necessary geometric clearance bound only; parent must freeze a new separately identified scene',
                    saved_ANN_max_sphere_leading_x=float(auto_front.max()),
                    saved_ANN_can_reach_necessary_forward_bound=bool(necessary_mask.any()),
                    first_saved_ANN_potential_reach=first(np,necessary_mask,extra['time'][5800:],kind='necessary x reach only; neither new-scene contact nor causal counterfactual'),
                    if_not_reached='Current saved9.6s ANN does not reach the required flat-prefix bound; do not silently shift obstacles closer. Any longer window or common translated reference is a separately preregistered task.')
            report[case] = case_report
        a,b = data['plane_control'],data['mild_blocks_seed17']
        prefix_differences = {}
        for key,aa,bb in (('world_sphere_centers',a[0]['pad_center_world'][:5800],b[0]['pad_center_world']),
            ('free_root_qpos',a[1]['qpos'][:5800],b[1]['qpos']),('actual_loads',a[1]['loads'][:5800],b[1]['loads'])):
            unequal = np.any(aa!=bb,axis=tuple(range(1,aa.ndim)))
            prefix_differences[key] = first(np,unequal,b[0]['time'],kind='first1ms exact difference vs qualified plane trajectory')
        warm_block = sum(count for name,count in native['warm_native_contacts_by_terrain'].items() if name.startswith('terrain_block_'))
        all_block = sum(count for name,count in native['native_contacts_by_terrain'].items() if name.startswith('terrain_block_'))
        after = {str(path.resolve()):sha(path) for path in paths}
        checks['selected_sources_unchanged_before_after'] = before==after
        checks['no_backend_or_NN_imported'] = not any(name in sys.modules for name in ('torch','mujoco','physics','flygym','student_wave'))
        if not all(checks.values()):
            raise ValueError('Post-diagnostic source/backend check failed')
        save(out/'SOURCE_HASHES_AFTER.json',after)
        result = dict(kind='SAVED_WARM_START_GEOMETRY_DIAGNOSTIC_V1',checks=checks,all_passed=True,
            reports=report,prefix_exact_differences=prefix_differences,
            mild_native_metadata=dict(actual_native_steps=native['actual_native_steps'],post_native_steps=native['post_native_steps'],
                all_block_contact_records=all_block,warm_block_contact_records=warm_block,
                all_native_contacts_by_terrain=native['native_contacts_by_terrain'],
                warm_native_contacts_by_terrain=native['warm_native_contacts_by_terrain'],
                contacted_block_source_geometry=[item for item in blocks if native['native_contacts_by_terrain'].get(item['name'],0)>0],
                settle_block_contact_records=all_block-warm_block,
                nonfoot_normal_peak_native=native['nonfoot_normal_peak_native'],extra_force_max=native['extra_force_max'],
                native_first_contact_time='UNKNOWN',native_per_contact_leg='UNKNOWN: only aggregate native counts retained'),
            original_guard=guard,leg_order=list(LEGS),sphere_radius_native=RADIUS,
            contact_force_diagnostic_tolerance=TOL,NN_forwards=0,Core_constructions=0,new_physical_steps=0,
            scene_modified=False,old_controls_reused_as_archive_only=True,
            mild_rough_behavior_status='NOT_ADMITTED_NOT_TESTED: stopped at5.8 before ANN; no autonomous rough-pass/fail claim',
            source_hashes_before=before,source_hashes_after=after,wall_s=time.monotonic()-started)
        save(out/'RESULT.json',result)
        print(json.dumps(dict(all_passed=True,wall_s=result['wall_s'],mild_status=result['mild_rough_behavior_status'],
            flat_prefix_analysis=report['plane_control']['flat_start_area_analysis_only'],native_settle_contacts=all_block-warm_block)))
    except BaseException as error:
        save(out/'FAILURE.json',dict(error=repr(error),source_hashes_before=before,new_physical_steps=0,scene_modified=False))
        raise


if __name__ == '__main__':
    main()
