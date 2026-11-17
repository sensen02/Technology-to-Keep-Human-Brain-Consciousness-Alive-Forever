"""Parent-only1Core/20ms scene identity/contact check; no fresh NN execution.

The prior actual147 zero-Core fourNN proof is explicitly REUSED after source
identity validation. Full warm-flat coverage is tested only by the new trial.
"""
from pathlib import Path
import argparse
import json
import os

HERE = Path(__file__).resolve().parent
TICKS = 20


def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-resource-released',action='store_true')
    args = parser.parse_args()
    if not args.parent_resource_released:
        print('SOURCE_ONLY no NN/Core/physical execution')
        return
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='1':
            raise ValueError(key+'=1 required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    import runtime
    runtime.verify()
    out = HERE/'ACTUAL_PREFLIGHT'
    out.mkdir(exist_ok=False)
    native_steps = 0
    try:
        old_proof = runtime.read(runtime.OLD_NEURAL/'PREFLIGHT.json')
        old_frozen = runtime.OLD_NEURAL/'FROZEN.json'
        if not old_proof['all_passed'] or not all(old_proof['checks'].values()) or old_proof['Core_constructions']!=0 or old_proof['state_total']!=336 or old_proof['sourcefreeze_sha256']!=runtime.sha(old_frozen):
            raise ValueError('Exact prior147 fourNN zeroCore source-qualified proof required')
        functions = runtime.neural_namespace()
        prepared,metrics,core_module = functions['terrain_modules']()
        base = core_module.load_base_physics()
        np,mj = base.np,base.mj
        Derived = core_module.core_type(base)
        recorder = metrics.TerrainPeakRecorder()
        core = Derived(case=runtime.CASE,recorder=recorder,adhesion=False)
        initial = core.measure()
        ledger = core.terrain_ledger
        first = min(item['footprint'][0] for item in ledger['terrain'] if item['name'].startswith('terrain_block_'))
        ray_checks = []
        position_checks = []
        for item in ledger['terrain']:
            if not item['name'].startswith('terrain_block_'):
                continue
            x0,x1,y0,y1 = item['footprint']
            height,hit = metrics.height(core,[(x0+x1)/2.,(y0+y1)/2.])
            ray_checks.append(dict(terrain=item['name'],height=height,expected=item['top_z'],hit=core.plant.geom(hit).name,
                                   correct=abs(height-item['top_z'])<=1e-10))
            gid = core.plant.geom(item['name']).id
            position_checks.append(abs(float(core.plant.geom_pos[gid,0])-(x0+x1)/2.)<=1e-12)
        ahead_flat_height,flat_hit = metrics.height(core,[first-.05,0.])
        initial_contacts = metrics.contacts(core)
        initial_block_records = sum(row['terrain_name'].startswith('terrain_block_') for row in initial_contacts['contacts'])
        checks = dict(actual1Core_complete_body=(core.plant.nq,core.plant.nv,core.plant.nu,core.plant.nmesh,core.plant.neq)==(49,48,42,71,0),
            one_free_root=int(np.sum(core.plant.jnt_type==mj.mjtJoint.mjJNT_FREE))==1,
            all71opaque=bool(np.all(core.plant.geom_rgba[core.plant.geom_type==mj.mjtGeom.mjGEOM_MESH,3]==1.)),
            original18pad_spheres=len(core.foot_ids)==18,
            derived_XML_is_new_timestamped_source=core.derived_xml.is_relative_to(HERE/'PREPARED') and core.derived_xml.parent.name!='PREPARED',
            new_first_front_exact=abs(first-1.35)<=1e-12,
            initial_all_pad_rays_flat=all(abs(sample['ground_height'])<=1e-10 for leg in initial['pad_ground_samples'] for sample in leg),
            initial_COM_ray_flat=abs(initial['com_ground_height'])<=1e-10,
            initial_no_block_contact_records=initial_block_records==0,
            all_block_height_rays=all(row['correct'] for row in ray_checks),
            compiled_world_block_positions_match_new_ledger=all(position_checks),
            flat_strip_before_new_front=abs(ahead_flat_height)<=1e-10,
            initial_root_untranslated=bool(np.array_equal(core.d.qpos,np.asarray(runtime.read(runtime.OLD_NEURAL/'trials/plane_control/INITIAL_STATE.json')['qpos']))),
            prior147_NN_proof_reused_not_fabricated=True)
        for _ in range(TICKS):
            core.tick(np.zeros(42))
            native_steps += 80
        native = recorder.close()
        checks['all1600_native_steps_recorded'] = native['actual_native_steps']==1600 and native_steps==1600
        checks['short_probe_no_block_contact_records'] = sum(count for name,count in native['native_contacts_by_terrain'].items() if name.startswith('terrain_block_'))==0
        checks['short_probe_no_nonfoot_normal_force'] = native['nonfoot_normal_peak_native']==0
        checks['short_probe_no_external_force'] = native['extra_force_max']==0 and not bool(np.any(core.d.xfrc_applied) or np.any(core.d.qfrc_applied))
        identity = core_module.actual_identity(core)
        save(out/'ACTUAL_MODEL_IDENTITY.json',identity)
        runtime.verify()
        result = dict(kind=runtime.KIND,all_passed=all(checks.values()),checks=checks,
            actual_Core_constructions=1,actual_native_steps=native_steps,NN_forwards=0,
            fresh_fourNN_test_performed=False,reused_fourNN_preflight=str(runtime.OLD_NEURAL/'PREFLIGHT.json'),
            reused_fourNN_preflight_sha256=runtime.sha(runtime.OLD_NEURAL/'PREFLIGHT.json'),
            prior_NN_Core_constructions=0,ray_checks=ray_checks,native=native,
            actual_model_identity=str(out/'ACTUAL_MODEL_IDENTITY.json'),actual_model_identity_sha256=runtime.sha(out/'ACTUAL_MODEL_IDENTITY.json'),
            actual_xml=str(core.derived_xml),actual_xml_sha256=runtime.sha(core.derived_xml),
            scene_repair=runtime.scene_metadata(),sourcefreeze_sha256=runtime.sha(HERE/'FROZEN.json'),
            full_warm_flat_tested=False,new_autonomous_rough_behavior_tested=False)
        save(out/'RESULT.json',result)
        save(HERE/'PREFLIGHT.json',result)
        print(json.dumps(dict(all_passed=result['all_passed'],actual_Core_constructions=1,actual_native_steps=native_steps,NN_forwards=0)))
        if not result['all_passed']:
            raise SystemExit(2)
    except BaseException as error:
        if not isinstance(error,SystemExit):
            save(out/'FAILURE.json',dict(kind=runtime.KIND,error=repr(error),actual_native_steps=native_steps,NN_forwards=0))
        raise


if __name__ == '__main__':
    main()
