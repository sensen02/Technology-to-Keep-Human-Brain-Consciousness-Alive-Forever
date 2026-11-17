"""Parent-only compile brown(noCore/no steps), construct1 grayCore/1600 steps.

Requires every compiled ndarray except geom_rgba to be byte-identical; only six
expected block RGB rows may change. No policy loading, renderer or NN forward.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent


def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-resource-released',action='store_true')
    args = parser.parse_args()
    if not args.parent_resource_released:
        print('SOURCE_ONLY no model/Core/NN/physical execution')
        return
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='1':
            raise ValueError(key+'=1 required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    import runtime
    runtime.verify()
    out = HERE/'ACTUAL_PREFLIGHT'
    out.mkdir(exist_ok=False)
    steps = 0
    try:
        functions = runtime.neural_functions()
        prepared,metrics,core_module = functions['terrain_modules']()
        base = core_module.load_base_physics()
        np,mj = base.np,base.mj
        old_ledger = runtime.read(runtime.BROWN/'PREPARED/MODEL_LEDGER.json')['cases'][runtime.BROWN_CASE]
        brown = base.act_mod.load_full_model(old_ledger['xml'])
        # This is exactly the runtime Core's declared MU override, not a new
        # physical change or brown integration. SourceXML defaults differ.
        brown.geom_friction[:,0] = base.rc.MU
        Derived = core_module.core_type(base)
        recorder = metrics.TerrainPeakRecorder()
        gray = Derived(case=runtime.CASE,recorder=recorder,adhesion=False)
        archived = runtime.read(runtime.BROWN_FOLDER/'IDENTITY.json')
        arrays,checks = {},{}
        for name in dir(brown):
            if name.startswith('_'):
                continue
            before = getattr(brown,name)
            if not isinstance(before,np.ndarray):
                continue
            after = np.asarray(getattr(gray.plant,name))
            bh,gh = hashlib.sha256(before.tobytes()).hexdigest(),hashlib.sha256(after.tobytes()).hexdigest()
            arrays[name] = dict(shape=list(before.shape),dtype=str(before.dtype),brown_sha256=bh,gray_sha256=gh,
                exact=before.shape==after.shape and str(before.dtype)==str(after.dtype) and bh==gh)
            checks['compiled:'+name] = bool(name=='geom_rgba' or arrays[name]['exact'])
            old = archived['compiled_arrays'][name]
            checks['brown_archive_compiled:'+name] = list(before.shape)==old['shape'] and str(before.dtype)==old['dtype'] and bh==old['sha256']
        for name in dir(brown.opt):
            if name.startswith('_'):
                continue
            a,b = getattr(brown.opt,name),getattr(gray.plant.opt,name)
            if isinstance(a,np.ndarray):
                checks['physical_option:'+name] = a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
            elif isinstance(a,(int,float,bool)):
                checks['physical_option:'+name] = a==b
        expected_rows = {brown.geom('terrain_block_'+str(i).zfill(2)).id for i in range(6)}
        different_rows = set(map(int,np.flatnonzero(np.any(brown.geom_rgba!=gray.plant.geom_rgba,axis=1))))
        checks['exact_six_RGB_rows_changed'] = different_rows==expected_rows
        checks['all_alpha_unchanged'] = bool(np.array_equal(brown.geom_rgba[:,3],gray.plant.geom_rgba[:,3]))
        # Mean is prepared from source decimals, not already-rounded float32.
        source_rgba = {g.attrib['name']:g.attrib['rgba'] for g in ET.parse(old_ledger['xml']).getroot().iter('geom') if g.attrib.get('name','').startswith('terrain_block_')}
        import math
        target = {name:math.fsum(map(float,text.split()[:3]))/3. for name,text in source_rgba.items()}
        checks['expected_arithmetic_mean_gray'] = all(np.array_equal(gray.plant.geom_rgba[gray.plant.geom(name).id,:3],
            np.full(3,value,dtype=gray.plant.geom_rgba.dtype)) for name,value in target.items())
        checks['complete71opaque'] = int(gray.plant.nmesh)==71 and bool(np.all(gray.plant.geom_rgba[gray.plant.geom_type==mj.mjtGeom.mjGEOM_MESH,3]==1.))
        checks['free_root_42PD18pads'] = (gray.plant.nq,gray.plant.nv,gray.plant.nu,gray.plant.neq)==(49,48,42,0) and len(gray.foot_ids)==18
        checks['initial_qpos_brown_exact'] = bool(np.array_equal(gray.d.qpos,np.asarray(runtime.read(runtime.BROWN_FOLDER/'INITIAL_STATE.json')['qpos'])))
        m = gray.measure()
        checks['initial_pad_and_COM_rays_flat'] = all(abs(s['ground_height'])<=1e-10 for leg in m['pad_ground_samples'] for s in leg) and abs(m['com_ground_height'])<=1e-10
        initial_contacts = metrics.contacts(gray)
        checks['initial_no_block_contacts'] = not any(row['terrain_name'].startswith('terrain_block_') for row in initial_contacts['contacts'])
        checks['terrain_geometry_ledger_exact'] = gray.terrain_ledger['terrain']==old_ledger['terrain']
        checks['new_actual_XML_not_brown'] = str(gray.derived_xml)!=old_ledger['xml'] and gray.derived_xml.is_relative_to(HERE/'PREPARED')
        checks['gravity_and_timestep_exact'] = bool(np.array_equal(brown.opt.gravity,gray.plant.opt.gravity)) and float(brown.opt.timestep)==float(gray.plant.opt.timestep)
        for _ in range(20):
            gray.tick(np.zeros(42))
            steps += 80
        native = recorder.close()
        checks['all1600_native_steps'] = steps==native['actual_native_steps']==1600
        checks['short_no_block_contacts'] = not any(name.startswith('terrain_block_') and count for name,count in native['native_contacts_by_terrain'].items())
        checks['short_NFnormal_and_worldz_zero'] = native['nonfoot_normal_peak_native']==0 and native['nonfoot_peak_native']==0
        checks['zero_external_force'] = native['extra_force_max']==0
        save(out/'ACTUAL_MODEL_IDENTITY.json',core_module.actual_identity(gray))
        runtime.verify()
        result = dict(kind=runtime.KIND,all_passed=all(checks.values()),checks=checks,
            compiled_array_comparison=arrays,excluded_visual_arrays=['geom_rgba'],actual_Core_constructions=1,
            brown_model_compile_count=1,brown_physical_steps=0,gray_native_steps=steps,NN_forwards=0,actual_render_tested=False,
            native=native,color_control=runtime.color_metadata(),actual_xml=str(gray.derived_xml),actual_xml_sha256=runtime.sha(gray.derived_xml),
            actual_model_identity_sha256=runtime.sha(out/'ACTUAL_MODEL_IDENTITY.json'),sourcefreeze_sha256=runtime.sha(HERE/'FROZEN.json'),
            full5800_warm_physical_parity_tested=False,new_autonomous_behavior_tested=False)
        save(out/'RESULT.json',result)
        save(HERE/'PREFLIGHT.json',result)
        print(json.dumps(dict(all_passed=result['all_passed'],compiled_arrays=len(arrays),gray_native_steps=steps,brown_physical_steps=0)))
        if not result['all_passed']:
            raise SystemExit(2)
    except BaseException as error:
        if not isinstance(error,SystemExit):
            save(out/'FAILURE.json',dict(kind=runtime.KIND,error=repr(error),gray_native_steps=steps,brown_physical_steps=0))
        raise


if __name__=='__main__':
    main()
