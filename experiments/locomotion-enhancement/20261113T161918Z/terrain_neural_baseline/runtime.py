"""Explicit source-neck four-NN terrain namespace. Stdlib until parent run flag.

Frozen upstream load_policies keeps its ORIGINAL globals/ROOT and reads the
actual064313 binding. New ROOT is used only by the derived runner's artifacts.
No original module/class/global is patched. Physics is imported only by pin(),
which the frozen loop calls AFTER all four policies have loaded.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import types

HERE = Path(__file__).resolve().parent
PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
UPSTREAM_ROOT = PROJECT/'experiments/neck-real-structure/20261113T064313Z/sensory_body_repair'
LOOP_ROOT = PROJECT/'experiments/neck-real-structure/20261113T045302Z/sensory_body'
TERRAIN_ROOT = HERE.parent/'terrain_body_baseline'
KIND = 'SOURCE_NECK_ROUGH_TERRAIN_BASELINE_V1'
CASES = ('plane_control','flat_box_control','mild_blocks_seed17')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    item = importlib.util.module_from_spec(spec)
    sys.modules[name] = item
    spec.loader.exec_module(item)
    return item


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify():
    value = read(HERE/'FROZEN.json')
    for path, digest in value['source_hashes'].items():
        if sha(path) != digest:
            raise ValueError('Terrain neural frozen source drift '+path)
    return value


def upstream():
    return module('_terrain_neural_exact_064313_runtime',UPSTREAM_ROOT/'runtime.py')


def terrain_modules():
    """Load frozen terrain modules with explicit temporary local import aliases."""
    absent = object()
    prior = {name:sys.modules.get(name,absent) for name in ('prepare','terrain_metrics')}
    try:
        prepared = module('_terrain_neural_frozen_prepare',TERRAIN_ROOT/'prepare.py')
        sys.modules['prepare'] = prepared
        metrics = module('_terrain_neural_frozen_contact_metrics',TERRAIN_ROOT/'terrain_metrics.py')
        sys.modules['terrain_metrics'] = metrics
        core = module('_terrain_neural_frozen_core',TERRAIN_ROOT/'terrain_core.py')
    finally:
        for name,value in prior.items():
            if value is absent:
                sys.modules.pop(name,None)
            else:
                sys.modules[name] = value
    return prepared,metrics,core


def digest_identity(value):
    canonical = {key:item for key,item in value.items() if key != 'identity_sha256'}
    return hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def context(case, *, audit=False):
    if case not in CASES:
        raise ValueError(case)
    old = upstream()
    derived_metrics = module('_terrain_neural_versioned_summary',HERE/'metrics.py')
    state = dict(case=case, old=old, rows=[], initial=None, core=None, extra_arrays=None)
    namespace = {key:value for key,value in vars(old).items() if not key.startswith('_')}
    namespace.update(ROOT=HERE,KIND=KIND,SUMMARY_KIND='SOURCE_NECK_ROUGH_TERRAIN_BASELINE_SUMMARY_V1',
        IDS=('V_empty',),TERRAIN_CASE=case,verify=verify)
    # Preserve this function object exactly: its globals are the064313 namespace.
    namespace['load_policies'] = old.load_policies

    def stable(auto,cycles,native):
        # Preserve the old formula separately; new unqualified functional flag
        # additionally rejects pure horizontal nonfoot side-wall contacts.
        return bool(old.stable(auto,cycles,native) and native['post_nonfoot_normal_peak_native']==0)

    namespace['stable'] = stable

    def identity_verify(value):
        if value['identity_sha256'] != digest_identity(value):
            raise ValueError('Actual derived identity digest mismatch')
        for path,digest in value['source_hashes'].items():
            if sha(path) != digest:
                raise ValueError('Actual derived identity source drift '+path)
        for name,version in value['package_versions'].items():
            if importlib.metadata.version(name) != version:
                raise ValueError('Runtime package version changed '+name)
        return True

    def identity_capture(core, config, extra_paths=()):
        value = state['terrain_core_module'].actual_identity(core)
        files = dict(value['source_hashes'])
        files.update(verify()['source_hashes'])
        files.update({str(Path(path).resolve()):sha(path) for path in extra_paths})
        value.update(source_hashes=files,config=dict(config,terrain_case=case,
            upstream_neural_condition='V_empty',new_calibration_claim=False),
            package_versions={name:importlib.metadata.version(name) for name in ('numpy','mujoco','flygym','torch')},
            executable=sys.executable,physical_neural_scope='actual derived terrain XML + unchanged four policies/source neck',
            full_NN_pipeline_ready=True)
        value['identity_sha256'] = digest_identity(value)
        return value

    def identity_runtime(value, core):
        identity_verify(value)
        np = core._terrain_np
        for name,expected in value['compiled_arrays'].items():
            actual = np.asarray(getattr(core.plant,name))
            if list(actual.shape) != expected['shape'] or str(actual.dtype) != expected['dtype'] or hashlib.sha256(actual.tobytes()).hexdigest() != expected['sha256']:
                raise ValueError('Actual derived compiled parameter drift '+name)
        return True

    def pin():
        sw,unused_identity = old.pin()
        prepared,terrain_metrics,core_module = terrain_modules()
        Derived = core_module.core_type(sys.modules['physics'])
        state.update(sw=sw,terrain_core_module=core_module,terrain_metrics=terrain_metrics)

        class Recorder(terrain_metrics.TerrainPeakRecorder):
            def __init__(self):
                super().__init__()
                self.post_normal_peak = 0.
                self.warm_nonzero_height_count = 0
                self.post_terrain_counts = {}
                self.warm_terrain_counts = {}

            def after(self,core,before,solved):
                super().after(core,before,solved)
                top = {item['name']:float(item['top_z']) for item in core.terrain_ledger['terrain']}
                if 1.-1e-8 <= before[0] < 5.8-1e-8:
                    for row in solved['contacts']:
                        name = row['terrain_name']
                        self.warm_terrain_counts[name] = self.warm_terrain_counts.get(name,0)+1
                        self.warm_nonzero_height_count += abs(top[name]) > 1e-10
                if before[0] >= 5.8-1e-8:
                    self.post_normal_peak = max(self.post_normal_peak,solved['nonfoot_normal'])
                    for row in solved['contacts']:
                        name = row['terrain_name']
                        self.post_terrain_counts[name] = self.post_terrain_counts.get(name,0)+1

            def close(self):
                value = super().close()
                value.update(post_nonfoot_normal_peak_native=self.post_normal_peak,
                    warm_nonzero_height_contact_count=self.warm_nonzero_height_count,
                    warm_native_contacts_by_terrain=self.warm_terrain_counts,
                    post_native_contacts_by_terrain=self.post_terrain_counts)
                return value

        state['Recorder'] = Recorder

        def core_factory(*,adhesion=False,recorder=None):
            core = Derived(case=case,adhesion=adhesion,recorder=recorder)
            state['core'] = core
            state['native_recorder'] = recorder
            state['initial'] = derived_metrics.capture(core,0)
            return core

        class Trace:
            def __init__(self,full=True):
                self.original = sw.Trace(full=full)

            def __getattr__(self,name):
                return getattr(self.original,name)

            def tick(self,core,action,source):
                value = self.original.tick(core,action,source)
                state['rows'].append(derived_metrics.capture(core,source))
                return value

            def save(self,path):
                import numpy as np
                value = self.original.save(path)
                rows = state['rows']
                state['extra_arrays'] = {key:np.asarray([row[key] for row in rows]) for key in rows[0]}
                extra_path = Path(path).with_name('terrain_1ms.npz')
                np.savez_compressed(extra_path,**state['extra_arrays'])
                old.save(Path(path).with_name('INITIAL_TERRAIN.json'),state['initial'])
                return value

        def continuity_snapshot(core,previous,hidden):
            if not state.get('warm_flat_checked'):
                import numpy as np
                warm = state['rows'][1000:5800]
                if len(state['rows']) != 5800 or len(warm) != 4800:
                    raise ValueError('Warm flat check requires complete5800ms prefix')
                checks = dict(all_pad_rays_zero=all(np.all(np.abs(row['pad_ground_height'])<=1e-10) for row in warm),
                    all_COM_rays_zero=all(abs(row['com_ground_height'])<=1e-10 for row in warm),
                    all_native_warm_contacts_zero_height=state['native_recorder'].warm_nonzero_height_count==0)
                old.save(HERE/'trials'/case/'WARM_FLAT_CHECK.json',dict(terrain_case=case,
                    checks=checks,all_passed=all(checks.values()),checked_before_ANN=True,no_state_reset=True))
                state['warm_flat_checked'] = True
                if not all(checks.values()):
                    raise ValueError('Terrain reached during frozen world-z warm teacher; ANN phase not admitted')
            return sw.continuity_snapshot(core,previous,hidden)

        proxy = types.SimpleNamespace(**vars(sw))
        proxy.Core,proxy.Trace = core_factory,Trace
        proxy.continuity_snapshot = continuity_snapshot
        ident = types.SimpleNamespace(capture=identity_capture,verify=identity_verify,
            verify_runtime=identity_runtime,atomic_json=old.save)
        return proxy,ident

    def module_override(name,path):
        if Path(path).resolve() == old.PEAK_SOURCE:
            if 'Recorder' not in state:
                raise ValueError('Terrain recorder requested before explicit physical pin')
            return types.SimpleNamespace(PeakRecorder=state['Recorder'])
        return old.module(name,path)

    def summarize(np,extra,native,ledger,cycles):
        value = derived_metrics.summarize(np,extra,native,ledger)
        complete = bool(cycles['all_legs_completed'])
        value['contact_metric_flags_before_cycle_gate'] = {name:value[name]
            for name in ('behavior_diagnostic_pass','rough_terrain_pass','all_pass')}
        value['all_six_leg_cycles_completed'] = complete
        value['functional_scope_version'] = 'TERRAIN_METRICS_PLUS_COMPLETE_SIX_LEG_CYCLES_V1'
        for name in ('behavior_diagnostic_pass','rough_terrain_pass','all_pass'):
            value[name] = bool(value[name] and complete)
        return value

    def save(path,value):
        path = Path(path)
        folder = HERE/'trials'/case
        if path == folder/'RESULT.json':
            import numpy as np
            extra = state['extra_arrays']
            if extra is None:
                raise ValueError('Missing saved terrain trace')
            summary = summarize(np,extra,value['native'],state['core'].terrain_ledger,value['autonomous_cycles'])
            value.update(terrain_case=case,terrain_summary=summary,
                legacy_allterrain_worldz_xy_autonomous=value['autonomous'],
                legacy_functional_stable_body=old.stable(value['autonomous'],value['autonomous_cycles'],value['native']),
                functional_stable_body_scope='legacy support/worldCOM/forward/cycles plus BOTH post-native world-z and contact-normal nonfoot zero',
                terrain_relative_functional_stable_body=bool(summary['behavior_diagnostic_pass'] and value['autonomous_cycles']['all_legs_completed']),
                terrain_observation_version=state['core'].observation_version,
                terrain_metric_version=state['core'].metric_version,
                source_neural_condition='V_empty',new_calibration_claim=False,
                real_terrain_RGB_used_by_vision=True,terrain_visual_input_forced_zero=False,
                terrain_trace=dict(path=str(folder/'terrain_1ms.npz'),sha256=sha(folder/'terrain_1ms.npz'),rows=len(state['rows'])),
                initial_terrain=dict(path=str(folder/'INITIAL_TERRAIN.json'),sha256=sha(folder/'INITIAL_TERRAIN.json')),
                warm_flat_proof=dict(path=str(folder/'WARM_FLAT_CHECK.json'),sha256=sha(folder/'WARM_FLAT_CHECK.json')),
                actual_xml=str(state['core'].derived_xml),actual_xml_sha256=sha(state['core'].derived_xml),
                legacy_five_gates_meaning='registered-terrain world-z support/XY slip/world-COM; diagnostic old formulas, not original frozen-world acceptance',
                original_D1_D8_new_acceptance_claim=False,mechanical_hook_or_pulvilli_claim=False)
        elif path == folder/'VERIFY.json':
            import numpy as np
            result = read(folder/'RESULT.json')
            checks = value['checks']
            for name,key in (('terrain_1ms.npz','terrain_trace'),('INITIAL_TERRAIN.json','initial_terrain'),('WARM_FLAT_CHECK.json','warm_flat_proof')):
                checks['terrain_artifact:'+name] = sha(folder/name) == result[key]['sha256']
            extra = old.arrays(folder/'terrain_1ms.npz')
            trace = old.arrays(folder/'trace_1ms.npz')
            decisions = old.arrays(folder/'neural_decisions.npz')
            for added,original in (('time','time'),('com_world_xyz','com'),('padz','padz'),
                ('loads_world_z_positive','loads'),('nonfoot_world_z_positive','nonfoot'),
                ('control_source','control_source')):
                checks['terrain_trace_tied_to_original:'+added] = bool(np.array_equal(extra[added],trace[original]))
            ledger = read(TERRAIN_ROOT/'PREPARED/MODEL_LEDGER.json')['cases'][case]
            geometry_check = derived_metrics.audit_observation(np,extra,decisions,read(folder/'INITIAL_TERRAIN.json'),terrain_ledger=ledger)
            # Contract returns {checks, diagnostics}; boolean checks are only gates.
            geometry_checks = geometry_check['checks'] if 'checks' in geometry_check else geometry_check
            for name,flag in geometry_checks.items():
                if isinstance(flag,(bool,np.bool_)):
                    checks['terrain_geometry:'+name] = bool(flag)
            summary = summarize(np,extra,result['native'],ledger,result['autonomous_cycles'])
            checks['terrain_summary_exact'] = old.jsonable(summary) == result['terrain_summary']
            ident = read(folder/'IDENTITY.json')
            checks['actual_terrain_case'] = result['terrain_case'] == case and result['actual_xml'] == ledger['xml'] and result['actual_xml_sha256'] == sha(ledger['xml'])
            checks['identity_case_and_actual_XML_tied_to_ledger'] = (ident['case']==case and ident['xml']==ledger['xml']
                and ident['xml_sha256']==ledger['xml_sha256']==sha(ledger['xml'])
                and ident['config']['terrain_case']==case and ident['config']['upstream_neural_condition']=='V_empty')
            checks['identity_pad_count_and_mask_disjoint'] = len(ident['foot_ids'])==18 and bool(ident['terrain_ids']) and not bool(set(ident['terrain_ids']) & set(ident['foot_ids']))
            checks['terrain_trace_complete'] = extra['time'].shape == (15400,)
            checks['terrain_trace_numeric_finite'] = all(np.isfinite(array).all() for array in extra.values())
            checks['original_legacy_autonomous_preserved'] = result['legacy_allterrain_worldz_xy_autonomous'] == result['autonomous']
            checks['legacy_functional_flag_preserved_separately'] = result['legacy_functional_stable_body'] == old.stable(result['autonomous'],result['autonomous_cycles'],result['native'])
            checks['new_functional_includes_native_normal_NF_zero'] = result['functional_stable_body'] == stable(result['autonomous'],result['autonomous_cycles'],result['native'])
            checks['relative_functional_flag_exact'] = result['terrain_relative_functional_stable_body'] == bool(summary['behavior_diagnostic_pass'] and result['autonomous_cycles']['all_legs_completed'])
            warm_proof = read(folder/'WARM_FLAT_CHECK.json')
            checks['warm_flat_checked_before_ANN'] = warm_proof['all_passed'] and all(warm_proof['checks'].values()) and warm_proof['checked_before_ANN'] and summary['warmflat']['covered']
            if not all(checks.values()):
                raise ValueError('Independent terrain audit failed '+str([name for name,flag in checks.items() if not flag]))
            value.update(terrain_case=case,terrain_observation_checks=old.jsonable(geometry_check),
                terrain_geometry_and_summary_replayed=True,old_warm_reproduction_required=case=='plane_control',
                new_physics_steps=0,Core_constructions=0)
        elif path == folder/'FAILURE.json':
            value.update(terrain_case=case,source_neural_condition='V_empty',new_calibration_claim=False)
            value['terrain_partial_artifacts'] = {name:dict(path=str(folder/name),sha256=sha(folder/name))
                for name in ('terrain_1ms.npz','INITIAL_TERRAIN.json','WARM_FLAT_CHECK.json') if (folder/name).is_file()}
        return old.save(path,value)

    namespace.update(pin=pin,module=module_override,save=save)
    return namespace,state
