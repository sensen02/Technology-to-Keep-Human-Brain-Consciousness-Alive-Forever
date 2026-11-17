"""Static source/schema proof; stdlib only, never imports the diagnostic module."""
from pathlib import Path
import ast
import hashlib
import json

HERE=Path(__file__).resolve().parent

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    binding=json.loads((HERE/'SOURCE_SCHEMA_BINDINGS.json').read_text())
    prereg=json.loads((HERE/'PREREG.json').read_text())
    checks={};texts={};trees={}
    for name in ('bind_schema.py','diagnose_saved.py','check_codeonly.py'):
        texts[name]=(HERE/name).read_text();trees[name]=ast.parse(texts[name],filename=str(HERE/name));compile(trees[name],str(HERE/name),'exec')
    checks['three_sources_AST_compile_only']=True
    checks['source_metadata_hashes_current']=all(sha(p)==digest for p,digest in binding['actual_sources_sha256'].items())
    checks['recorded_original_sources_all_match']=all(h['current_sha256']==h['recorded_sha256'] for c in binding['cases'] for h in c['source_hashes'].values())
    checks['actual_header_only_preparation']=binding['only_header_arrays'] is True
    checks['long44200_V2_25000']= [c['archives']['trace_1ms.npz']['headers']['time']['shape'] for c in binding['cases']]==[[44200],[25000]]
    checks['neural8840_V2_5000']=[c['archives']['neural_decisions.npz']['headers']['time']['shape'] for c in binding['cases']]==[[8840],[5000]]
    checks['all_geometry_reference_hashes_exact']=all(all(c['geometry_reference_equal'].values()) for c in binding['cases'])
    checks['native_dt12p5us']=all(abs(c['xml_schema']['native_dt_s']-12.5e-6)<1e-15 for c in binding['cases'])
    checks['Thorax_world_root_source_proof']=all(c['xml_schema']['root_point_is_Thorax_frame_origin'] for c in binding['cases'])
    checks['first_edge_original_1p35_p02_p04']=all(abs(c['xml_schema']['blocks'][0]['front_x']-1.35)<1e-12 and c['xml_schema']['blocks'][0]['center_z']==.02 and c['xml_schema']['blocks'][0]['top_z']==.04 for c in binding['cases'])
    checks['original18sphere_noadhesion']=all(c['xml_schema']['foot_spheres']==18 and not c['xml_schema']['adhesion'] for c in binding['cases'])
    checks['preregistered_four_equal_windows']=prereg['window_edges_seconds']==[5.8,15.4,25.,34.6,44.2] and prereg['preaction_index']==5799
    imports=[(n.module or '').split('.')[0] for n in ast.walk(trees['diagnose_saved.py']) if isinstance(n,ast.ImportFrom)]
    imports += [a.name.split('.')[0] for n in ast.walk(trees['diagnose_saved.py']) if isinstance(n,ast.Import) for a in n.names]
    checks['imports_only_stdlib_plus_deferred_numpy']=set(imports)<= {'pathlib','argparse','hashlib','json','resource','signal','sys','time','numpy'}
    checks['no_module_scope_numpy']=not any(isinstance(n,ast.Import) and any(a.name=='numpy' for a in n.names) for n in trees['diagnose_saved.py'].body)
    checks['no_numpy_in_header_binder']='numpy' not in {a.name for n in ast.walk(trees['bind_schema.py']) if isinstance(n,ast.Import) for a in n.names}
    code=texts['diagnose_saved.py']
    checks['explicit_twoflag_parent_guard']='if not (args.execute_parent_saved_arrays and args.parent_resource_released):' in code
    checks['exact125712_environment']="20261110T125712Z/env/bin/python" in code
    checks['budget90s1500MiB']='signal.setitimer(signal.ITIMER_REAL,90)' in code and '1500*1024**2' in code and 'rss<=1500' in code
    checks['bound_archive_SHA_before_load']="require(sha(path)==spec['recorded_sha256']" in code
    checks['allow_pickle_false']='np.load(path,allow_pickle=False)' in code
    checks['every_field_shape_dtype_finite']='a.dtype.str' in code and 'np.isfinite(a).all()' in code
    checks['full_window_mass_and_no_partial']='NOT_RECORDED_FULL_WINDOW' in code and 'hi-lo==9600' in code and 'int(neural.sum())==1920' in code and 'n-5800' in code
    checks['actual_boundary_and_postsample_duration']='boundary_duration_actual_s' in code and 'postsample_endpoint_duration_actual_s' in code
    checks['target_minus_q_source_order']="t['target']-t['qpos'][:,qadr]" in code and "struct.pack('<84i'" in texts['bind_schema.py'] and "struct.pack('<86d'" in texts['bind_schema.py']
    checks['ctrl_force_recorded_endpoint_only']='actualctrl_native_1ms_endpoints' in code and 'actuator_force_native_1ms_endpoints' in code and "native_substep_tau_peak='NOT_RECORDED'" in code
    checks['front_facing_sphere_gap']="gap=b['front_x']-(centers[:,:,0]+.008)" in code
    checks['no_block_contact_window_fabrication']="per_block_window_contact='NOT_RECORDED" in code and "per_block_native_window_split='NOT_RECORDED'" in code
    checks['slip_original_registered_separate']='orig_autonomous=' in code and 'reference_registered=' in code
    checks['slip_numerator_net_path_denominators_and_null']='numerator_loadweighted_tangent_speed_native_per_s' in code and 'denominator_net_COM_xy_speed_native_per_s' in code and 'denominator_path_COM_xy_speed_native_per_s' in code and 'clamping_applied=False' in code
    checks['strict_original_cycle_events_preserved']='original_strict_cycles_saved=' in code and 'original_strict_cycle_complete_count=' in code
    checks['component_cycles_explicitly_separate']='per_leg_component_cycle_events' in code and 'original .03advance and4stimeout' in code
    checks['zero_updates_Core_NN_integration_render']='optimizer_updates=0,Core_constructions=0,NN_forwards=0,physical_integration_steps=0,render_calls=0' in code
    checks['independent_review_present']=(HERE/'INDEPENDENT_STATIC_REVIEW.md').exists()
    if not all(checks.values()):
        raise ValueError('Static failures: '+str([k for k,v in checks.items() if not v]))
    output=dict(status='CODE_READY_NUMERICS_NOT_EXECUTED',checks=checks,all_passed=True,check_count=len(checks),actual_sources_hash=binding['actual_sources_sha256'],preparation_execution=dict(NPZ_payload_reads=0,NumPy_imports=0,Torch_imports=0,models_loaded=0,Core=0,NN=0,physics=0,render=0,optimizer_updates=0),runtime_budget_verified=False,behavior_or_unique_root_cause_claim=False)
    with (HERE/'CODECHECK.json').open('x') as f:
        json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=output['status'],checks=len(checks),all_passed=True)))

if __name__=='__main__':
    main()
