"""V2 stdlib-only guard/namespace validation, then inherited zeroCore preparation."""
import ast
import inspect
import sys
import textwrap
import runtime as r

def main():
    if (r.HERE/'READY.json').exists():
        r.verify()
        print(str(r.HERE/'READY.json'))
        return
    r.base.verify()
    v1 = r.V1
    failed = v1/'trials'/r.CASE
    failure = r.read(failed/'FAILURE.json')
    if failure['error']!="ValueError('captureexact20msindices')" or failure['artifacts']['trace']['samples']!=15400:
        raise ValueError('Actual preserved capture-horizon15400 failure required')
    if (failed/'RESULT.json').exists():
        raise ValueError('V1 must remain failed partial, not full long result')
    # Load only stdlib namespace definitions; no camera/Core/arrays constructed.
    namespace,state = r.context(audit=True)
    old_camera = state['old'].camera_runtime()
    capture = old_camera['_capture']
    original_path = r.Path(capture.__code__.co_filename.split('::')[0])
    original_ast = ast.parse(original_path.read_text())
    cls = next(node for node in original_ast.body if isinstance(node,ast.ClassDef) and node.name=='CameraStream')
    function = next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='capture')
    duration = [node for node in ast.walk(function) if isinstance(node,ast.Constant) and node.value==15400]
    if len(duration)!=1:
        raise ValueError('Unique original15400 capture limit required')
    source = textwrap.dedent(ast.get_source_segment(original_path.read_text(),function))
    tree,changes = r.scheduled_tree(source,original_path)
    if len(changes)!=1:
        raise ValueError('Exactly one symbolic change required')
    # Evaluate just guard expression, never invoke capture or import its backend.
    guard = next(node.test for node in tree.body[0].body if isinstance(node,ast.If))
    guard_code = compile(ast.Expression(guard),str(original_path)+'::guard_only','eval')
    decisions = {str(ms):bool(eval(guard_code,dict(r.SCHEDULE,ms=ms))) for ms in (0,15380,15400,24980,25000,15401)}
    if decisions!={'0':False,'15380':False,'15400':False,'24980':False,'25000':True,'15401':True}:
        raise ValueError('Guard boundaries failed')
    camera_cls = old_camera['CameraStream']
    if camera_cls.capture.__globals__['_capture'] is not capture or capture.__globals__['TOTAL_MS']!=25000:
        raise ValueError('Actual nested far-camera capture routing not replaced')
    if any(name.split('.')[0] in {'numpy','torch','mujoco','flygym','physics','student_wave'} for name in sys.modules):
        raise ValueError('Heavy import during CODEONLY')
    # Hash binary archives as opaque bytes only; never read array payloads.
    extra = dict(r.read(v1/'FROZEN.json')['source_hashes'])
    extra.update({str(path):r.sha(path) for path in (v1/'FROZEN.json',v1/'READY.json',v1/'JOBS.json',v1/'trial.log',original_path)})
    extra.update({str(path):r.sha(path) for path in failed.glob('*') if path.is_file()})
    for name,item in failure['artifacts'].items():
        if r.sha(item['path'])!=item['sha256']:
            raise ValueError('Failed artifact SHA '+name)
    r.save(r.HERE/'CAPTURE_HORIZON_PROOF.json',dict(kind=r.KIND,
        preserved_V1_failure=failure['error'],preserved_V1_samples=15400,
        V1_status='ENGINEERING_CAPTURE_HORIZON_FAIL_NOT_FULL_LONG_RESULT',
        original_source=str(original_path),original_line=duration[0].lineno,
        original_value=15400,new_expression='TOTAL_MS',new_value=25000,
        inverse_full_original_capture_AST_exact=True,actual_nested_camera_namespace_routed=True,
        guard_invalid_by_ms=decisions,real_RGB_capture_continues_every20ms=True,
        invalid_zero_fake_RGB_fallback=False,physical_parameters_changed=False,
        Core_constructions=0,NN_forwards=0,array_payload_loads=0,physics_steps=0,
        extra_source_hashes=extra))
    compiled = {str(path):bool(compile(path.read_text(),str(path),'exec')) for path in r.HERE.glob('*.py')}
    r.save(r.HERE/'CODECHECK.json',dict(kind=r.KIND,all_passed=all(compiled.values()),compiled=compiled,
        checks=dict(capture_inverse_AST_exact=True,exactly_one_horizon_symbol=True,
            actual_nested_camera_capture_routed=True,guard_boundaries_exact=True,
            V1_failure_preserved=True,V1_archived_payload_SHA_verified=True,
            no_heavy_imports=True),Core_constructions=0,NN_forwards=0,array_payload_loads=0,physics_steps=0))
    module = r.load('_frozen_v1_zeroCore_preflight',v1/'preflight.py')
    module.main()
    r.verify()

if __name__=='__main__':
    main()
