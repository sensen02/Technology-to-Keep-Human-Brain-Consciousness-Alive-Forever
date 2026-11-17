"""Targeted stdlib-only failed-render fix proof; original CODEONLY ledger unchanged."""
from pathlib import Path
import ast,json,sys,hashlib
HERE=Path(__file__).resolve().parent
LIB=Path('/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/mujoco/rendering/classic/renderer.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 source=(HERE/'render_source.py').read_text();tree=ast.parse(source);compile(tree,str(HERE/'render_source.py'),'exec');checks={}
 checks['model_segmentation_before_markers']=source.index('renderer.enable_segmentation_rendering()')<source.index('for j in range(32):')<source.index('rgb=renderer.render()')
 checks['one_segmentation_pass_only']=source.count('renderer.enable_segmentation_rendering()')==source.count('renderer.disable_segmentation_rendering()')==1
 checks['markers_explicitly_excluded_from_segmentation']='geom.objid=-1;geom.segid=-1;' in source and 'geom.objtype=int(mj.mjtObj.mjOBJ_UNKNOWN)' in source and 'geom.category=int(mj.mjtCatBit.mjCAT_DECOR)' in source
 library=LIB.read_text();checks['actual_renderer_convention_confirmed']='if g.segid != -1' in library and 'g.segid + 1' in library
 calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='mj']
 checks['zero_integration_call_unchanged']=not any(x in calls for x in ('mj_step','mj_step1','mj_step2')) and 'mj_forward' in calls
 checks['full_body_timing_runtime_checks_retained']='range(0,15400,20)' in source and 'count==770' in source and "verify_runtime(data['identity'],c)" in source and "mesh.sum()==71" in source
 checks['no_heavy_imports']=not any(n in sys.modules for n in ('numpy','torch','mujoco','physics','matplotlib'))
 result=dict(status='RENDER_SEGID_FIX_CODE_ONLY_READY',all_passed=all(checks.values()),checks=checks,source_hashes={str(HERE/'render_source.py'):sha(HERE/'render_source.py'),str(HERE/'check_fix.py'):sha(HERE/'check_fix.py')},renderer_library=str(LIB),renderer_library_sha256=sha(LIB),original_codeonly_ledger_retained=str(HERE/'CODEONLY_CHECK.json'),physical_inputs_modified=False,Core_constructions=0,physics_steps=0,render_frames=0,model_loads=0,array_loads=0)
 with (HERE/'CODEONLY_CHECK_FIX.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result,indent=2));assert all(checks.values())
if __name__=='__main__':main()
