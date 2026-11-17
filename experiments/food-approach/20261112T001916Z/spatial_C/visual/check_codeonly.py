"""Small stdlib AST/path check; no actual trial or numerical imports."""
from pathlib import Path
import ast,json,sys,runpy
import contracts_visual as c
HERE=Path(__file__).resolve().parent

def main():
 checks={};names=('contracts_visual.py','plot_paths.py','render_source.py','check_codeonly.py')
 for name in names:
  p=HERE/name;tree=ast.parse(p.read_text());compile(tree,str(p),'exec');imports=[]
  for n in tree.body:
   if isinstance(n,ast.Import):imports.extend(a.name for a in n.names)
   elif isinstance(n,ast.ImportFrom):imports.append(n.module)
  checks[name+'_syntax_light']=not any(v and v.split('.')[0] in ('numpy','torch','mujoco','physics','matplotlib','runtime','spatial') for v in imports)
 checks['CROOT_HEAD_BASE_paths']=c.ROOT==HERE.parent and c.HEAD==Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json') and c.BASE==Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
 text=(HERE/'contracts_visual.py').read_text();video=(HERE/'render_source.py').read_text();plot=(HERE/'plot_paths.py').read_text()
 checks['spatial_audit_not_Bsummary']='source_fixed_world_after_handover' in text and 'C_DEVELOPMENT_SUMMARY.json' in text and 'B_FUNCTIONAL_SUMMARY.json' not in text
 checks['plot_new_Cnoodor_relative_eachsource']="baseline=allrows['C_noodor']" in plot and 'np.linalg.norm(bp-center,axis=1)' in plot
 checks['source_marker_scene_only']='mj.mjv_initGeom' in video and 'mj.mjv_connector' in video and 'scene.ngeom+=1' in video and 'model.geom_pos[' not in video
 tree=ast.parse(video);calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='mj']
 checks['zero_integration_calls']=not any(v in calls for v in ('mj_step','mj_step1','mj_step2')) and 'mj_forward' in calls
 checks['original_failure_and_passthrough_shown']="r['engineering_pass']" in video and 'overshoot_or_pass_through' in video and 'NOT parking or ingestion' in video
 for name in ('plot_paths.py','render_source.py'):runpy.run_path(str(HERE/name),run_name='_spatial_visual_codeonly_inspection')
 checks['heavy_imports_absent']=all(v not in sys.modules for v in ('numpy','torch','mujoco','physics','matplotlib','runtime','spatial'))
 result=dict(status='SPATIAL_C_VISUAL_CODE_ONLY_READY',all_passed=all(checks.values()),checks=checks,source_hashes={str(HERE/n):c.sha(HERE/n) for n in names+('PREREG.json',)},model_loads=0,array_loads=0,Core_constructions=0,physics_steps=0,render_frames=0,plot_calls=0,trial_reads=0)
 c.save(HERE/'CODEONLY_CHECK.json',result);print(json.dumps(result,indent=2));assert all(checks.values())
if __name__=='__main__':main()
