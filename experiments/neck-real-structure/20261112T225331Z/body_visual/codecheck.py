"""Stdlib code-only deliverable schema/entry checks; no arrays or trial data opened."""
from pathlib import Path
import ast,json,hashlib,sys,importlib.util
HERE=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 checks={};files=('contracts.py','render.py','plot_sync.py','codecheck.py');trees={}
 for name in files:trees[name]=ast.parse((HERE/name).read_text());checks['syntax_'+name]=True
 pr=json.loads((HERE/'PREREG.json').read_text());checks['sixconditions']=len(pr['conditions'])==6;checks['twoactualvideos']=pr['representative_videos']==['P_left_pulse','P_left_DN_cut'];checks['no_newphysics']=pr['new_physics_steps']==pr['new_training_updates']==0
 for name in ('render.py','plot_sync.py'):
  checks['public_main_'+name]=any(isinstance(n,ast.FunctionDef) and n.name=='main' for n in trees[name].body);checks['explicit_flag_'+name]=any(isinstance(n,ast.Constant) and n.value=='--execute-after-parent-release' for n in ast.walk(trees[name]))
 spec=importlib.util.spec_from_file_location('_body_visual_codeonly_contracts',HERE/'contracts.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
 checks['newbodypath']=str(c.ROOT).endswith('/20261112T225331Z/body_pilot');checks['newbrain_neck_motor_identity']=c.BRAIN_SHA.startswith('eed1ad3c') and c.NECK_SHA.startswith('9e88a04d') and c.MOTOR_SHA.startswith('789f1a31');checks['source_originalrenderer_exists']=c.ORIGINAL_RENDER.is_file();checks['no_heavy_imports']=not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','model','student_wave'))
 render=(HERE/'render.py').read_text();contract=(HERE/'contracts.py').read_text();plot=(HERE/'plot_sync.py').read_text()
 checks['actual3080neuralrows_guard']='d[\'X238\'].shape==(3080,238)' in contract;checks['actual272chain_guard']='(3080,272)' in contract and 'np.array_equal(d[\'full_hidden_before\'][1:]' in contract;checks['all6audits13stages_guard']='len(jobs)==13' in contract and 'all(j[\'exit_code\']==0 for j in jobs)' in contract;checks['no_cameraRGB_or_decor_source']='capture_rgb.npz' not in render and 'render_eye_pair' not in render and 'add_decor' not in render;checks['sidepanel_uses_actualheldports']="d[direction+'_held'][qi]" in render and "d[direction+'_capture_step'][qi]" in render and "d['physical_step_ms'][qi]" in render;checks['nominalfloor_neuralindex']='qi=min(3079,physical_ms//5)' in render;checks['phase_from_actual_controlsource']="t['control_source'][i]" in render;checks['sourcecut_notturngate_label']='source silence + receiver0; no brain/motor reset' in render;checks['all770_decode_required']='decode_exit==0 and count==770' in render;checks['71opaque_containment']='all_frame_fullbody_containment=True' in render and 'model.geom_rgba[mesh,3]==1' in render;checks['physics_steps_patched_forbidden']='mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden' in render;checks['compiled_restored_before_verification']='model.vis.global_.offwidth,model.vis.global_.offheight=oldbuffer' in render;checks['plots_all6_and_matchedcuts']='plt.subplots(6,3' in plot and 'MATCHED_DN_AN_CUT_COMPARISONS.png' in plot
 if not all(checks.values()):raise ValueError(str({k:v for k,v in checks.items() if not v}))
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(HERE/n):sha(HERE/n) for n in files+('PREREG.json',)},actual_arrays_loaded=False,model_loads=0,Core_constructions=0,render_calls=0,training_updates=0,physics_steps=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False,actualtrialarrays=False)))
if __name__=='__main__':main()
