"""Stdlibvisualsource/explicitroot/closedcoveradapter checks,no actualresults or heavyimports."""
from pathlib import Path
import ast,json,sys,hashlib
HERE=Path(__file__).resolve().parent

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 checks={};files=[HERE/n for n in ('PREREG.json','contracts.py','adapter.py','render.py','plot_sync.py','check_codeonly.py')]
 for p in files:
  if p.suffix=='.py':ast.parse(p.read_text());checks[p.name+'_syntax']=True
 import contracts,adapter
 checks['newROOT']=contracts.ROOT==HERE.parent/'body';checks['newVIS']=contracts.VIS==HERE.parents[1];checks['newvisualSHA']=contracts.VISUAL_SHA=='996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87';checks['oldsummary_ast_rebound']='VISUAL_OPPONENT_FAR_BODY_V7' in ast.unparse(contracts.NewSummary().visit(ast.parse((contracts.ORIGINAL/'contracts.py').read_text())))
 for name in ('render.py','plot_sync.py'):compile(adapter.source(name),name+'::codeonly','exec');checks[name+'_exactadaptercompile']=True
 render=ast.unparse(adapter.source('render.py'));checks['closedcovers_halfsize_fixed']='[0.18, 0.18, 1.4]' in render;checks['savedworldcovers_notlaterheadrecomputed']='recorded_covers(g, cap)' in render and 'geometry_path =' not in render;checks['actualcapcoverrotation']='capture[\'cover_rotation\']' in render;checks['engineeringopponentlabel']='engineering red-opponent' in render
 old=(contracts.ORIGINAL/'render.py').read_text();checks['nointegration770fullDecode']='mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden' in old and "count==770" in old and 'range(0,15400,20)' in old;checks['bodysegmentation_beforedecor']=old.index('renderer.enable_segmentation_rendering()')<old.index('camera.add_decor(renderer.scene')
 checks['RGBeyeinsets_savedonly']='Image.fromarray(rgb[fid,eye])' in old;checks['sourcecapture_release_notfuture']="cap[fid]['release_time_s']<=now+1e-10" in old;checks['normalcontrolseparate']='case[\'iscontrol\']' in old;checks['all72offline_required_contract']='len(required)==72' in (HERE/'contracts.py').read_text();checks['newfixedidentitycontract']='VISION_FIXED_NEURAL_OPPONENT_ARCH_V7' in (HERE/'contracts.py').read_text()
 heavy={k:k in sys.modules for k in ('numpy','torch','mujoco','physics','runtime','model','matplotlib')};checks['noheavyimports']=not any(heavy.values())
 if not all(checks.values()):raise ValueError(checks)
 files += [contracts.ORIGINAL/n for n in ('contracts.py','render.py','plot_sync.py')]
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(p):sha(p) for p in files},heavy_imports=heavy,arrays_read=0,actual_trial_reads=0,Core_constructions=0,model_forward_calls=0,render_calls=0,plot_calls=0,physics_steps=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=heavy)))
if __name__=='__main__':main()
