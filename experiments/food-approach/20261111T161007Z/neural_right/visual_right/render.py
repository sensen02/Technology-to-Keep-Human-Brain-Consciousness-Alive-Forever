"""Conditional render-only preparation. Heavy imports occur only in main()."""
from pathlib import Path
import hashlib,json,math
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REFERENCE=ROOT.parent.parent/'20261111T072012Z/render.py'

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def require_contract(result,audit,run,checkpoint,digest):
 if result['condition']!='right' or audit['condition']!='right':raise ValueError('RIGHT only')
 if Path(audit['run']).resolve()!=run.resolve():raise ValueError('audit run mismatch')
 if not audit['integrity_all_pass'] or not audit['checks'] or not all(v is True for v in audit['checks'].values()):raise ValueError('incomplete integrity audit')
 if Path(result['checkpoint']).resolve()!=checkpoint.resolve():raise ValueError('only final2000')
 if result['checkpoint_sha256']!=digest or audit['checkpoint_sha256']!=digest:raise ValueError('checkpoint identity mismatch')
 if audit['engineering_pass'] is not result['engineering_pass']:raise ValueError('audit status mismatch')
 native=result['native']
 if native['actual_native_steps']!=1232000 or native['post_native_steps']!=768000:raise ValueError('incomplete native recorder')
 if not result['transition']['all_exact'] or not result['transition']['no_reset']:raise ValueError('handover mismatch')
 # Failed measured outcomes remain eligible for honest video review.
 return dict(target_right_degrees=30.,actual_right_degrees=-math.degrees(result['yaw_change_rad']),engineering_pass=result['engineering_pass'])

def build_source():
 prereg=json.loads((HERE/'PREREG.json').read_text())
 if sha(REFERENCE)!=prereg['reference_sha256']:raise ValueError('frozen renderer drift')
 source=REFERENCE.read_text()
 def replace(old,new):
  nonlocal source
  if source.count(old)!=1:raise ValueError('renderer adaptation mismatch: '+old)
  source=source.replace(old,new)
 replace("BASE=HERE.parent.parent/'banc-walking-recovery/20261110T152714Z'", "BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')")
 replace("run=Path(a.run);out=HERE/('VIDEO_'+a.condition);assert not out.exists()", "run=Path(a.run).resolve();out=HERE/('VIDEO_'+a.condition);assert not out.exists()\n assert run==ROOT/'EVAL_right' and Path(a.audit).resolve()==ROOT/'VERIFY_right.json' and a.condition=='right'")
 replace("assert r['checkpoint_sha256']=='6a6c23feb9a4b36fcf761f62f9d63309b5c81763f950afe520f42c09a5855e2f'", """checkpoint=ROOT/'TRAIN/checkpoint_002000.pt'
 require_contract(r,audit,run,checkpoint,sha(checkpoint))
 import importlib.util,torch
 torch.set_num_threads(1)
 spec=importlib.util.spec_from_file_location('_bigturn_right_checked_model',ROOT/'model.py');io=importlib.util.module_from_spec(spec);sys.modules[spec.name]=io;spec.loader.exec_module(io)
 checked_model,payload=io.load_checked(checkpoint,r['checkpoint_sha256'],require_final=True)
 assert payload['format_version']=='RIGHT_ONLY_V1' and payload['curve'][-1]['step']==2000
 assert 'yaw/.06' in payload['identity']['command_domain'] and payload['identity']['command_domain'].startswith('forward0; turn[-1,0]')
 del checked_model,payload""")
 replace("[Path(a.audit),Path(__file__).resolve(),HERE/'PREREG.json']", "[Path(a.audit),Path(__file__).resolve(),HERE/'PREREG.json',REFERENCE,ROOT/'model.py',ROOT/'EVAL_PREREG.json']")
 replace("assert len(t['time'])==15400", "assert r['identity_sha256']==ident['identity_sha256'] and audit['checkpoint_sha256']==r['checkpoint_sha256']\n assert len(t['time'])==15400")
 replace("cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.distance=11.;cam.azimuth=140.;cam.elevation=-25.;cam.lookat[:]=(t['com'].min(axis=0)+t['com'].max(axis=0))/2", """# Fixed camera fitted to all saved COM and actual mesh bounding spheres, no motion edits.
 center=(t['com'].min(axis=0)+t['com'].max(axis=0))/2
 radius=float(np.linalg.norm(t['com'].max(axis=0)-t['com'].min(axis=0))/2)
 for i in range(0,15400,20):
  c.d.qpos[:]=t['qpos'][i];c.d.qvel[:]=t['qvel'][i];c.d.time=float(t['time'][i]);mj.mj_forward(m,c.d)
  radius=max(radius,float(np.max(np.linalg.norm(c.d.geom_xpos[mesh]-center,axis=1)+m.geom_rbound[mesh])))
 cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.distance=max(11.,radius/np.sin(np.deg2rad(float(m.vis.global_.fovy)/2))*1.6);cam.azimuth=140.;cam.elevation=-25.;cam.lookat[:]=center""")
 replace("f'{a.condition.upper()} TURN | VISUAL REVIEW | {phase}'", "f'BIG RIGHT TURN | VISUAL REVIEW | {phase}'")
 replace("f'ANN window: slip {r[\"autonomous\"][\"slip_ratio_max\"]:.6f} (original limit 0.5) | yaw {r[\"yaw_change_rad\"]:+.6f} rad'", "f'ANN slip {r[\"autonomous\"][\"slip_ratio_max\"]:.6f} (limit .5) | RIGHT {-np.degrees(r[\"yaw_change_rad\"]):.3f} deg (target 30)' ")
 replace("'1.0 s settle + 4.8 s teacher + 9.6 s ANN | original gate '", "'1s settle + 4.8s teacher + 9.6s ANN | measured acceptance '")
 replace("verify_runtime(ident,c);assert all(sha(file)==h for file,h in hashes.items())", """verify_runtime(ident,c)
 m.vis.global_.offwidth,m.vis.global_.offheight=original_framebuffer
 assert [int(m.vis.global_.offwidth),int(m.vis.global_.offheight)]==original_framebuffer
 verify_runtime(ident,c);assert all(sha(file)==h for file,h in hashes.items())""")
 replace("framebuffer_visual_only={'before':original_framebuffer,'after':[width,height]}", "framebuffer_visual_only={'before':original_framebuffer,'during':[width,height],'after':original_framebuffer},fixed_camera={'center':center.tolist(),'distance':cam.distance,'all_saved_mesh_bound_radius':radius},target_right_degrees=30.,actual_right_degrees=-float(np.degrees(r['yaw_change_rad'])),checkpoint_sha256=r['checkpoint_sha256'],model_format='RIGHT_ONLY_V1',model_domain='yaw/.06'")
 return source

def main():
 # Called only after the parent's sole heavy worker has finished and audit exists.
 source=build_source()
 namespace=dict(__name__='_conditional_big_right_render',__file__=str(Path(__file__).resolve()),ROOT=ROOT,REFERENCE=REFERENCE,require_contract=require_contract)
 exec(compile(source,str(Path(__file__).resolve())+'::frozen_renderer_adapted','exec'),namespace)
 namespace['main']()
if __name__=='__main__':main()
