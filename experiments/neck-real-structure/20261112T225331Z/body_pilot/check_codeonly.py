"""Stdlib-only source contracts before actual policy/physical runtime; not numerical proof."""
from pathlib import Path
import ast,json,hashlib,sys,importlib.util
ROOT=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 checks={}
 def check(name,v):checks[name]=bool(v)
 names=('runtime.py','runner.py','audit.py','prepare.py','preflight.py','summary.py','orchestrate.py','check_codeonly.py')
 trees={n:ast.parse((ROOT/n).read_text()) for n in names};check('all8_scripts_parse',len(trees)==8)
 spec=importlib.util.spec_from_file_location('_codeonly_bodypilot_runtime',ROOT/'runtime.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 check('default_import_no_heavy',not any(k in sys.modules for k in ('torch','numpy','mujoco','physics','student_wave','model')))
 check('six_unique_cases',len(m.IDS)==len(set(m.IDS))==6);check('actual_fixed_checkpoint200',m.BRIDGE_SHA=='eed1ad3cc61a1b59086ff2324175f6bf3b5da5aa197165e6c4b009c17027c95e' and m.BRIDGE.name=='checkpoint_000200.pt');check('actualneckdigest',m.BINDING_SHA=='9e88a04dd91aa6f9c1c1c68e0562325205e0f36f5cdd9ae41d1dd46c8171dc23')
 check('localbodycuezero_matchesempty',all(m.cue('P_empty',i)==m.cue('P_left_cuezero',i)==0 for i in (0,5800,6400,7400,9400,15400)))
 check('cueleft_cutvariants_identical',all(m.cue('P_left_pulse',i)==m.cue('P_left_DN_cut',i)==m.cue('P_left_AN_cut',i) for i in (5800,6400,7400,9399,9400)))
 check('cueplusminusonlywindow',m.cue('P_left_pulse',6400)==1 and m.cue('P_right_pulse',6400)==-1 and m.cue('P_left_pulse',6399)==0 and m.cue('P_left_pulse',9400)==0)
 for n in names[1:-1]:check('publicmain_'+n,any(isinstance(node,ast.FunctionDef) and node.name=='main' for node in trees[n].body))
 src=(ROOT/'runtime.py').read_text();runner=(ROOT/'runner.py').read_text();audit=(ROOT/'audit.py').read_text()
 check('brainonly_receives_AN_and_cue',"self.model.brain_step(used_an,torch.tensor([[instruction]]" in src)
 check('decoderonly_uses_delivered_DN',"decoded=self.model.decoder(used_dn)" in src)
 check('warmbrainholdzero_DN0',"if ms<5800:dn=torch.zeros((1,2)" in src)
 check('cut_heldport_silence_not_turngate',"self.held[d]=torch.zeros_like(self.held[d])" in src and "turn=decoded[:,0] if ms>=5800" in src)
 check('queue_only1ms_advance_inside_tick',"tr.tick(c,action,source);ms+=1;loop.advance()" in runner)
 check('complete3080neural15400body',"range(3080)" in runner and "for _ in range(5)" in runner and "fullbody" not in runner)
 check('state272_not_old288',"state_total=272" in runner and "state_total=288" not in runner and "(3080,272)" in audit)
 check('handover_1160_warmup_realAN',"d['AN_capture_step'][1160]==5795" in audit and "d['AN_release_step'][1160]==5797" in audit)
 check('teacheronly960_beforeANN',"if now==1000:teacher=" in runner and "elif warm:action,_=teacher.action" in runner)
 check('source_queue_and_captureconsumption_saved',"save(out/'SOURCE_QUEUE.json',queue)" in runner and 'all_source_queue_and_consumption_exact' in audit)
 check('no_camera_or_worldtargetconstruction',all(x not in runner for x in ('Renderer','mjv_initGeom','render_eye_pair','CameraStream','target_centers_xyz')))
 check('no_body_statewrites',all(x not in runner for x in ('c.d.qpos[:]=','c.d.qvel[:]=','mj.mj_forward','mj.mj_step')))
 check('actualoffline_integrity_not_fakeallPASS',"offline_all_functional_PASS_claim=False" in (ROOT/'prepare.py').read_text())
 check('all6functionalnot_required',"all6functionalPASS_claim=False" in (ROOT/'summary.py').read_text())
 prereg=json.loads((ROOT/'PREREG.json').read_text());check('registered6completebudget',prereg['budget']['physical_trials']==6 and prereg['records']['neural5ms_including_settle_AN']==3080)
 if not all(checks.values()):raise ValueError('CODE_ONLY '+str({k:v for k,v in checks.items() if not v}))
 out=ROOT/'CODEONLY_CHECK.json'
 with out.open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(ROOT/n):sha(ROOT/n) for n in (*names,'PREREG.json')},heavy_imports=False,arrays_loaded=0,model_forward_calls=0,Core_constructions=0,physics_steps=0,training_updates=0,interpretation='Staticsourcecontracts only;parentactual0Corepreflight then6fullphysicaltrials remainunexecuted'),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False)))
if __name__=='__main__':main()
