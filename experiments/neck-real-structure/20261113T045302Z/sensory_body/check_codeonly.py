"""Standard-library source/ABI checks only, no numerical module or payload execution."""
from pathlib import Path
import ast,json,sys,hashlib,subprocess
ROOT=Path(__file__).resolve().parent

def main():
 checks={};hashes={};trees={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('SensoryCODEONLY '+k)
 def sha(p):
  with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
 names=('runtime.py','runner.py','audit.py','prepare.py','preflight.py','summary.py','review.py','orchestrate.py','check_codeonly.py')
 for name in names:
  p=ROOT/name;trees[name]=ast.parse(p.read_text());compile(trees[name],str(p),'exec');hashes[str(p)]=sha(p);check(name+'_ASTcompiles',True)
 for name in names[1:]:check(name+'_publicmain',any(isinstance(n,ast.FunctionDef) and n.name=='main' for n in trees[name].body))
 p=ROOT/'PREREG.json';pr=json.loads(p.read_text());hashes[str(p)]=sha(p);check('14conditions',len(pr['conditions'])==14 and len(set(pr['conditions']))==14);check('336layout',pr['state_layout']==[128,64,64,32,32,16] and pr['state_total']==336);check('newkindknown',pr['summary_kind']=='SOURCE_NECK_MULTISENSORY_BODY_SUMMARY_V1');check('calrecoveryreasonprebind',pr['calibration_root'].endswith('/sensory_fusion_recovery_v1') and 'zerooptimizer/physics' in pr['before_any_body_binding_recovery_reason']);check('Onewengineeringgatesdeclared','025' in json.dumps(pr['gates']).replace('.25','025') or '.25' in json.dumps(pr['gates']))
 r=(ROOT/'runtime.py').read_text();check('exactHeadsourceoffsets_loaded',"read(HEAD_AUDIT)['sensor_offsets']['local_xyz']" in r);check('exactoriginalfilterclass_reused',"module('_fusion_exact_original_clampfilter',PEAK_SOURCE).ClampFilter()" in r and "self.filter.advance(C)" in r);check('previouspending_notcurrent',"used=sensory['receptors'].copy()" in r);check('braindirect_onlyAN_andmodalities2',"self.bridge.brain_step(anheld,torch.cat((visual_raw.reshape(1,1),odor_raw.reshape(1,1)),-1),self.brain)" in r);check('nobodytargetbearing_actorinput','source_xy' not in r[r.index('class BrainNeckLoop:'):]);check('olfnaturalbias_nogating',"odor.mean_step" in (ROOT/'runner.py').read_text() and "if warm:" in (ROOT/'runner.py').read_text());check('sourcequeue14full',"proof['actual_steps']!=15400" in r);check('partialfailures_preserved',"FAILURE.json" in (ROOT/'runner.py').read_text());check('savedactual336replay',"actualRGB_field_receptor_queue_4NN_replay:" in (ROOT/'audit.py').read_text());check('summaryvisuallegacyunchanged',"visual_body_v1/summary.py" in (ROOT/'summary.py').read_text());check('summaryexec_nonmain_namespace',"__name__='_fusion_legacy_visual_thresholds'" in (ROOT/'summary.py').read_text());check('no_unknownSHA_source_guess',"digest=a.brain_sha256" in (ROOT/'prepare.py').read_text());check('parent_finalrecoveryFORMAT',pr['calibration_format']=='SOURCE_NECK_VISUAL_OLFACTORY_FUSION_RECOVERY_V1')
 sys.path.insert(0,str(ROOT));import runtime
 check('runtime_standardlibraryonly',not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','student_wave','flygym')));check('actualCALpath_recovery',runtime.CAL.name=='sensory_fusion_recovery_v1' and runtime.FORMAT==pr['calibration_format']);check('actualIDS_matchPREREG',list(runtime.IDS)==pr['conditions']);check('camera_mapsENVonly',runtime.camera_condition('F_left_DN_cut')=='V_left_pulse' and runtime.camera_condition('O_left_swap')=='V_empty');check('readonlyoldcoreexists',all(p.is_file() for p in (runtime.PREV/'runtime.py',runtime.PREV/'runner.py',runtime.PEAK_SOURCE,runtime.HEAD_AUDIT)))
 recover=runtime.CAL/'common.py';tree=ast.parse(recover.read_text());texts=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)];check('actualrecoveryFORMATread',runtime.FORMAT in texts);hashes[str(recover)]=sha(recover)
 for script in ('runner.py','preflight.py','orchestrate.py'):
  cp=subprocess.run([sys.executable,'-B',str(ROOT/script)],capture_output=True,text=True,check=False);check(script+'_defaultnoheavyflag',cp.returncode==0 if script!='runner.py' else cp.returncode==2)
 # Runner argparse requires condition; parse failure is before any numericalcalls.
 out=ROOT/'CODEONLY_CHECK_FINAL.json'
 with out.open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes=hashes,stdlib_only=True,model_loads=0,array_reads=0,Core_constructions=0,training_updates=0,new_physics_steps=0,heavy_imports=False),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False)))
if __name__=='__main__':main()
