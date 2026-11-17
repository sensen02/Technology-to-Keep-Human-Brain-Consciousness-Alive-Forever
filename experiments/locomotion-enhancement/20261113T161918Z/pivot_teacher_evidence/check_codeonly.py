"""Stdlib source/schema/completeglobal resolution checks, no payload/model/Core/render."""
from pathlib import Path
import ast,json,hashlib,sys,dis,types,builtins,importlib.util
HERE=Path(__file__).resolve().parent
FILES=('contracts.py','render.py','plot_sync.py','check_codeonly.py')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def globals_used(code):
 out=set(x.argval for x in dis.get_instructions(code) if x.opname in ('LOAD_GLOBAL','LOAD_NAME'))
 for x in code.co_consts:
  if isinstance(x,types.CodeType):out.update(globals_used(x))
 return out

def main():
 checks={};texts={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('TeacherEvidenceCODEONLY '+k)
 for name in FILES:
  text=(HERE/name).read_text();t=ast.parse(text);compile(t,str(HERE/name),'exec');texts[name]=text;check(name+'_ASTcompile',True);check(name+'_noeagerheavyimports',not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(a.name.split('.')[0] in ('numpy','torch','mujoco','physics') for a in n.names) for n in t.body))
 import contracts,render
 check('actualstop_left_right_sources',contracts.CASES==('stop','left_pivot','right_pivot') and contracts.STOP.name=='stop_shadow_control' and contracts.PIVOT.name=='pivot_fixed_shadow_v2');check('sourceactualarchiveheaders_only',json.loads((HERE/'ACTUAL_ARCHIVE_HEADERS.json').read_text())['header_only_no_payload']);check('actualteacheronly_noANNgates','ALGORITHM_TEACHER_ONLY_NOT_ANN' in texts['contracts.py'] and "r['no_neural_models']" in texts['contracts.py']);check('actualStopNoSpeedGate',"['stop_speed_PASS'] is None" in texts['contracts.py']);check('actualgeometry14400_aftersettle',"g['thorax_yaw'].shape==(14400,)" in texts['contracts.py'] and 'i-1000' in texts['render.py']);check('actual2880teacher_samples','(2880,42)' in texts['contracts.py']);check('actualsource6candidate',"np.full(9600,6)" in texts['contracts.py']);check('allmatchedsourcequalification','MATCHED_COMPARISON.json' in texts['contracts.py'] and "'MATCHED_'+case+'.json'" in texts['contracts.py']);check('header_teacher_notANN_neck','ALGORITHM TEACHER — NOT ANN/NECK PIVOT' in texts['render.py']);check('stop_not_learnedPASS','LEARNED STOP NOT VALIDATED' in texts['render.py']);check('cyclecounts_actualeventtimes','e[\'time\']<=tt+1e-10' in texts['contracts.py'] and 'FINAL cyclecounts' in texts['render.py']);check('targetrate_requested_notachieved','Request yawrate (rad/s)' in texts['render.py']);check('legacyfive_separate','not pivot success criterion' in texts['render.py']);check('zero_integration_renderer','mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden' in texts['render.py']);check('complete770decode77050fps','count==770' in texts['render.py'] and "'50/1'" in texts['render.py']);check('allpose_COM_head_yawchecks',all(k in texts['render.py'] for k in ('ActualCOMreconstruction','Actualthoraxyawrotation','Actualarchivedheadreconstruction','Wholebodyframecontainment')));check('no_fake_eyes_or_neuraldata','no eyeRGB or neuralstate exists' in texts['render.py']);check('nonfutureteacherlookup',"d['time'][qi]<=now+1e-8" in texts['render.py']);check('nativegeometry_scale_notSI','absolute SI NOT_ESTABLISHED' in texts['render.py']);check('allsourcebeforeafterledger','unchanged(hashes)' in texts['render.py'])
 names=globals_used(render.main.__code__);missing=sorted(n for n in names if n not in render.main.__globals__ and not hasattr(builtins,n));check('complete_render_main_nestedglobals_resolved',not missing);check('no_numericalimports',not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','student_wave','flygym')));check('no_actualoutputs',not any((HERE/n).exists() for n in ('SYNCHRONIZED_EVIDENCE','VIDEO_stop','VIDEO_left_pivot','VIDEO_right_pivot')))
 # JSON-only actual scientific schema and sourceproof presence, no archives payloadopened.
 for root,count in ((contracts.STOP,2),(contracts.PIVOT,4)):
  j=contracts.read(root/'JOBS.json');check(root.name+'_actual_stage_exits',len(j)==count and all(x['exit_code']==0 for x in j))
 for case in contracts.CASES:
  root=contracts.STOP if case=='stop' else contracts.PIVOT;f=root/'trials'/case;r=contracts.read(f/'RESULT.json');a=contracts.read(f/'VERIFY.json');check(case+'_actualfullteacherproof',r['full_duration'] and r['actual_ms']==15400 and r['control']=='ALGORITHM_TEACHER_ONLY_NOT_ANN' and a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(f/'RESULT.json'))
 report=dict(all_passed=True,checks=checks,source_hashes={str(HERE/n):sha(HERE/n) for n in FILES+('PREREG.json','ACTUAL_ARCHIVE_HEADERS.json')},all_renderer_global_names=sorted(names),unresolved_globals=missing,arrays_loaded=0,models_loaded=0,Core_constructions=0,render_calls=0,policy_forwards=0,training_updates=0,physics_steps=0)
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),globals=len(names),unresolved=missing,arrays=0,models=0,Core=0,render=0)))
if __name__=='__main__':main()
