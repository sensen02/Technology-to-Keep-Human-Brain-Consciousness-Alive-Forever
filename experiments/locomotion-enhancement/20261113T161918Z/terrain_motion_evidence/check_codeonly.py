"""Stdlib derivedCore-source/actualmetadata/completeglobal checks, no actualCore or arrays."""
from pathlib import Path
import ast,json,hashlib,sys,dis,types,builtins
HERE=Path(__file__).resolve().parent;FILES=('contracts.py','render.py','plot_sync.py','check_codeonly.py')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def globals_used(code):
 names=set(x.argval for x in dis.get_instructions(code) if x.opname in ('LOAD_GLOBAL','LOAD_NAME'))
 for c in code.co_consts:
  if isinstance(c,types.CodeType):names.update(globals_used(c))
 return names
def main():
 checks={};texts={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('TerrainevidenceCODEONLY '+k)
 for name in FILES:
  texts[name]=(HERE/name).read_text();tree=ast.parse(texts[name]);compile(tree,str(HERE/name),'exec');check(name+'_syntax',True);check(name+'_noeagerheavyimports',not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(x.name.split('.')[0] in ('numpy','torch','mujoco','physics') for x in n.names) for n in tree.body))
 import contracts,render
 check('exact3actualcases',contracts.CASES==('plane_control','flat_box_control','mild_blocks_seed17_start_repair_v1'));check('actualderived_context_factory_only',all(x in texts['contracts.py'] for x in ('runtime.context(audit=True)','runtime.context(case,audit=True)',"ns['pin']()")) and 'from physics import Core' not in texts['render.py']);check('derivedCorefactory_used',"c=sw.Core(adhesion=False,recorder=None)" in texts['render.py']);check('actualXMLandcompiledchecked','ActualderivedXMLused' in texts['render.py'] and "identity.verify_runtime(case['identity'],c)" in texts['render.py']);check('allmjstep_forbidden_beforeCore','mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden' in texts['render.py']);check('terrain_group5_true_scale','option.geomgroup[5]=1' in texts['render.py'] and 'no height exaggeration' in texts['render.py']);check('same336NNnotnewmotor',contracts.MOTOR=='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd' and contracts.BRAIN.startswith('92dc'))
 check('warmfailure_preservednot_alljobsPASS','oldjobs[4][\'exit_code\']!=0' in texts['contracts.py']);check('realterrainRGBnotzeroed','real_terrain_RGB_used_by_vision' in texts['contracts.py']);check('roughgateFAILhonest','rough_terrain_pass' in texts['render.py'] and "else 'FAIL'" in texts['render.py']);check('noaddedredodorENVmarkers','add_decor' not in texts['render.py'] and 'No addedredtarget/odorENVdecor' in texts['render.py']);check('first20ms_no_negativeRGB','NO RELEASED RGB' in texts['render.py'] and '0<=fid<770' in texts['render.py']);check('oldbodyfieldnames_fromactualheaders',all(x in texts['render.py'] for x in ('pad_ground_height','com_relative_height','registered_contact_count','loads_normal','nonfoot_normal')));check('no_contactcounts_astrials','Events are solvercontacts, not trials' in texts['render.py']);check('no_wholecourse_claim','No wholefield completion claim' in texts['render.py']);check('frame770full_decode','count==770' in texts['render.py'] and "'50/1'" in texts['render.py']);check('fullbodyposechecks',all(x in texts['render.py'] for x in ('COMreconstruction','Yawreconstruction','Actualheadreconstruction','Fullbodyframecontainment')));check('nominal5ms_no_future','physical_ms//5' in texts['render.py'] and "d['time'][qi]<=now+1e-8" in texts['render.py'])
 names=globals_used(render.main.__code__);missing=sorted(x for x in names if x not in render.main.__globals__ and not hasattr(builtins,x));check('allcompiledmainnestedglobals_resolved',not missing)
 for case in contracts.CASES:
  folder=contracts.root(case)/'trials'/case;r=contracts.read(folder/'RESULT.json');a=contracts.read(folder/'VERIFY.json');check(case+'_actualfullfreshNNproof',a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(folder/'RESULT.json') and r['state_total']==336 and r['terrain_case']==case and r['native']['actual_native_steps']==1232000);check(case+'_actualXMLhash',sha(r['actual_xml'])==r['actual_xml_sha256'])
 check('never_loadpolicy_or_forward',all(x not in texts['render.py'] for x in ('load_policies(','.mean_step(','.brain_step(','.load_checkpoint(')));check('package_import_not_neuralexecution_guard',"'torch' not in sys.modules" not in texts['render.py'])
 check('actualheaderpayloadnotread',json.loads((HERE/'ACTUAL_ARCHIVE_HEADERS.json').read_text())['header_only_no_payload']);check('noarraysTorchCoreimport',not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','student_wave','flygym')));check('noactualrenderoutputs',not any((HERE/n).exists() for n in ('SYNCHRONIZED_EVIDENCE',)+tuple('VIDEO_'+x for x in contracts.CASES)))
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(HERE/n):sha(HERE/n) for n in FILES+('PREREG.json','ACTUAL_ARCHIVE_HEADERS.json')},complete_renderer_globalnames=sorted(names),unresolved_globals=missing,array_reads=0,models_loaded=0,Core_constructions=0,render_calls=0,NN_forward_calls=0,physics_steps=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),globals=len(names),unresolved=missing,arrays=0,models=0,Core=0,render=0)))
if __name__=='__main__':main()
