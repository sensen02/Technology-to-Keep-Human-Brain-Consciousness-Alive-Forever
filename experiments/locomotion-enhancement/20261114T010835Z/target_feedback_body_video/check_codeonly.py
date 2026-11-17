"""ONLYCODE: source AST, old ZIP NPY headers, JSON mocks and globals recursion.
Never call qualify(), build_model(), graph(), render_case() or any backend.
"""
from pathlib import Path
import ast,builtins,copy,dis,inspect,json,math,struct,sys,tempfile,types
import contract,render,mp4_timing,deliver,freeze
ROOT=Path(__file__).resolve().parent
OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_body_component')
CODE=('contract.py','render.py','mp4_timing.py','deliver.py','check_codeonly.py','freeze.py')

def globals_check(module):
 checked=[]
 def code_check(code,env,where):
  for instruction in dis.get_instructions(code):
   if instruction.opname=='LOAD_GLOBAL':
    name=instruction.argval
    if name not in env and not hasattr(builtins,name):raise AssertionError('Missing actual global '+where+':'+name)
  for value in code.co_consts:
   if isinstance(value,types.CodeType):code_check(value,env,where+'/'+value.co_name)
 for name,value in vars(module).items():
  if isinstance(value,types.FunctionType) and value.__module__==module.__name__:
   if value.__globals__ is not vars(module):raise AssertionError('Foreign function globals '+name)
   code_check(value.__code__,value.__globals__,module.__name__+'.'+name);checked.append(name)
 return checked

def main():
 checks={};asts={name:ast.parse((ROOT/name).read_text()) for name in CODE};checks['all_python_AST']=True
 for name,tree in asts.items():
  if name=='check_codeonly.py':continue
  for node in tree.body:
   if isinstance(node,(ast.Import,ast.ImportFrom)):
    names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module]
    if any(x.split('.')[0] in ('numpy','torch','mujoco','PIL','physics','flygym','model') for x in names):raise AssertionError('Heavy top-level import '+name)
 checks['all_heavy_imports_deferred']=True
 require=contract.require;digest='a'*64;negative=0
 required={k:True for k in ('NN352_shapes','NN352_state_chain','phase_source_labels','native_counts_bound','all71opaque','identity_canonical_digest','finite_PD_actual_torque')}
 def mock(condition,n):
  return dict(kind=contract.FORMAT,condition=condition,command=contract.CASES[condition],checkpoint_sha256=digest,final_step=600,motor_format='INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1',state_dims=[128,64,64,96],state_total=352,actual_ms=n,actual_native_steps=n*80,native={'actual_native_steps':n*80},neural_records=math.ceil(n/5),full_duration=n==15400,early_stop=None if n==15400 else {'time_s':n/1000,'reason':'MOCK_METADATA_ONLY'},teacher_calls_after_handover=0,physical_component_only=True,brain_or_sensory_models_loaded=False,neck_ports_present=False,training_updates=0,all71opaque=True,candidate_metrics={'task_pass_candidate':False},original_five={'all_gates':False})
 for condition in contract.CASES:
  for n in (15400,6371,917):
   r=mock(condition,n);a=dict(checkpoint_sha256=digest,integrity_all_pass=True,checks=required.copy(),state_total=352)
   timing=contract.metadata_contract(r,a,condition,digest);require(timing['frames']==math.ceil(n/20) and render.indices(n)[-1]==n-1,'Exact short/full endpoint')
   require(abs(sum([.02]*(timing['frames']-1))+timing['last_frame_ms']/1000-n/1000)<1e-9,'Exact planned movie duration')
   for key,value in [('checkpoint_sha256','b'*64),('integrity_all_pass',False),('checks',{}),('checks',{'NN352_shapes':False})]:
    bad=copy.deepcopy(a);bad[key]=value
    try:contract.metadata_contract(r,bad,condition,digest)
    except ValueError:negative+=1
    else:raise AssertionError('Bad audit admitted')
 checks['four_cases_full_and_earlywarm_earlyANN_mocks']=True;checks['behavior_FAIL_not_entry_gate']=True;checks['last_actual_qpos_included']=True;checks['shortwindow_duration_no_padding']=True
 try:contract.metadata_contract(mock('walk',15400),dict(checkpoint_sha256=digest,integrity_all_pass=True,checks=required,state_total=352),'walk','UNKNOWN')
 except ValueError:negative+=1
 else:raise AssertionError('UnknownSHA admitted')
 checks['unknownSHA_rejected']=True
 # Only synthetic MP4 timing boxes, no encoded media/image/preview; fixture auto-deleted.
 box=mp4_timing.box
 stts=b'\0'*4+struct.pack('>I',1)+struct.pack('>II',3,20)
 mdhd=b'\0'*12+struct.pack('>II',1000,60)+b'\0'*4
 mvhd=mdhd+b'\0'*60
 tkhd=b'\0'*20+struct.pack('>I',60)+b'\0'*40
 elst=b'\0'*4+struct.pack('>I',1)+struct.pack('>IiHH',60,0,1,0)
 synthetic=box(b'ftyp',b'isom\0\0\0\0')+box(b'mdat',b'TIMING_ONLY_NOT_VIDEO')+box(b'moov',box(b'mvhd',mvhd)+box(b'trak',box(b'tkhd',tkhd)+box(b'edts',box(b'elst',elst))+box(b'mdia',box(b'mdhd',mdhd)+box(b'minf',box(b'stbl',box(b'stts',stts))))))
 with tempfile.TemporaryDirectory(prefix='bodyvideo-metadata-only-') as tmp:
  fixture=Path(tmp)/'timing-boxes.bin';fixture.write_bytes(synthetic);info=mp4_timing.shorten(fixture,51,11);require(info['last_sample_ms']==11 and info['media_bytes_unchanged'],'Timing tail')
  updated=fixture.read_bytes();require(b'TIMING_ONLY_NOT_VIDEO' in updated,'Media unchanged')
  def collect(data):
   found={}
   for tag,payload in mp4_timing.boxes(data):
    if tag in (b'moov',b'trak',b'mdia',b'minf',b'stbl',b'edts'):found.update(collect(payload))
    else:found[tag]=payload
   return found
  found=collect(updated);require(struct.unpack('>I',found[b'mdhd'][16:20])[0]==51,'Exact duration metadata');require(struct.unpack('>IIII',found[b'stts'][8:24])==(2,20,1,11),'Tail sample metadata')
 checks['pure_timing_box_mock_no_video_and_fixture_cleaned']=True
 actual_headers={}
 for name in ('trace_1ms.npz','geometry_1ms.npz','neural_decisions.npz','runtime_geometry.npz'):
  actual_headers[name]=contract.headers(OLD/'trials/walk'/name)
 require(actual_headers['trace_1ms.npz']['qpos']['shape']==[15400,49] and actual_headers['trace_1ms.npz']['qvel']['shape']==[15400,48] and actual_headers['trace_1ms.npz']['actualctrl']['shape']==[15400,42],'Observed old actual trace schema')
 require(actual_headers['neural_decisions.npz']['hidden_before']['shape']==[3080,352],'Observed old352 schema');checks['old_actual_ZIP_headers_only_schema']=True
 source=(ROOT/'render.py').read_text();require('mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden' in source and 'mj.mj_forward(model,data)' in source,'Only forward entrypoint')
 require('mj_contactForce' not in source and '.contact' not in source,'No reconstructed contact measurement');checks['zero_integration_and_no_contactmetrics_source_guard']=True
 require('spec.compile()' in source and source.count('spec.compile()')==1 and 'Core(' not in source and '.step_details(' not in source,'One direct model compile/zeroCoreNN');checks['one_direct_MjSpec_zeroCoreNN_source_guard']=True
 require('FRAME_MAPPING.json' in source and 'pts[-1]+durations[-1]-n/1000' in source and 'decode_rc==0 and count==len(sample)' in source,'Runtime decode/time evidence guards');checks['runtime_moviePTS_physicalend_full_decode_guards']=True
 function_globals={m.__name__:globals_check(m) for m in (contract,render,mp4_timing,deliver,freeze)};checks['recursive_ownfunction_nestedcode_globals_resolve']=True
 require(not any(k in sys.modules for k in ('torch','numpy','mujoco','physics','flygym','model','PIL')),'No heavy imports during ONLYCODE');checks['actual_zero_heavy_modules_imported']=True
 result=dict(status='ONLYCODE_PASS_CONDITIONAL_ACTUAL_UNKNOWN',checks=checks,negative_mock_cases=negative,function_globals_checked=function_globals,old_actual_headers=actual_headers,NPZ_payloads_read=0,PT_payloads_read=0,new_actual_inputs_read=0,model_loads=0,NN_forwards=0,Core_constructions=0,model_compiles=0,physics_steps=0,render_frames=0,previews_created=0,claim='Code-only readiness; no actual final600/body/video qualification or acceptance claimed.')
 contract.save(ROOT/'CODECHECK.json',result);print(json.dumps(dict(status=result['status'],checks=len(checks),negative_mock_cases=negative,globals_functions=sum(map(len,function_globals.values())),heavy_imports=0)))
if __name__=='__main__':main()
