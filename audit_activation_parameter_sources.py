"""S4/P3 XML/source-only tau provenance. No model compile, forward, step, neural."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET,csv,sys
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s4-activation-unit-provenance/20261027T211000Z')
XML=P/'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
UP=P/'venv_body/lib/python3.12/site-packages/flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml'
MID=P/'outputs/muscles_six_legs/fruitfly_six_leg_muscles.xml'
META=P/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/native_floor/metadata.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def hits(p,needle):return [dict(path=str(p),line=i,text=s.strip(),sha256=sha(p)) for i,s in enumerate(p.read_text().splitlines(),1) if needle in s]
def main():
 O.mkdir(parents=True,exist_ok=False);scripts=[P/s for s in ['tools_mirror_muscles.py','tools_stand_and_ground.py','tools_repair_moment_arms.py','tools_repair_muscle_lengthrange.py','engine/canonical_physics.py','run_neural_muscle_closure.py','run_body_units_audit.py']];files=[Path(__file__),XML,UP,MID,META]+scripts;hashes={str(p):sha(p) for p in files};assert sha(XML)=='65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6'
 dump(O/'INPUT_HASHES.json',hashes);root=ET.parse(XML).getroot();up=ET.parse(UP).getroot();mid=ET.parse(MID).getroot();act=list(root.find('actuator'));uact={a.get('name'):a for a in up.find('actuator')};mact={a.get('name'):a for a in mid.find('actuator')};compiled=json.loads(META.read_text());assert len(act)==90;assert 'actuator_dynprm' in compiled
 records=[]
 for i,a in enumerate(act):
  name=a.get('name');prefix=next(l for l in ['LF','RF','LM','RM','LH','RH'] if name.startswith(l));source='LF'+name[2:];ua=uact[source];dyn=list(map(float,a.get('dynprm').split()));assert dyn==list(map(float,ua.get('dynprm').split()))==list(map(float,mact[name].get('dynprm').split()));assert dyn==compiled['actuator_dynprm'][i]
  records.append(dict(index=i,actuator=name,leg=prefix,canonical=hits(XML,'name="'+name+'"')[0],intermediate=hits(MID,'name="'+name+'"')[0],upstream=hits(UP,'name="'+source+'"')[0],upstream_name=source,declared_class=a.get('class'),effective_dynprm=dyn,activation_tau_model_time=dyn[0],deactivation_tau_model_time=dyn[1],declared_project_seconds=[dyn[0],dyn[1]],source_type='explicit actuator override; not inherited default',default_not_used=[.01,.04],compiled_saved_match=True,local_tau_conversion='none: exact scalar tokens preserved',physiological_calibration='NOT_ESTABLISHED'))
 dump(O/'ACTUATOR_SOURCES_90.json',records)
 with (O/'ACTUATOR_SOURCES_90.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['index','name','upstream_name','activation_tau','deactivation_tau','canonical_line','upstream_line','status'])
  for r in records:w.writerow([r['index'],r['actuator'],r['upstream_name'],r['activation_tau_model_time'],r['deactivation_tau_model_time'],r['canonical']['line'],r['upstream']['line'],r['physiological_calibration']])
 refs=[]
 for p,words in [(XML,['dynprm="0.01','gravity=','STANDING VARIANT']), (UP,['dynprm="0.01','converted from','gravity=']), (scripts[0],['SRC =','dict(src_act.attrib)','ET.parse(SRC)']), (scripts[1],['SRC_XML =','OUT_XML =']), (P/'engine/canonical_physics.py',['MODEL_TIME_UNIT_S','MM_PER_MODEL_LENGTH =','MODEL_MASS_UNIT_STATUS','GRAVITY_MODEL =']), (P/'run_neural_muscle_closure.py',['DT_NEURAL =','DT_COMMAND =','net.t_ms/1000'])]:
  for word in words:refs.extend(hits(p,word))
 dump(O/'SOURCE_REFERENCES.json',refs)
 dump(O/'UNIT_CONTRACT.json',dict(project_time=dict(unit='s',seconds_per_model_time=1,source=hits(P/'engine/canonical_physics.py','MODEL_TIME_UNIT_S'),clock_conversion='neural ms /1000 compared to body time; no tau rescale in this lineage',activation_ms=.1,deactivation_ms=.4),length=dict(project_unit='mm',mm_per_model_length=1,status='project declared and geometry/source supported; not inferred solely from gravity'),mass=dict(status='NOT_ESTABLISHED',note='asset masses have no unit declarations; inferred gram not validated calibration'),gravity=dict(value=[0,0,-9801],under_project_units='mm/s^2,9.801m/s^2',warning='large number alone is not time-unit error'),upstream_tau=dict(explicit=[.0001,.0004],unused_class_default=[.01,.04],ratio_explicit_to_default=[.01,.01],conversion_rationale='NOT_ESTABLISHED; numeric ratio is not evidence of time conversion'),errors=dict(proven_local_tau_conversion_error=False,unresolved='original reason/measurement for upstream explicit overrides and dimensional conversion derivation remain missing'),required_evidence=['pinned upstream conversion source/commit with activation/deactivation assignment rationale','original OpenSim muscle dynamical parameters and verified converter version/config','authoritative coupled length/mass/time and force unit declaration for this musculoskeletal artifact'],limits='source equality and implementation consistency do not imply physiological suitability or cause of sinking'))
 assert all(sha(Path(p))==v for p,v in hashes.items());dump(O/'EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=0,mj_forward=0,mj_step=0,neural_step=0,actuators=90,compiled_parameter_source=str(META),compiled_parameters='reused saved metadata; no new compile',failures=[]));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('90 explicit upstream tau overrides preserved and saved compiled dynprm exact')
if __name__=='__main__':main()
