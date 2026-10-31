"""S151/P1 ONE <=10ms fixed LF channel4 actual-interface contrast to S149.

Inert preparation; unchanged raw neural decode, native act dynamics and original
six gates/height reference. Saturation is not fault proof or unique cause evidence.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, hashlib, json, sys, types
ROOT=Path(__file__).resolve().parent
S149=ROOT/'run_half_gain_physical_probe.py'
S142=ROOT/'run_free_physical_prewarm_probe.py'
S149_REFERENCE=Path('/home/sensen/Desktop/engine修复验收/s149-half-gain-physical/20261024T213108429114Z')
S150_MAPPING=Path('/home/sensen/Desktop/engine修复验收/s150-leg-command-unloading/20261024T213608033385Z/S149_ACTUATOR_MAP.json')
OUTPUT_PARENT=Path('/home/sensen/Desktop/engine修复验收/s151-lf-channel-gain')
S145_SELECTION=Path('/home/sensen/Desktop/engine修复验收/s145-secant-contact-preload/20261024T203438782961Z')
GAINS=[0.5]*90
GAINS[4]=0.25

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inert(path,name,source=None):
    m=types.ModuleType(name); m.__file__=str(path)
    exec(compile(path.read_text() if source is None else source,str(path),'exec'),m.__dict__)
    return m

def prepare(selection):
    s149=inert(S149,'_s151_s149')
    s142,binding,_,_=s149.prepare(selection)
    refs=[S149_REFERENCE/n for n in ('PREREG.json','INPUT_HASHES.json','COMPLETE.json','HALF_GAIN_INDEPENDENT_CHECK.json','POSE_INITIALIZATION.npz','SECANT_SELECTION_BINDING.json','SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py','SOURCE_FREEZE/HASHES.json','SOURCE_FREEZE/S146_run_half_gain_physical_probe.py','init80ms/TRAJECTORY.npz','init80ms/INITIALIZATION.npz','init80ms/NEURAL_COLD.npz','init80ms/NEURAL_COLD_META.json','init80ms/NEURAL_RELEASE_META.json','init80ms/MOMENT_METADATA.json','init80ms/MODEL_PROVENANCE.json')]
    refs += [S150_MAPPING,S150_MAPPING.parent/'INPUT_HASHES.json',S150_MAPPING.parent/'OUTPUT_HASHES.json',S150_MAPPING.parent/'PARENT_LEG_INDEPENDENT_CHECK.json',S149,Path(__file__).resolve()]
    assert all(p.is_file() for p in refs)
    mapping=json.loads(S150_MAPPING.read_text())
    target=mapping[4]
    assert len(mapping)==90 and target==dict(actuator=4,name='LFC_sternal_anterior_rotator',tendon='LFC_sternal_anterior_rotator_tendon',wrap_type='spatial',referenced_crossed_joints=['joint_LFCoxa_pitch','joint_LFCoxa_roll','joint_LFCoxa_yaw'],dofs=[7,8,6],leg='lf',functional_role='UNKNOWN: no name-based flex/ext inference')
    mm=json.loads((S149_REFERENCE/'init80ms/MOMENT_METADATA.json').read_text())
    assert mm['actuator_names'][4]==target['name']
    for joint,dof in zip(target['referenced_crossed_joints'],target['dofs']):
        assert mm['jnt_dofadr'][mm['joint_names'].index(joint)]==dof
    manifest=json.loads((S150_MAPPING.parent/'OUTPUT_HASHES.json').read_text())
    assert {str(Path(k).resolve()):v for k,v in manifest.items()}[str(S150_MAPPING.resolve())]==sha(S150_MAPPING)
    assert sha(S149)==sha(S149_REFERENCE/'SOURCE_FREEZE/S146_run_half_gain_physical_probe.py')
    binding['hashes'].update({str(p):sha(p) for p in refs})
    provenance=dict(reference=str(S149_REFERENCE),mapping_path=str(S150_MAPPING),mapping_sha256=sha(S150_MAPPING),target=target,zero_based_channel=4,interface_gains=GAINS,raw_decoded_preserved=True,intervention='fixed all calls: channel4 gain0.25; other89 gain0.5; no feedback controller, no tau/weights/pose change',interpretation='cause contrast only; raw saturation is not fault proof; functional role UNKNOWN')
    # Enrich the existing pre-model JSON hooks without modifying old disk files.
    module_source=S142.read_text()
    old="        elif path.name == 'PREREG.json':\n            value = dict(value,source_hashes="
    new="        elif path.name == 'PREREG.json':\n            value = dict(value, s151_channel_gain=s151_provenance, interface_gains=s151_gains, gains_artifact=str(out/'INTERFACE_GAINS.npz'), source_initialization=str(out/'SOURCE_INITIALIZATION.json'))\n            value = dict(value,source_hashes="
    module_source,_=s149.replace_exact(module_source,[(old,new),("protocol='S142 ONE bounded free physical diagnostic; no assistance or activation assignment'","protocol='S151 ONE <=10ms LF_CHANNEL_GAIN; no assistance or activation assignment', EXPERIMENT='LF_CHANNEL_GAIN', candidate=False, S119_same_condition_candidate=False, half_interface_gain=True")])
    # Reuse S149's exact S146 transform and original-height stop hook.
    prior=s142.private_source
    reasons=s142.stop_reasons
    s142=inert(S142,'_s151_private_s142',module_source)
    s142.stop_reasons=reasons
    s142.s151_provenance=provenance; s142.s151_gains=GAINS
    def transform(text):
        source,changes=prior(text)
        source,extra=s149.replace_exact(source,[
          ('                d.ctrl[:] = 0.5 * u','                d.ctrl[:] = s151_gains * u'),
          ('                applied=0.5*np.asarray(commands),decoded=np.asarray(commands),current=np.asarray(currents),','                applied=np.asarray(commands)*s151_gains,decoded=np.asarray(commands),current=np.asarray(currents),'),
          ("applied_interface='all90 ctrl=0.5*decoded raw u after decode; no act overwrite; no tau change'","applied_interface='90-vector s151_gains*decoded raw u; channel4=.25 others=.5; no act overwrite; no tau change'"),
          ("str(ROOT/'run_half_gain_physical_probe.py')]:","str(ROOT/'run_half_gain_physical_probe.py'), str(ROOT/'run_lf_channel_gain_probe.py')]:"),
          ("    atomic_npz(out/'DIRECTION.npz', weights=weights, **arms,","    atomic_npz(out/'INTERFACE_GAINS.npz', interface_gains=np.asarray(s151_gains), zero_based_channel=np.asarray(4))\n    atomic_json(out/'SOURCE_INITIALIZATION.json', dict(s151_provenance, gains_artifact=str(out/'INTERFACE_GAINS.npz'), gains_sha256=sha(out/'INTERFACE_GAINS.npz'), private_runner_sha256=sha(out/'SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py'), source_hashes=identities))\n    atomic_npz(out/'DIRECTION.npz', weights=weights, **arms,"),
          ("clocks='continuous neural/FIR .08s offset; source END at 83ms; physical START at 80ms operator split'","clocks='PRE160 genuine neural-only0..80ms; physical0..<=10ms source80..<=90ms; .5ms neural/25us body/5ms live sensory if reached'")])
        source='s151_gains='+repr(GAINS)+'\ns151_provenance='+repr(provenance)+'\n'+source
        ast.parse(source)
        compile(source,str(s142.ORIGINAL),'exec')  # AST alone does not reject duplicate kwargs.
        return source,changes+extra
    s142.private_source=transform
    source,changes=transform(s142.ORIGINAL.read_text())
    assert 'd.ctrl[:] = s151_gains * u' in source and 'd.ctrl[:] = 0.5 * u' not in source
    assert 'applied=np.asarray(commands)*s151_gains,decoded=np.asarray(commands)' in source
    return s142,binding,source,changes

def main():
    p=argparse.ArgumentParser(); p.add_argument('--selection',type=Path,required=True); p.add_argument('--output',default='auto'); a=p.parse_args()
    out=OUTPUT_PARENT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') if a.output=='auto' else Path(a.output)
    if not out.is_absolute() or out.exists(): raise ValueError('fresh absolute output only; no resume')
    s142,binding,source,changes=prepare(a.selection)
    assert binding['hashes']=={n:sha(n) for n in binding['hashes']}
    old=sys.argv
    try: sys.argv=[str(Path(__file__).resolve()),'--output',str(out)]; s142.main()
    finally: sys.argv=old
if __name__=='__main__': main()
