"""One matched higher-stiffness followup, preserving original torque authority.
Wave gain8 passed support/stepping/heading but failed actual material slip.
Scale32 tests the measured jointtracking/settledpose deficit; not a force-limit increase.
"""
import importlib.util
from pathlib import Path
import sys
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T081009Z/run_wave.py')
spec=importlib.util.spec_from_file_location('reviewed_wave_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
n.OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T082037Z')
b.b.DiagnosticActuator.scale=32.


def prereg():
    n.OUT.mkdir(parents=True,exist_ok=True)
    path=n.OUT/'PREREGISTRATION.json'
    if path.exists():return
    n.dump(path,dict(importance='核心必需',stage='S5 matched wave tracking stiffness followup',
        reason='Gain8teacher passesD/allfeet/heading but contactslip .290-.577native/s vsbody .171; gain2to8 showed majorjointtrackingimprovement. Test oneadditionalstiffnessarm, no torqueauthorityincrease.',
        reference=str(n.GAIT),reference_sha256=n.sha(n.GAIT),frequency_hz=1.,
        change=dict(previous_scale=8.,current_scale=32.,kp='native_limit/.2*32',kd='native_limit*.005*sqrt32',
                    torque_cap='original unchanged',target_rate_cap_rad_s=30.),
        controller='teacher mode continuous algorithmdiagnostic; ann mode1steacher thenreplacementMLP only, no phase/lookup/clock/rootassist afterhandover',
        physics='same full49/48/42/71/free root/neq0, gravity[0,0,-9801], unchangedcontact/friction',
        assessment='same fullreleaseDgates, sustained allsixlegs, actualsampledmaterialslip, exactreplay, straightness/heading; no successfromteacher',
        sources={str(p):n.sha(p) for p in [Path(__file__),SOURCE,Path(n.__file__),n.GAIT]},
        units='native; absoluteSIforce NOT_ESTABLISHED'))


n.prereg=prereg

if __name__=='__main__':
    prereg()
    if len(sys.argv)>1 and sys.argv[1]=='teacher':
        seconds=float(sys.argv[2]) if len(sys.argv)>2 else 4.
        tag=sys.argv[3] if len(sys.argv)>3 else 'teacher_gain32_4s'
        inert=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T073258Z/MODEL.npz')
        n.simulate(inert,tag,seconds,'teacher',False)
    else:n.main()
