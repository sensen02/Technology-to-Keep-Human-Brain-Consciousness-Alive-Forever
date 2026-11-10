"""One-variable offline lift experiment: .25 -> .18, all other gates unchanged.

No physical integration. Uses accepted101114 trajectory for initialization, exact
083316 stance poses, free-air orientation and unchanged bounded trajectory GN.
"""
import hashlib
import importlib.util
from pathlib import Path
import types

SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T101114Z/prepare_rate_path.py')
SOURCE_SHA256='9b2f9dfa771c90fd7414e03f6426850da7575c72d6d7f9543f009b64cdaec338'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA256,'Frozen numerical solver changed'
spec=importlib.util.spec_from_file_location('lower_lift_accepted_solver',SOURCE)
accepted=importlib.util.module_from_spec(spec);spec.loader.exec_module(accepted)
assert accepted.EFFECTIVE_SHA256=='8beec5fe6c39fae22c66e270128bb104e8a4a16f463699c76eca2ec9474ce877'
HERE=Path(__file__).resolve()
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T114444Z')
WARMSTART=OUT.parent/'20261107T101114Z/TRAJECTORY.npz'
patches=[
    ("OUT=INPUT.parent/'20261107T101114Z'","OUT=INPUT.parent/'20261107T114444Z'"),
    ('files=[Path(__file__),',"files=[Path('"+str(HERE)+"'),Path('"+str(SOURCE)+"'),Path('"+str(WARMSTART)+"'),Path(__file__),"),
    ('start=time.monotonic();prep=b.Preparation()',
     'start=time.monotonic();prep=LowerPreparation()'),
    ("target=E['target_world'][::8];seed=C['source']+C['final'];qknots=prep.old.copy()",
     """target=E['target_world'][::8].copy()
    for leg in range(6):
        swing=~prep.stance[:,leg]
        target[swing,leg,2]=prep.stance_z[leg]+.72*(target[swing,leg,2]-prep.stance_z[leg])
    seed=b.b.load(Path('"""+str(WARMSTART)+"""'))['qpos_hinges'];qknots=prep.old.copy()"""),
    ("version='lift-first-rate-constrained-v3-numerical-acceptance'",
     "version='lift-first-rate-constrained-lower-lift-0p18'"),
    ('    if result[\'passed\']:\n        np.savez_compressed',
     "    meta['parameters']=dict(meta['parameters'],clearance=.18)\n    meta['lift_experiment']=dict(old=.25,new=.18,swing_z_scale=.72,warmstart='"+str(WARMSTART)+"')\n    if result['passed']:\n        np.savez_compressed"),
    ("preparation='"+str(SOURCE)+"',preparation_sha256=hashes['"+str(SOURCE)+"']",
     "preparation='"+str(HERE)+"',preparation_sha256=hashes['"+str(HERE)+"']"),
]
effective=accepted.effective
for old,new in patches:
    assert effective.count(old)==1,('Expected one frozen effective-code match',old)
    effective=effective.replace(old,new,1)
EFFECTIVE_SHA256=hashlib.sha256(effective.encode()).hexdigest()
runtime=types.ModuleType('lower_lift_effective_solver');runtime.__file__=str(accepted.BASE)
exec(compile(effective,str(HERE)+'::effective','exec'),runtime.__dict__)
OriginalPreparation=runtime.b.Preparation


class LowerPreparation(OriginalPreparation):
    def target(self,phase,leg):
        original,goal,orientation,stance=super().target(phase,leg)
        if not stance:
            goal[2]=self.stance_z[leg]+.72*(goal[2]-self.stance_z[leg])
        return original,goal,orientation,stance


runtime.LowerPreparation=LowerPreparation
original_dump=accepted.original_dump


def dump(path,value):
    if path==OUT/'PREREGISTRATION.json':
        artifact=OUT/'EFFECTIVE_PREPARATION.py'
        if artifact.exists():raise FileExistsError(artifact)
        artifact.write_text(effective)
        value.update(stage='S5 ONElower-lift engineeringparameter geometry experiment',
            reason='Actualboundedoffsetsearches left hindlegmaterialslip near.7; halfcadence didnotfix it. TestONElowerairlegexcursion/base-impulsehypothesis; no claimedforce/friction/neuralimprovement.',
            height_change=dict(previous_peak=.25,new_peak=.18,swing_z_scale=.72,
                formula='Only nominalSWING: znew=quietstanceZ+.72*(101114desiredZ-quietstanceZ). ActualnominalSTANCECartesiantargets andjointknotsunchanged.',
                not_threshold_change='Actualwholefootplateauclearance gate remains .1nativeabovecommonstanceplane, position.005,exactrate30@1Hz,nativeboundtol1e-9.'),
            warmstart=dict(source=str(WARMSTART),sha256=runtime.b.b.sha(WARMSTART),
                method='Nearestnative/rateconstrainedprojectionofaccepted101114trajectory, withoutsearchoffsets. Parentmayapplyseparatelymeasured+.02RHtrochanteryaw onlyinitsphysicalruntime.'),
            unchanged='XYtargets,duty5/6,513grid,freeairorientation,source083316stanceknots,sourceprior.001,D2.01,trust.05,max30GN/leg,20rad/sknot,exact30PCHIP,1e-8primal+trust+descentinexactacceptance,solvereps/maxiter.',
            no_postprocessing='No filters, localreprojection, extraoutercycles, force/gain/friction/roottrajectorychanges.',
            adaptation=dict(source=str(SOURCE),source_sha256=SOURCE_SHA256,
                source_effective_sha256=accepted.EFFECTIVE_SHA256,wrapper=str(HERE),wrapper_sha256=runtime.b.b.sha(HERE),
                effective_source=str(artifact),effective_sha256=EFFECTIVE_SHA256,
                patches=[dict(old=old,new=new,required_matches=1) for old,new in patches]),
            physical_status='NOT_RUN; acceptedFKassetisnotphysicalteacheracceptanceorANNsuccess.')
        value['source_hashes'][str(accepted.BASE)]=runtime.b.b.sha(accepted.BASE)
    original_dump(path,value)


runtime.b.b.dump=dump


if __name__=='__main__':
    raise SystemExit(runtime.main())
