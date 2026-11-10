"""Numerical proposal acceptance correction; all physical/geometry gates unchanged.

Count-checked adaptation of frozen100212. Status MAX_ITER is usable only with
finite primal-feasible/trust-bounded proposals and actual nonlinear descent.
"""
import hashlib
from pathlib import Path
import types

BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T100212Z/prepare_rate_path.py')
BASE_SHA256='954f76e87ad24007e4c1c24b113b51ef9c5a8370dc62bd54fcabc5e7f3924463'
HERE=Path(__file__).resolve()
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T101114Z')
source=BASE.read_text()
assert hashlib.sha256(BASE.read_bytes()).hexdigest()==BASE_SHA256,'Frozen baseline changed'
patches=[
    ("OUT=INPUT.parent/'20261107T100212Z'","OUT=INPUT.parent/'20261107T101114Z'"),
    ('files=[Path(__file__),SOURCE,b.OLD,',"files=[Path(__file__),Path('"+str(HERE)+"'),SOURCE,b.OLD,"),
    ("            log=dict(iteration=iteration,status=result.info.status,solver_iterations=int(result.info.iter),",
     """            log=dict(iteration=iteration,status=result.info.status,solver_iterations=int(result.info.iter),
                status_val=int(result.info.status_val),primal_residual=float(result.info.prim_res),
                dual_residual=float(result.info.dual_res),duality_gap=float(result.info.duality_gap),
                inexact_accepted=False,"""),
    ('result.info.status_val not in (1,2):','result.info.status_val not in (1,2,7):'),
    ("            if check['constraint_violation']>1e-8:",
     """            log['proposed_primal_constraint_violation']=check['constraint_violation']
            log['proposed_native_bound_excess']=check['native_bound_excess']
            log['proposed_trust_step_max']=float(abs(candidate-x).max())
            if (check['constraint_violation']>1e-8 or check['native_bound_excess']>1e-8
                or result.info.prim_res>1e-8
                or np.max(abs(candidate-x))>PARAM['trust_step_rad']+1e-8):"""),
    ('            log.update(accepted_fraction=fraction,step_max=stepmax,objective_after=values[0],',
     '            log.update(inexact_accepted=result.info.status_val==7,accepted_fraction=fraction,step_max=stepmax,objective_after=values[0],'),
    ("all(l['knot_speed_max']<=20+1e-7 for l in logs)",
     "all(l['constraint_violation']<=1e-8 for l in logs)"),
    ("version='lift-first-rate-constrained-v3'","version='lift-first-rate-constrained-v3-numerical-acceptance'"),
    ('preparation=str(Path(__file__).resolve()),preparation_sha256=hashes[str(Path(__file__))]',
     "preparation='"+str(HERE)+"',preparation_sha256=hashes['"+str(HERE)+"']"),
]
effective=source
for old,new in patches:
    assert effective.count(old)==1,('Expected one frozen source match',old)
    effective=effective.replace(old,new,1)
EFFECTIVE_SHA256=hashlib.sha256(effective.encode()).hexdigest()
runtime=types.ModuleType('numerical_acceptance_rate_path')
runtime.__file__=str(BASE)
exec(compile(effective,str(BASE),'exec'),runtime.__dict__)
original_dump=runtime.b.b.dump


def dump(path,value):
    if path==OUT/'PREREGISTRATION.json':
        artifact=OUT/'EFFECTIVE_PREPARATION.py'
        if artifact.exists():raise FileExistsError(artifact)
        artifact.write_text(effective)
        value.update(stage='S5 rate-constrained trajectory numerical acceptance correction',
            numerical_acceptance=dict(status_vals=[1,2,7],initialization_status_vals=[1,2],
                rule='Finite proposedjointcurve; native/primalconstraintviolations<=1e-8, solverprimalres<=1e-8, truststep<=.05+1e-8; then actualnonlinearobjectivebacktracking[1,.5,.25]mustdecreaseobjective.',
                motivation='Frozen100212LH/RHfirstGNmaxiterproposals hadprimalres1.46e-9/2.83e-9 anddualres~1e-8, satisfiedactualconstraints anddecreasednonlinearobjective, butstatus-onlyrejectionstoppedthem.',
                logging='Everyiterstatus/status_val/primalres/dualres/gap; inexact_accepted trueonlywhenMAX_ITERproposalactuallyaccepted.',
                knot_constraint_units='Finalinternalprimalconstraintcheck1e-8rad matchesGNacceptance;20rad/s knotbound hasnumericalconversiontolerance5.12e-6rad/s. ActualexactPCHIPgate30rad/s unchanged.',
                unchanged='Freshsame095519initialprojection foreachallsixlegs; max30GN,solvereps1e-9/maxiter20000,sourceprior.001,D2.01,trust.05,geometry/lift/frequency/rate/nativebounds<=1e-9; nofilters/reprojection/extracycles.'),
            adaptation=dict(baseline=str(BASE),baseline_sha256=BASE_SHA256,
                wrapper=str(HERE),wrapper_sha256=runtime.b.b.sha(HERE),effective_source=str(artifact),
                effective_sha256=EFFECTIVE_SHA256,patches=[dict(old=old,new=new,required_matches=1) for old,new in patches]))
    original_dump(path,value)


runtime.b.b.dump=dump


if __name__=='__main__':
    raise SystemExit(runtime.main())
