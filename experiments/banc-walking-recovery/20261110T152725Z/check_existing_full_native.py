"""Read-only streaming validation on completed failed300 native evidence.
Reuses all 602 immutable chunks, retains only capped scalar quantile channels.
No neural archive exists for this history, so this is NOT full wrapper acceptance.
"""
import json
from pathlib import Path
import final_native_audit as a

run=a.ROOT/'20261110T114148Z/EVAL_GRU_ROLE_NATIVE'
result=a.read_json(run/'RESULT.json')
legacy=a.read_json(run/'NATIVE_CHECKS.json')
windows=a.recorded_windows(run)
report=a.extra_native(result['native'],.002494271478*9801.,run/'trace_1ms.npz',windows=windows)
a.require(report['windows']['auxiliary']['D6_sensitivity'][0]['samples']==384000,'AUX incomplete')
a.require(report['windows']['autonomous']['D6_sensitivity'][0]['samples']==768000,'ANN incomplete')
comparisons={}
for key in a.WINDOWS:
    rows=report['windows'][key]
    primary=next(r for r in rows['D6_sensitivity'] if r['tau_ms']==1.)
    old=legacy['windows'][key]
    signed=abs(primary['signed_p95_W']-old['original_D6_signed_p95_W'])
    absolute=abs(primary['absolute_p95_W']-old['original_D6_absolute_p95_W'])
    a.require(signed<1e-10 and absolute<1e-10,'D6 differs from historical frozen result '+key)
    comparisons[key]={'signed_p95_difference':signed,'absolute_p95_difference':absolute,
        'D7_median_at_0_1W':rows['D7_sensitivity'][-1]['median']}
a.require(comparisons['autonomous']['D7_median_at_0_1W']==4.,'historical D7 .1W median mismatch')
output={'status':'PASS','physical_steps':0,'historical_writes':0,'source':str(run),
    'source_result_sha256':a.sha(run/'RESULT.json'),'legacy_checks_sha256':a.sha(run/'NATIVE_CHECKS.json'),
    'comparison_to_frozen_legacy':comparisons,'diagnostics':report,'actual_windows':windows,
    'native_phase_counts':{'settle':80000,'auxiliary':384000,'autonomous':768000},
    'precision_correction':'nominal15.4 could omit final interval; use actual recorded source endpoints, all declared steps retained; earlier diagnostic preserved separately',
    'scope':'full native streaming logic verified on failed300 only; no neural archive; slip failure remains unchanged; not final candidate acceptance'}
path=Path(__file__).parent/'EXISTING_FULL_NATIVE_CHECK.json'
with path.open('x') as f:json.dump(output,f,indent=2)
print(json.dumps({'status':'PASS','output':str(path),'samples':report['samples'],
    'native_trace_endpoints':report['native_trace_endpoints_bitexact'],'comparison':comparisons,'physical_steps':0}),flush=True)
