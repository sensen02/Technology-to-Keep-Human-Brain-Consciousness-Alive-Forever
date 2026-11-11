"""Descriptive cycle-event summary only; does not alter thresholds or gates."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent

def main():
    result={}
    for case in ('halfspeed_straight','halfspeed_smallleft'):
        report=json.loads((HERE/case/'RESULT.json').read_text())
        per={}
        for leg in ('LF','LH','LM','RF','RH','RM'):
            ev=[e for e in report['cycles']['events'] if e['leg']==leg]
            per[leg]=dict(complete=sum(e['type']=='complete' for e in ev),
                abort_reasons={reason:sum(e['type']=='abort' and e.get('reason')==reason for e in ev) for reason in ('timeout','recontact','terminal')},
                event_max_lift=max((e.get('lift',0) for e in ev),default=0),
                event_max_advance=max((e.get('advance',0) for e in ev),default=0),
                events=ev)
        result[case]=per
    (HERE/'CYCLE_EVENT_SUMMARY.json').write_text(json.dumps(dict(cases=result,
        interpretation='Read-only event summaries. Lift/advance values are event-window maxima and do not establish completion without unchanged loaded-grounded dwell before timeout. No causal root explanation established by this descriptive summary.'),indent=2)+'\n')
    print(json.dumps({case:{leg:{k:v for k,v in data.items() if k!='events'} for leg,data in per.items()} for case,per in result.items()},indent=2))
if __name__=='__main__':main()
