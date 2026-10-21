"""Fixed physical spike times across dt; no gains or runtime changes.
Current bin-average filter vs end-time point-event analytic reference, explicitly
alternative interpretation of spikes, not automatic permission to replace decoder.
"""
import json
import math
from pathlib import Path
import numpy as np
from engine.embodied.motor_output import MotorOutput, MotorOutputConfig
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/model_branches/decoder_clock_response'


def compare(dt, events=(.010,.020), end=.025):
    cfg=MotorOutputConfig()
    manifest={'n_neurons':1,'n_actuators':1,'rows':[{'actuator_index':0,'local_indices':[0]}]}
    decoder=MotorOutput(manifest,cfg)
    n=round(end/dt)
    sp=np.zeros((n,1),bool)
    for t in events:
        k=round(t/dt)-1
        if k<0 or k>=n or not math.isclose((k+1)*dt,t,abs_tol=1e-12):
            raise ValueError('event not on grid')
        sp[k,0]=True
    command,trace=decoder.consume(sp,spike_dt_s=dt,start_s=0.,end_s=end)
    reference=sum(math.exp(-(end-t)/cfg.filter_tau_s)/cfg.filter_tau_s for t in events)
    rate=trace['rates_hz'][0]
    return dict(dt_s=dt,events_s=list(events),end_s=end,rate_hz=rate,
        point_event_reference_hz=reference,relative_rate_error=rate/reference-1,
        command=float(command[0]),single_spike_bin_gain_hz=(1-math.exp(-dt/cfg.filter_tau_s))/dt,
        single_spike_point_event_gain_hz=1/cfg.filter_tau_s)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[compare(dt) for dt in (.0005,.00025,.000125,.0000625)]
    result=dict(label='手工研究原型，不是成品',rows=rows,
        interpretation='Finite bin-average filter converges toward end-event impulse reference. This diagnostic changes no runtime decoder. Difference is modeling convention, not evidence that impulse reference is physiological truth.')
    (OUT/'clock_response.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
