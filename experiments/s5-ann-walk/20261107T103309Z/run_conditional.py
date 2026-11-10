"""Matched physical test of torque-saturation conditional integration.
Target memory/zero-output behavior remains; original native torque/rate caps remain.
Conditional integration is enabled only at the declared1s handover, not in warmup.
"""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
from conditional_integrator import ConditionalIntegrator,OriginalIncrementalActuator,INSTANCES
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T102842Z/run_student.py')
spec=importlib.util.spec_from_file_location('current_lift_student_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T103309Z')
n.OUT=OUT
MODEL=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T102842Z/MODEL.npz')


class ConditionalGain128(ConditionalIntegrator):
    def __init__(self,model,data,**kwargs):
        kwargs['kp']=np.asarray(kwargs['kp'])*64.
        kwargs['kd']=np.asarray(kwargs['kd'])*np.sqrt(64.)
        super().__init__(model,data,**kwargs)

    def update(self,rate,dt):
        enabled=float(self.data_reference.time)>=1.-1e-9
        if enabled:
            target=super().update(rate,dt)
            self.logs[-1]['enabled']=True
            return target
        # Exact original actuator behavior through shared physical warmup.
        requested=np.asarray(rate,dtype=float).copy()
        limited=np.clip(requested,-self.max_rate_rad_s,self.max_rate_rad_s)
        before=self.target_rad.copy();q,v=self._pose(self.data_reference)
        raw=self.kp*(before-q)-self.kd*v
        target=OriginalIncrementalActuator.update(self,requested,dt)
        self.last_requested_input_rate_rad_s=requested.copy()
        self.last_requested_limited_rate_rad_s=limited.copy()
        self.last_requested_rate_clip_mask=requested!=limited
        self.last_torque_block_mask=np.zeros(42,bool);self.last_raw_torque_before=raw.copy()
        self.logs.append(dict(time=float(self.data_reference.time),command_tick=self.command_ticks,
            requested_rate=requested,limited_requested_rate=limited,effective_input_rate=limited.copy(),
            requested_rate_clip_mask=requested!=limited,torque_block_mask=np.zeros(42,bool),
            raw_torque_before=raw,target_before=before,target_after=target.copy(),
            blocked_command_ticks=self.blocked_command_ticks,blocked_joint_updates=self.blocked_joint_updates,enabled=False))
        return target


n.IncrementalActuator=ConditionalGain128


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['teacher','ann','hold','motor_cut'],default='teacher')
    ap.add_argument('--seconds',type=float,default=4.);ap.add_argument('--tag',default='teacher_conditional_4s')
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    prereg=OUT/(args.tag+'_PROTOCOL.json')
    if prereg.exists():raise FileExistsError('preserve prior conditional diagnostic')
    n.dump(prereg,dict(importance='核心必需',stage='S5 matched torque-windup conditional integrator diagnostic',
        previous='Currentbroadlift gain128teacher stillfailsLH/RHslip; RHmiddleSTANCEnativecaps repeated150/140ms, rawPDdemandupto25.5xnativecap. Higherblanketgainnotjustified.',
        interface_change='Originalincrementalrate clip thenblockoutward-only rate whenCURRENTrawPD>=nativecap. Inwardreversal/zero unchanged; no targetprojectiontowardphysicalpose, no body/rootstatewrites.',
        original_interface_retained='42rate inputs, targetmemory,1msintegration,nativejointbounds andsamefinitePDtorque(), gain128 andoriginalnativeτcaps/30rad/s unchanged',
        activation='Onlyfromt>=1s; first1s exactlyoriginalwarmup andrealpolicyhistory, no reset atactivation',
        mode=args.mode,total_seconds=args.seconds,teacher_seconds=1.,
        controller='Inteacher modecontinuousdeclaredgaitreference; inannmodefrozen102842weights onlyafter1s, no teacher/clock/reference/anchors atdeployment',
        model=str(MODEL),model_sha256=n.sha(MODEL),reference=str(n.GAIT),reference_sha256=n.sha(n.GAIT),
        assessment='OriginalwholeDgates/realsteps/heading/slip<=.5body perleg; independentreplayMUSTreconstructtorqueblockfromactualq/v every1ms, nottrustsavedmaskonly. Zero-ratecutstillholdstargetexactly.',
        integration_trace='ANTIWINDUP.npz every1ms recordsrequested/effective/mask/rawPD/targetbefore/after/enabled. COMMANDS keepsoriginalrequest. No hiddencommandsubstitution.',
        sources={str(p):n.sha(p) for p in [Path(__file__),Path(__file__).with_name('conditional_integrator.py'),SOURCE,Path(n.__file__),n.GAIT,MODEL]},
        limits='No addedforce/friction/gravity/inertia/rootassist. Thisisexplicitnewlocalservointerfacevariant, notsame-hasholdactuator.',
        units='native; biologicalvalidity/originalbrainnecessity/absoluteSIforce NOT_ESTABLISHED'))
    result=None
    try:result=n.simulate(MODEL,args.tag,args.seconds,args.mode,False)
    finally:
        logs=[row for a in INSTANCES for row in a.logs]
        if logs:
            folder=OUT/args.tag
            np.savez_compressed(folder/'ANTIWINDUP.npz',**{k:np.asarray([row[k] for row in logs]) for k in logs[0]})
            if result is not None:
                result.update(actuator_interface='conditional_raw_PD_outward_block',conditional_from_s=1.,gain_scale=128.,
                    conditional_ticks=int(sum(row['enabled'] for row in logs)),
                    torque_blocked_joint_updates=int(sum(np.asarray(row['torque_block_mask']).sum() for row in logs)),
                    antiwindup_sha256=n.sha(folder/'ANTIWINDUP.npz'))
                n.dump(folder/'RESULT.json',result)


if __name__=='__main__':main()
