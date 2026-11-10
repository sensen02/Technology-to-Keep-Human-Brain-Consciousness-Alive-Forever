"""Finite-PD torque antiwindup: conditional target integration, not extra force.

Explicit interface change: original request is rate-limited, then outward target
increments are blocked when CURRENT raw PD demand is already at a native cap.
Inward reversal and exact zero commands remain valid. Physical data is read only.
Base last_input_rate_rad_s records EFFECTIVE input; last_requested_* retains intent.
No brain/control provenance is implied. Parent owns physics/verifier/trace saving.
"""
import importlib.util
from pathlib import Path
import numpy as np

ORIGINAL_SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
spec=importlib.util.spec_from_file_location('conditional_original_actuator',ORIGINAL_SOURCE)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
OriginalIncrementalActuator=original.IncrementalActuator
INSTANCES=[]


class ConditionalIntegrator(OriginalIncrementalActuator):
    """Original bounded torque(), bounds and gains; only update() is conditional."""
    def __init__(self,model,data,**kwargs):
        kwargs.setdefault('max_rate_rad_s',30.)
        super().__init__(model,data,**kwargs)
        if np.any(self.max_rate_rad_s>30.):raise ValueError('Rate cap must not exceed original30rad/s')
        self.data_reference=data
        self.logs=[];self.blocked_command_ticks=0;self.blocked_joint_updates=0
        self.last_requested_input_rate_rad_s=np.zeros(42)
        self.last_requested_limited_rate_rad_s=np.zeros(42)
        self.last_requested_rate_clip_mask=np.zeros(42,dtype=bool)
        self.last_torque_block_mask=np.zeros(42,dtype=bool)
        self.last_raw_torque_before=np.zeros(42)
        INSTANCES.append(self)

    def update(self,delta_rate_rad_s,dt_s):
        requested=np.asarray(delta_rate_rad_s,dtype=float)
        if requested.shape!=(42,) or not np.isfinite(requested).all():
            raise ValueError('Requested rate must be finite shape(42,)')
        requested=requested.copy()
        limited=np.clip(requested,-self.max_rate_rad_s,self.max_rate_rad_s)
        q,v=self._pose(self.data_reference)
        before=self.target_rad.copy()
        with np.errstate(over='raise',invalid='raise'):
            raw=self.kp*(before-q)-self.kd*v
        if not np.isfinite(raw).all():raise ValueError('Nonfinite raw PD demand')
        blocked=((raw>=self.torque_limit)&(limited>0))|((raw<=-self.torque_limit)&(limited<0))
        effective=np.where(blocked,0.,limited)
        target=super().update(effective,dt_s)  # Original dt/angle bounds and accounting.
        if not requested.any():assert np.array_equal(before,target),'Zero command moved target'
        self.last_requested_input_rate_rad_s=requested
        self.last_requested_limited_rate_rad_s=limited.copy()
        self.last_requested_rate_clip_mask=requested!=limited
        self.last_torque_block_mask=blocked.copy();self.last_raw_torque_before=raw.copy()
        self.blocked_command_ticks+=int(blocked.any());self.blocked_joint_updates+=int(blocked.sum())
        self.logs.append(dict(time=float(self.data_reference.time),command_tick=self.command_ticks,
            requested_rate=requested.copy(),limited_requested_rate=limited.copy(),effective_input_rate=effective.copy(),
            requested_rate_clip_mask=self.last_requested_rate_clip_mask.copy(),torque_block_mask=blocked.copy(),
            raw_torque_before=raw.copy(),target_before=before,target_after=target.copy(),
            blocked_command_ticks=self.blocked_command_ticks,blocked_joint_updates=self.blocked_joint_updates))
        return target

    def state_dict(self):
        result=super().state_dict()
        result.update(integration_contract='rate clip then CURRENT raw-PD native-cap outward-only block; inward reversal always allowed',
            original_source=str(ORIGINAL_SOURCE),blocked_command_ticks=self.blocked_command_ticks,
            blocked_joint_updates=self.blocked_joint_updates,base_last_input_semantics='effective limited/unblocked input')
        for key in ('last_requested_input_rate_rad_s','last_requested_limited_rate_rad_s',
                    'last_requested_rate_clip_mask','last_torque_block_mask','last_raw_torque_before'):
            result[key]=getattr(self,key).tolist()
        return result
