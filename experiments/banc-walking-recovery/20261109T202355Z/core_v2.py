"""Isolated physics fix: xfrc_applied torque must be about body COM.
Only adhesion wrench reference changed; no new support forces, model or actuator changes.
"""
import numpy as np
from frame_audit import rc

def __getattr__(name):
    return getattr(rc,name)

class Core(rc.Core):
    def apply_adhesion(self,loads):
        # Preserve old contact selection, force magnitudes, release and update timing.
        super().apply_adhesion(loads)
        f=self.d.xfrc_applied[:,:3].copy()
        self.d.xfrc_applied[:,3:6]+=np.cross(self.d.xpos-self.d.xipos,f)
