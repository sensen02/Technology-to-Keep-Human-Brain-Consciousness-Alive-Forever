"""Use passive native oral joints between genuine anterior taste contacts."""
import json
from pathlib import Path
import sys
import numpy as np
import run_recovery as platform_run
from context import tr

def main():
    active=platform_run.active_build
    switch=[]
    def build(crop):
        env,bank=active(crop)
        # Reset/recompile happens in the common runner. Bind actuator identities only afterwards.
        base_start=platform_run.tr.fit_spawn
        def configure(crop,env,base):
            q=base_start(crop,env,base)
            phy=env.physics;m=phy.model.ptr
            native=env.task.before_step
            parts={'rostrum','haustellum','haustellum_abduct','labrum_left','labrum_right',
                   'adhere_labrum_left','adhere_labrum_right'}
            actuators=[i for i in range(m.nu) if m.actuator(i).name.removeprefix('walker/') in parts]
            names=[m.actuator(i).name for i in actuators]
            assert {'walker/rostrum','walker/haustellum','walker/labrum_left','walker/labrum_right'}<=set(names)
            affected=np.array(actuators)
            gains=m.actuator_gainprm[affected].copy();bias=m.actuator_biasprm[affected].copy()
            foot={i for i in range(m.ngeom) if any(n in m.body(m.geom_bodyid[i]).name for n in ('tarsus','claw'))}
            active_now=None
            def toggle(on):
                nonlocal active_now
                if active_now==on:return
                m.actuator_gainprm[affected]=gains*int(on)
                m.actuator_biasprm[affected]=bias*int(on)
                active_now=on
                switch.append([float(phy.time()),int(on)])
            toggle(False)
            def update(phy,action,random_state):
                touching=any(c.efc_address>=0 and ((c.geom1 in foot and c.geom2 in bank.skin_ids) or
                    (c.geom2 in foot and c.geom1 in bank.skin_ids)) for c in phy.data.contact[:phy.data.ncon])
                toggle(touching)
                return native(phy,action,random_state)
            env.task.before_step=update
            meta.update(actuators=names,active_gain=gains.tolist(),active_bias=bias.tolist(),idle_gain=0.,idle_bias=0.)
            return q
        platform_run.tr.fit_spawn=configure
        return env,bank
    meta={};platform_run.active_build=build
    platform_run.run()
    folder=Path(sys.argv[2])
    meta['switches']=switch
    assert switch and switch[0][1]==0
    (folder/'ORAL_ACTUATORS.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
