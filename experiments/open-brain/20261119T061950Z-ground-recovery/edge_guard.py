"""Keep near-surface speed low once approach or asymmetric support is detected."""
import numpy as np


class EdgeGuard:
    def __init__(self, phy):
        model=phy.model
        self.th=model.name2id('walker/thorax','body')
        self.leg=[]
        for i in range(model.ngeom):
            n=model.id2name(model.geom_bodyid[i],'body') or ''
            front='T1' in n and ('claw' in n or 'tarsus' in n)
            self.leg.append((n.startswith('walker/'), n.rsplit('_',1)[-1] if front else None))
        self.close_at=-np.inf
        self.rows=[]

    def commands(self,phy,chosen,speed,turn):
        top,side=set(),set()
        for c in phy.data.contact[:phy.data.ncon]:
            if c.efc_address<0:
                continue
            a,b=self.leg[int(c.geom1)],self.leg[int(c.geom2)]
            if a[0] and b[0]:
                continue
            label=a[1] or b[1]
            if label:
                (top if abs(c.frame[2])>.55 else side).add(label)
        t=phy.time()
        close=False
        if chosen and chosen['box']:
            u,_,v,_=chosen['box']
            close=v-u>=360 and chosen['horiz']<=.74
        if close or (side and top):
            self.close_at=t
        armed=t-self.close_at<.32
        up=phy.data.xmat[self.th,8]
        clipped=min(speed,.492) if armed else speed
        self.rows.append((t,float(close),len(top),len(side),up,float(armed),speed,clipped,turn))
        return clipped,turn
