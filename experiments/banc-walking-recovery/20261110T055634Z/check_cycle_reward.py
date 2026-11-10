"""Deterministic sensed traces only; no model/controller/actions or physics writes."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from cycle_reward import CycleTracker, CycleConfig, LEG_ORDER

HERE = Path(__file__).resolve().parent

class Trace:
    def __init__(self, gamma=1.):
        self.c = CycleTracker(dict(gamma=gamma))
        self.n=0; self.rows=[]; self.events=[]; self.samples=[]
    def step(self, n=1, leg=0, load=.2, height=0., x=0.):
        out=None
        for _ in range(n):
            self.n+=1
            a=np.full(6,.2); z=np.zeros(6); f=np.zeros((6,3))
            a[leg]=load; z[leg]=height; f[leg,0]=x
            sample=dict(time=self.n*.001,loads_W=a,padz=z,relativefoot=f)
            out=self.c.update(sample)
            self.rows.append(out); self.samples.append(sample); self.events.extend(out['events'])
        return out
    def finish(self):
        r=self.c.finalize()
        if self.rows:
            self.rows[-1]['shape_reward']+=r['shape_reward']
        self.events.extend(r['events'])
        return r
    def bonus(self):
        return sum(r['completion_reward'] for r in self.rows)
    def shape(self):
        return sum(self.c.config.gamma**i*r['shape_reward'] for i,r in enumerate(self.rows))

def run_checks():
    checks=[]
    def ok(name, condition, details=None):
        if not condition:
            raise AssertionError(name+': '+str(details))
        checks.append(dict(name=name,passed=True,details=details))

    a=Trace(); a.step(1000); a.finish()
    ok('stationary grounded no reward/event',a.bonus()==0 and abs(a.shape())<1e-12 and not a.events)

    a=Trace(); a.step(10,load=0.,height=.03,x=.1); a.step(10)
    ok('floating reset cannot start attempt',a.bonus()==0 and not a.events)

    a=Trace(); a.step(15); a.step(10,load=.02,height=0.,x=.10); a.step(15)
    a.finish()
    ok('ground dragging cannot complete',a.bonus()==0 and abs(a.shape())<1e-12 and any(e.get('reason')=='recontact' for e in a.events))

    a=Trace(); a.step(15); a.step(10,load=0.,height=.025,x=0.); a.step(15); a.finish()
    ok('hopping without advance cannot complete',a.bonus()==0 and abs(a.shape())<1e-12)

    a=Trace(); a.step(15); a.step(5,load=0.,height=.02); a.step(100,load=0.,height=.03,x=.04)
    peak=max(r['potential'] for r in a.rows)
    a.finish()
    ok('holding qualified foot airborne never pays completion',a.bonus()==0 and abs(a.shape())<1e-12)
    ok('per leg potential cap',peak<=.05+1e-12)
    ok('terminal clearing idempotent',a.c.potential==0 and a.c.finalize()['shape_reward']==0)

    a=Trace(); a.step(15); a.step(5,load=0.,height=.02); a.step(850,load=0.,height=.03,x=.05); a.step(15); a.finish()
    ok('long timeout aborts and cannot later claim completion',a.bonus()==0 and any(e.get('reason')=='timeout' for e in a.events) and abs(a.shape())<1e-12)

    a=Trace(); a.step(15); a.step(5,load=0.,height=.02,x=0.); a.step(10,load=0.,height=.03,x=.04)
    a.step(9,x=.04)
    ok('reload dwell 9ms insufficient',a.bonus()==0)
    a.step(1,x=.04); a.step(100,x=.04)
    complete=[e for e in a.events if e['type']=='complete']
    a.finish()
    ok('complete cycle exactly once',a.bonus()==.1 and len(complete)==1 and complete[0]['leg']=='LF' and abs(a.shape())<1e-12,complete)

    a=Trace(); a.step(15)
    for _ in range(50):
        a.step(3,load=.02,height=.03,x=.04); a.step(3,load=.2)
    a.finish()
    ok('load chatter cannot bypass 5ms unload',a.bonus()==0 and abs(a.shape())<1e-12)

    a=Trace(); a.step(15)
    a.step(5,load=0.,height=.02,x=0.); a.step(10,load=0.,height=.03,x=.04)
    p1=a.c.potential
    a.step(1,load=0.,height=.02,x=.005)
    ok('reversed physical advance decreases potential',a.c.potential<p1)
    a.finish()

    # Geometry advance on ground before lift must not count as airborne advance.
    a=Trace(); a.step(15); a.step(5,load=0.,height=0.,x=0.)
    a.step(5,load=0.,height=0.,x=.1); a.step(5,load=0.,height=.03,x=.005); a.step(15); a.finish()
    ok('advance measured only while unloaded and clear',a.bonus()==0)

    # Chattering reload must accumulate a CONTINUOUS 10ms; then a new cycle
    # requires new sensed unload/clear/advance rather than a cooldown timer.
    a=Trace(); a.step(15); a.step(5,load=0.,height=.02)
    a.step(10,load=0.,height=.03,x=.04)
    for _ in range(5):
        a.step(9,x=.04); a.step(1,load=.07,height=.01,x=.04)
    ok('reload chatter cannot accumulate separated dwell',a.bonus()==0)
    a.step(10,x=.04)
    a.step(5,load=0.,height=.02,x=0.)
    a.step(10,load=0.,height=.03,x=.04); a.step(10,x=.04); a.finish()
    ok('two real cycles pay twice without encoded cooldown',a.bonus()==.2 and len([e for e in a.events if e['type']=='complete'])==2)

    a=Trace(); a.step(15)
    for _ in range(20):
        a.step(5,load=0.,height=.02); a.step(5,load=0.,height=.03,x=.005); a.step(15)
    a.finish()
    ok('repeated failed attempts cannot farm shaping',a.bonus()==0 and abs(a.shape())<1e-12)

    a=Trace(); a.step(9,load=.2); a.step(5,load=.08)
    ok('before loaded arming dwell no discovery reward',a.c.potential==0)

    # Subthreshold load changes have a continuous, reversible discovery signal.
    a=Trace(); a.step(15); a.step(10,load=.15)
    mid=a.c.potential
    a.step(10,load=.2); a.finish()
    ok('smooth armed unloading before threshold',0<mid<.01 and a.bonus()==0 and abs(a.shape())<1e-12)

    for i,leg in enumerate(LEG_ORDER):
        a=Trace(); a.step(15); a.step(5,leg=i,load=0.,height=.02)
        a.step(10,leg=i,load=0.,height=.03,x=.04); a.step(10,leg=i,x=.04)
        complete=[e for e in a.events if e['type']=='complete']
        ok('actuator order '+leg,len(complete)==1 and complete[0]['leg']==leg and a.bonus()==.1)

    a=Trace(); r=a.step(15); sample=a.samples[-1]
    duplicate=a.c.update(sample)
    ok('identical repeated timestamp zero idempotent',duplicate['total_reward']==0 and duplicate['diagnostics']['duplicate'])
    bad=dict(sample); bad['loads_W']=np.zeros(6)
    try: a.c.update(bad)
    except ValueError: rejected=True
    else: rejected=False
    ok('conflicting repeated timestamp rejected',rejected)
    for name,change in [('negative load',dict(loads_W=np.full(6,-1.))),
                        ('NaN',dict(padz=np.full(6,np.nan))),
                        ('shape',dict(relativefoot=np.zeros((5,3)))),
                        ('skipped time',dict(time=a.c.time+.002))]:
        bad=dict(time=a.c.time+.001,loads_W=np.ones(6),padz=np.zeros(6),relativefoot=np.zeros((6,3)))
        bad.update(change)
        try: a.c.update(bad)
        except ValueError: rejected=True
        else: rejected=False
        ok(name+' rejected',rejected)

    # Copied sensing values cannot be retroactively changed by caller buffers.
    a=Trace(); a.step(15); sample=a.samples[-1]
    sample['loads_W'][:]=0.
    ok('stored sensing detached from input buffers',np.all(a.c._last[0]==.2))

    # 1ms discounted telescope, including terminal correction on SAME transition.
    a=Trace(gamma=math.exp(-.001)); a.step(15); a.step(5,load=0.,height=.02)
    a.step(70,load=0.,height=.03,x=.04); a.finish()
    ok('gamma discounted terminal telescope',abs(a.shape())<1e-12,a.shape())

    # Parent contract: physical potential observed, shaping evaluated only per5ms.
    a=Trace(gamma=math.exp(-.001)); a.step(15); a.step(5,load=0.,height=.02)
    a.step(30,load=0.,height=.03,x=.04)
    gp=a.c.config.gamma**5; prev=0.; rewards=[]
    for k in range(4,len(a.rows),5):
        nxt=a.rows[k]['potential'] if k<len(a.rows)-1 else 0.
        rewards.append(gp*nxt-prev); prev=nxt
    a.c.finalize()
    total=sum(gp**i*r for i,r in enumerate(rewards))
    ok('parent5ms boundary potential telescope terminal',abs(total)<1e-12,total)

    a=Trace(); a.step(15); a.step(5,load=0.,height=.015)
    a.step(10,load=0.,height=.015,x=.04); a.step(10); a.finish()
    ok('clearance threshold strictly greater not equal',a.bonus()==0)

    c=CycleTracker(dict(gamma=1.)); ev=[]; maxphi=0.
    for n in range(1,41):
        low=16<=n<=30
        z=np.full(6,.03 if low else 0.)
        x=.04 if n>=21 else 0.
        feet=np.zeros((6,3)); feet[:,0]=x
        out=c.update(dict(time=n*.001,loads_W=np.full(6,0. if low else .2),padz=z,relativefoot=feet))
        ev.extend(out['events']); maxphi=max(maxphi,c.potential)
    ok('six symmetric trackers no tripod restriction',len([e for e in ev if e['type']=='complete'])==6 and maxphi<=.30+1e-12)

    # Native loads wrapper and normalized loads have same event output.
    class Data: pass
    class Core:
        def __init__(self):
            self.d=Data(); self.d.time=.001; self.d.subtree_com=np.array([[0.,0.,1.4]])
            self.th=0
        def measure(self):
            return dict(loads=np.full(6,.2*24.446354755878005),padz=np.zeros(6),foot=np.zeros((6,3)))
    c=CycleTracker(); r=c.measure(Core())
    ok('measure wrapper sensed native force only',r['potential']==0 and c.time==.001)
    report=dict(label='curriculum proxy sensing checks; not walking proof',
                all_passed=all(x['passed'] for x in checks),check_count=len(checks),
                config=CycleConfig().__dict__,leg_order=LEG_ORDER,
                api=dict(parent='potential boundary gamma_policy*phi_after-phi_before; terminal after0; completion block lump .1 per complete event',
                         local='gamma*phi_next-phi_prev every1ms; finalize correction added same last transition; do not double count parent'),
                sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'cycle_reward.py',Path(__file__)]},
                 source_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'cycle_reward.py',Path(__file__)]},checks=checks)
    (HERE/'CYCLE_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(all_passed=report['all_passed'],check_count=len(checks))))

if __name__=='__main__':
    run_checks()
