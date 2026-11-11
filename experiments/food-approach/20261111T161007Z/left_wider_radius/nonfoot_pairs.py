"""Bounded native nonfoot attribution; no MuJoCo import until actual solved()."""
import numpy as np


class PairPeaks:
    """At most64 geom pairs; counts contact observations, not distinct steps."""
    def __init__(self,limit=64):
        if limit!=64:raise ValueError('Preregistered pair bound is64')
        self.limit=limit;self.pairs={};self.overflow_count=0;self.overflow_peak=None
        self.positive_native_steps=0;self.post_positive_native_steps=0

    def observe(self,record):
        key=':'.join(map(str,sorted(record['geom_ids'])))
        if key not in self.pairs:
            if len(self.pairs)>=self.limit:
                self.overflow_count+=1
                if self.overflow_peak is None or record['positive_world_z_native']>self.overflow_peak['positive_world_z_native']:
                    self.overflow_peak=dict(record)
                return
            self.pairs[key]=dict(contact_observations=0,post_contact_observations=0,first_time_before_s=record['time_before_s'],last_time_before_s=record['time_before_s'],peak=None,post_peak=None)
        pair=self.pairs[key];pair['contact_observations']+=1;pair['last_time_before_s']=record['time_before_s']
        if pair['peak'] is None or record['positive_world_z_native']>pair['peak']['positive_world_z_native']:pair['peak']=dict(record)
        if record['time_before_s']>=5.8-1e-8:
            pair['post_contact_observations']+=1
            if pair['post_peak'] is None or record['positive_world_z_native']>pair['post_peak']['positive_world_z_native']:pair['post_peak']=dict(record)

    def summary(self):
        return dict(max_geom_pairs=self.limit,pairs=self.pairs,overflow_contact_observations=self.overflow_count,
            overflow_peak=self.overflow_peak,positive_native_steps=self.positive_native_steps,post_positive_native_steps=self.post_positive_native_steps,
            time_convention='Contact force/pos are beginning-of-step integration solve; time_before_s is parent before(), time_after_s is d.time before mj_forward',
            force_convention='positive world Z on nonfoot geom, same signed convention as parent PeakRecorder',
            counts='per-pair contact_observations may exceed native-step count if multiple points for a pair',full_native_arrays=False)


def recorder_class(parent_recorder):
    class NonfootPairRecorder(parent_recorder):
        def __init__(self):
            super().__init__();self.pair_peaks=PairPeaks()

        def solved(self,c,before):
            total=super().solved(c,before)
            if total<=0.:return total
            # Only real-run positive native solves import/query MuJoCo.
            import mujoco as mj
            self.pair_peaks.positive_native_steps+=1
            if before>=5.8-1e-8:self.pair_peaks.post_positive_native_steps+=1
            for ci in range(c.d.ncon):
                contact=c.d.contact[ci];ga,gb=int(contact.geom1),int(contact.geom2)
                if not(c.planes[ga] or c.planes[gb]):continue
                geom=gb if c.planes[ga] else ga
                if int(c.geom_leg[geom])>=0:continue
                f=np.zeros(6);mj.mj_contactForce(c.plant,c.d,ci,f)
                on_nonfoot=(contact.frame.reshape(3,3).T@f[:3])*(1 if c.planes[ga] else -1)
                positive=max(0.,float(on_nonfoot[2]))
                if positive==0.:continue
                body=int(c.plant.geom_bodyid[geom])
                self.pair_peaks.observe(dict(geom_ids=[ga,gb],
                    geom_names=[c.plant.geom(ga).name or '',c.plant.geom(gb).name or ''],
                    nonfoot_geom_id=geom,nonfoot_geom_name=c.plant.geom(geom).name or '',
                    nonfoot_body_id=body,nonfoot_body_name=c.plant.body(body).name or '',
                    time_before_s=float(before),time_after_s=float(c.d.time),positive_world_z_native=positive,
                    contact_normal_force_native=float(f[0]),force_world_on_nonfoot_native=on_nonfoot.tolist(),
                    contact_position_world=np.array(contact.pos,copy=True).tolist()))
            return total

        def close(self):
            info=super().close();info['nonfoot_contact_pairs']=self.pair_peaks.summary();return info
    return NonfootPairRecorder
