"""Fixed-shape raw sensory inputs for a future trainable afferent translator.
No policy, neural current, normalization learned on future data, or motor output.
Missing modalities are separate masks, not fabricated zero-valued observations.
"""
import numpy as np


class SensoryFeatures:
    VERSION = 'raw_sensory_v2'
    CONTACT_FIELDS = ('touch', 'normal_force', 'tangent_force',
                      'force_x', 'force_y', 'force_z', 'position_x', 'position_y', 'position_z',
                      'normal_x', 'normal_y', 'normal_z',
                      'slip_velocity_x', 'slip_velocity_y', 'slip_velocity_z', 'slip_speed')
    JOINT_FIELDS = ('position_rad', 'reference_deviation_rad', 'velocity_rad_s')

    def __init__(self, sensing):
        self.sensing = sensing
        m = sensing.m
        self.sites = sorted({m.geom(i).name for i,(region,_) in sensing.geom_regions.items() if region is not None})
        import mujoco
        self.joints = [m.joint(j).name for j in range(m.njnt)
                       if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        self.names = [f'{kind}/{site}/{field}' for kind in ('external','self')
                      for site in self.sites for field in self.CONTACT_FIELDS]
        self.names += [f'joint/{j}/{field}' for j in self.joints for field in self.JOINT_FIELDS]
        self.names += ['unmodelled/'+s for s in ('airflow','hair_deflection','vibration','nociception')]
        self.schema = {'version': self.VERSION, 'names': self.names,
            'sites': self.sites, 'joints': self.joints,
            'site_metadata': {m.geom(i).name: {'region': region, 'segment': segment,
                'body': m.body(int(m.geom_bodyid[i])).name,
                'contype': int(m.geom_contype[i]), 'conaffinity': int(m.geom_conaffinity[i])}
                for i,(region,segment) in sensing.geom_regions.items() if region is not None},
            'joint_metadata': {m.joint(j).name: {'axis_local': m.jnt_axis[j].tolist(),
                'range_rad': m.jnt_range[j].tolist(), 'limited': bool(m.jnt_limited[j]),
                'reference_rad': float(sensing.reference[m.jnt_qposadr[j]])}
                for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE},
            'units': 'raw model units, angles rad, time s; absolute physical calibration NOT_ESTABLISHED',
            'position': 'normal-load-weighted centroid in attached body frame; mask false without load',
            'surface_vectors': 'normal and slip in attached body frame, normal points toward fly; slip is fly relative to other body',
            'aggregation': 'load-weighted vector averages may cancel; slip_speed is average magnitude; full contacts retain individual points',
            'self_contact': 'mask false if disabled in model',
            'unmodelled': 'mask false, placeholder zero MUST NOT be interpreted as absent stimulus',
            'routing': 'raw observation only, not anatomical receptive fields or neural current'}

    def observe(self, data):
        obs = self.sensing.observe(data)
        values, valid = [], []
        for kind in ('external','self'):
            for site in self.sites:
                rows=[r for r in obs['contacts'] if r['geom']==site and r['kind']==kind]
                normal=sum(r['normal_force'] for r in rows)
                tangent=sum(r['tangent_force'] for r in rows)
                force=np.sum([r['force_body'] for r in rows],axis=0) if rows else np.zeros(3)
                position=sum((np.array(r['position_body'])*r['normal_force'] for r in rows),np.zeros(3))/normal if normal>0 else np.zeros(3)
                direction=sum((np.array(r['normal_body'])*r['normal_force'] for r in rows),np.zeros(3))/normal if normal>0 else np.zeros(3)
                slip=sum((np.array(r['slip_velocity_body'])*r['normal_force'] for r in rows),np.zeros(3))/normal if normal>0 else np.zeros(3)
                slip_speed=sum(r['slip_speed']*r['normal_force'] for r in rows)/normal if normal>0 else 0.
                values.extend([float(bool(rows)),normal,tangent,*force,*position,*direction,*slip,slip_speed])
                enabled=kind=='external' or obs['self_contact_enabled']
                valid.extend([enabled]*6+[enabled and normal>0]*10)
        byname={j['name']:j for j in obs['joint_states']}
        for name in self.joints:
            values.extend(byname[name][k] for k in self.JOINT_FIELDS);valid.extend([True]*3)
        values.extend([0.]*4);valid.extend([False]*4)
        x=np.asarray(values,float);mask=np.asarray(valid,bool)
        if not np.isfinite(x).all():raise ValueError('nonfinite sensory observation')
        return {'time_s':float(data.time),'values':x,'valid':mask,'schema_version':self.VERSION}
