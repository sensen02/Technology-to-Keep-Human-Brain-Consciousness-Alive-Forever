"""Canonical whole-body contact observations and subtype leg afferents.

Engineering transduction, NOT calibrated physiology. All external contacts are
observed; unsupported nonleg receptive fields stay UNMAPPED. Air/hair mechanics
are explicitly unmodelled, never replaced by a hidden constant current.
"""
from dataclasses import asdict
import numpy as np
from .sensory_channels import (ChannelSensoryEncoder, ChannelConfig,
                               ChannelQuantities, channel_index_map, LEGS)


def canonical_region(name):
    for leg in LEGS:
        if name.upper().startswith(leg.upper()):
            return leg, name[2:].removesuffix('_geom')
    if name.startswith(('Head', 'LEye', 'REye', 'LPedicel', 'RPedicel',
                        'LFuniculus', 'RFuniculus', 'LArista', 'RArista',
                        'Rostrum', 'Haustellum')):
        return 'head', name.removesuffix('_geom')
    if name.startswith('Thorax'):
        return 'thorax', name.removesuffix('_geom')
    if name.startswith(('A1A2', 'A3', 'A4', 'A5', 'A6')):
        return 'abdomen', name.removesuffix('_geom')
    if 'Wing' in name:
        return 'wing', name.removesuffix('_geom')
    if 'Haltere' in name:
        return 'haltere', name.removesuffix('_geom')
    return None, name


class WholeBodySensing:
    def __init__(self, model, built, reference_qpos, *, gain=.3):
        import mujoco
        self.m = model
        self.reference = np.asarray(reference_qpos, float).copy()
        if self.reference.shape != (model.nq,) or not np.isfinite(self.reference).all():
            raise ValueError('invalid reference qpos')
        if not np.isfinite(gain) or gain < 0:
            raise ValueError('gain must be finite and nonnegative')
        self.geom_regions = {i: canonical_region(model.geom(i).name or '')
                             for i in range(model.ngeom)}
        self.groups, self.coverage = channel_index_map(built['n'], built['tier']['root_ids'])
        self.encoder = ChannelSensoryEncoder(self.groups, built['n'], ChannelConfig(current_nA_per_unit=gain))
        self.gain = float(gain)
        self.regional_groups = {'thorax': [], 'abdomen': []}
        # Exact body-part + tactile/bristle annotation, not broad nerve substring.
        annotations = built.get('data', {})
        for li, gi in enumerate(built.get('global_index', [])):
            def label(key):
                return str(annotations[key][gi]) if key in annotations else ''
            region = label('ann_body_part')
            if (region in self.regional_groups and 'tactile' in label('ann_function')
                    and label('ann_class') == 'bristle_neuron'
                    and label('ann_flow') == 'afferent'):
                self.regional_groups[region].append(li)
        edges = built.get('edges', {})
        self.regional_outgoing = {region: [int(sum(edges[k][:, i].nnz for k in ('We', 'Wi')))
            for i in indices] for region, indices in self.regional_groups.items()}
        self.joints = {}
        for leg in LEGS:
            ids = [j for j in range(model.njnt)
                   if (model.joint(j).name or '').startswith('joint_'+leg.upper())]
            tib = model.joint('joint_'+leg.upper()+'Tibia_pitch').id
            self.joints[leg] = (ids, tib)
        self.weight_per_leg = float(model.body_mass.sum()*abs(model.opt.gravity[2])/6)
        if self.weight_per_leg <= 0:
            raise ValueError('positive model weight required')

    def observe(self, data):
        import mujoco
        records = []
        regions = {}
        for i, (region, _) in self.geom_regions.items():
            if region is not None:
                regions.setdefault(region, {'normal_force': 0., 'tangent_force': 0.,
                                             'contacts': 0, 'foot_contacts': 0})
        force = np.zeros(6)
        for ci, c in enumerate(data.contact):
            if c.efc_address < 0:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            r1, s1 = self.geom_regions[g1]; r2, s2 = self.geom_regions[g2]
            if r1 is None and r2 is None:
                continue
            mujoco.mj_contactForce(self.m, data, ci, force)
            category = 'self' if r1 is not None and r2 is not None else 'external'
            frame = c.frame.reshape(3, 3)
            world_force_on_g2 = frame.T @ force[:3]
            point_velocities = []
            for gid in (g1, g2):
                body = int(self.m.geom_bodyid[gid])
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(self.m, data, mujoco.mjtObj.mjOBJ_BODY,
                                        body, velocity, 0)
                point_velocities.append(velocity[3:] + np.cross(velocity[:3], c.pos-data.xipos[body]))
            relative_velocity = point_velocities[1]-point_velocities[0]
            tangent_velocity = relative_velocity-frame[0]*np.dot(frame[0], relative_velocity)
            for g, region, segment, sign, other in [(g1,r1,s1,-1,g2),(g2,r2,s2,1,g1)]:
                if region is None:
                    continue
                bid = int(self.m.geom_bodyid[g]); rotation = data.xmat[bid].reshape(3,3)
                row = {'region':region,'segment':segment,'geom':self.m.geom(g).name,
                       'other_geom':self.m.geom(other).name,'kind':category,
                       'position_body':(rotation.T @ (c.pos-data.xpos[bid])).tolist(),
                       'force_body':(rotation.T @ (sign*world_force_on_g2)).tolist(),
                       'normal_force':float(max(0.,force[0])),
                       'tangent_force':float(np.linalg.norm(force[1:3])),
                       'normal_body':(rotation.T @ (sign*frame[0])).tolist(),
                       'slip_velocity_body':(rotation.T @ (sign*tangent_velocity)).tolist(),
                       'slip_speed':float(np.linalg.norm(tangent_velocity))}
                records.append(row)
                # Self contacts are logged separately, not double-counted into external load.
                if category == 'external':
                    v=regions[region];v['normal_force']+=row['normal_force'];v['tangent_force']+=row['tangent_force'];v['contacts']+=1
                    if segment.startswith('Tarsus'):v['foot_contacts']+=1
        proprio = {}
        for leg,(ids,tib) in self.joints.items():
            qa=self.m.jnt_qposadr[tib];va=self.m.jnt_dofadr[tib]
            delta=[float(data.qpos[self.m.jnt_qposadr[j]]-self.reference[self.m.jnt_qposadr[j]]) for j in ids]
            v=regions[leg]
            proprio[leg]=ChannelQuantities(
                tibia_joint_dev_rad=float(data.qpos[qa]-self.reference[qa]),
                tibia_joint_vel_rad_s=float(data.qvel[va]), multi_joint_dev_rad=float(np.mean(delta)),
                contact_force_norm=v['normal_force']/self.weight_per_leg,
                # Coarse leg bristle routing includes all leg segments, not only tarsal floor.
                foot_contact=float(v['contacts']>0))
        joint_states = []
        for j in range(self.m.njnt):
            # Scalar internal hinges only; free-root world pose is not a skin sensor.
            if self.m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            qa, va = int(self.m.jnt_qposadr[j]), int(self.m.jnt_dofadr[j])
            joint_states.append({'name': self.m.joint(j).name,
                'position_rad': float(data.qpos[qa]),
                'reference_deviation_rad': float(data.qpos[qa]-self.reference[qa]),
                'velocity_rad_s': float(data.qvel[va]),
                'range_rad': self.m.jnt_range[j].tolist()})
        return {'contacts':records,'regions':regions,'proprio':proprio,
                'joint_states':joint_states,
                'self_contact_enabled': bool(any(self.m.geom_contype[i] & self.m.geom_conaffinity[j]
                    for i in self.geom_regions for j in self.geom_regions
                    if i != j and self.geom_regions[i][0] is not None and self.geom_regions[j][0] is not None)),

                'air_hair':{'status':'UNMODELLED_NO_CALIBRATED_RECEPTIVE_FIELD','neural_current':0.},
                'units':'model forces; physical calibration NOT_ESTABLISHED'}

    def encode(self, data):
        obs=self.observe(data)
        current,record=self.encoder.encode(obs['proprio'])
        record['whole_body']={k:v for k,v in obs.items() if k!='proprio'}
        record['proprio_quantities']={k:asdict(v) for k,v in obs['proprio'].items()}
        record['regional_channels'] = {}
        for region, indices in self.regional_groups.items():
            value = float(obs['regions'].get(region, {}).get('contacts', 0) > 0)
            current[indices] += self.gain * value
            record['regional_channels'][region] = {'value': value, 'n_cells': len(indices),
                'local_indices': indices, 'modeled_outgoing_edges': self.regional_outgoing[region],
                'propagation_established': False,
                'mapping': 'exact body_part+tactile+bristle; coarse region broadcast; no transmitter overlay'}
        record['unmapped_contact_regions']=[k for k,v in obs['regions'].items()
            if k not in LEGS and not self.regional_groups.get(k) and v['contacts']]
        record['assumptions']=['leg bristles receive coarse any-segment external contact, not per-cell skin fields',
                               'contact normal load is a strain proxy, not measured cuticle strain',
                               'signed joint deviation/velocity use existing declared channel convention',
                               'thorax/abdomen use exact annotation candidates, not precise receptive fields',
                               'unknown regions have no invented neural routing; air/hair unmodelled']
        return current,record
