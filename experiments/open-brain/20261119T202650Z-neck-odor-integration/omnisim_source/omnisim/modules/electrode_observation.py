"""Reduced volume conductor for identifiable axon stimulation and recording."""

from dataclasses import dataclass
from math import ceil, isfinite, pi

import numpy as np

from .conduction_damage import path_damage
from .tissue import TISSUE


@dataclass(frozen=True)
class AxonProbe:
    name: str
    points_m: tuple
    speed_m_s: float = .35
    activation_V: float = 0.004
    action_current_A: float = 1e-9
    owner: int | None = None


class ElectrodeObservation:
    """Finite-size reciprocal point leads in a homogeneous conducting medium."""

    def __init__(self, contacts_m, routes, *, conductivity_S_m=.3, contact_radius_m=1e-6,
                 detection_V=8e-6, sensor_patches=None, activation_spacing_m=2e-6,
                 pickup_separation_m=2e-6, damage_field=None):
        self.contacts = np.asarray(contacts_m, dtype=float)
        self.routes = tuple(routes)
        self.sensor_patches = tuple(range(len(self.contacts))) if sensor_patches is None else tuple(sensor_patches)
        if (self.contacts.ndim != 2 or self.contacts.shape[1] != 3 or not np.isfinite(self.contacts).all()
                or not all(isfinite(v) and v > 0 for v in (conductivity_S_m, contact_radius_m, detection_V,
                                                                                 activation_spacing_m, pickup_separation_m))
                or any(not 0 <= channel < len(self.contacts) for channel in self.sensor_patches)
                or len({route.name for route in self.routes}) != len(self.routes)):
            raise ValueError('invalid electrode geometry, channels or axon names')
        self.conductivity = conductivity_S_m
        self.radius = contact_radius_m
        self.detection = detection_V
        self.activation_spacing_m = activation_spacing_m
        self.pickup_separation_m = pickup_separation_m
        self.damage_field = damage_field
        self._links = {}
        self.events = []

    def _lead(self, points, sites):
        distance2 = np.square(points[:, None] - sites).sum(axis=2)
        return 1 / (4 * pi * self.conductivity * np.sqrt(distance2 + self.radius**2))

    def attach(self, world, name):
        for route in self.routes:
            points = np.asarray(route.points_m, dtype=float)
            if (not route.name or points.ndim != 2 or points.shape[1] != 3 or len(points) < 3
                    or not np.isfinite(points).all() or not all(isfinite(v) and v > 0 for v in (
                        route.speed_m_s, route.activation_V, route.action_current_A))):
                raise ValueError('invalid axon observation route')
            distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
            if np.any(distance <= 0):
                raise ValueError('axon path requires distinct consecutive positions')
            self._links[route.name] = (route, points, np.r_[0.0, np.cumsum(distance)])

    def pulse(self, world, source, sink, current_A, time_s, *, first_axon_material=None):
        """Launch reduced threshold events, then compute propagation and measured peaks at both ends."""
        if (source == sink or not all(0 <= i < len(self.contacts) for i in (source, sink))
                or not isfinite(current_A) or not isfinite(time_s) or time_s < world.time_s):
            raise ValueError('invalid electrode pulse')
        for route, points, arc in self._links.values():
            along = np.linspace(0, arc[-1], max(2, ceil(arc[-1] / self.activation_spacing_m - 1e-12)) + 1)
            positions = self._at(points, arc, along)
            lead = self._lead(positions, self.contacts[[source, sink]])
            drive = current_A * (lead[:, 0] - lead[:, 1])
            curvature = np.diff(drive, n=2) * (self.activation_spacing_m / (along[1] - along[0]))**2
            center = int(np.argmax(curvature)) + 1
            peak = float(curvature[center - 1])
            if peak < route.activation_V:
                self.events.append(dict(axon=route.name, time_s=time_s, kind='below_threshold', activation_V=peak))
                continue
            allowed = None if route.owner is None or first_axon_material is None else first_axon_material + route.owner
            materials = (TISSUE,) if allowed is None else (TISSUE, allowed)
            point = positions[center]
            if (world.material_at(point) not in materials or
                    (self.damage_field is not None and world.sample(self.damage_field, point) >= 1)):
                self.events.append(dict(axon=route.name, time_s=time_s, kind='activation_at_failed_voxel',
                                        activation_V=peak))
                continue
            self.events.append(dict(axon=route.name, time_s=time_s, kind='axon_fired', node=center, activation_V=peak))
            path_left = np.vstack((point, points[arc < along[center]][::-1]))
            path_right = np.vstack((point, points[arc > along[center]]))
            for end, path in (('a', path_left), ('b', path_right)):
                self._propagate(world, route, path, points, arc, end, time_s, materials)
        self.events.sort(key=lambda e: e['time_s'])

    @staticmethod
    def _at(points, arc, distances):
        return np.column_stack([np.interp(distances, arc, points[:, axis]) for axis in range(3)])

    def observe_spike(self, world, neuron, time_s, first_axon_material):
        for route, points, arc in self._links.values():
            if route.owner == neuron:
                self.events.append(dict(axon=route.name, time_s=time_s, kind='axon_fired', node=0))
                self._propagate(world, route, points, points, arc, 'b', time_s, (TISSUE, first_axon_material + neuron))

    def _propagate(self, world, route, path, points, arc, end, time_s, materials):
        effect = path_damage(world, path, self.damage_field, allowed_materials=materials)
        arrival = time_s + effect.delay_s(route.speed_m_s)
        if effect.block_at_m is not None:
            self.events.append(dict(axon=route.name, end=end, time_s=arrival, kind='conduction_block',
                                    path_to_block_m=effect.block_at_m))
            return
        separation = min(self.pickup_separation_m, arc[-1])
        s = (0, separation) if end == 'a' else (arc[-1], arc[-1] - separation)
        lead = self._lead(self._at(points, arc, s), self.contacts[list(self.sensor_patches)])
        peak_V = route.action_current_A * effect.gain * (lead[0] - lead[1])
        self.events.append(dict(axon=route.name, end=end, time_s=arrival, kind='axon_arrived', gain=effect.gain))
        for patch, amplitude in zip(self.sensor_patches, peak_V):
            self.events.append(dict(axon=route.name, end=end, time_s=arrival, patch=patch, peak_V=float(amplitude),
                                    kind='recorded' if abs(amplitude) >= self.detection else 'missed'))

    def step(self, world, name, dt_s):
        end_s = world.time_s + dt_s
        existing, future = [], []
        for event in self.events:
            (existing if event['time_s'] <= end_s else future).append(event)
        self.events = future
        for event in existing:
            world.emit(name, event['kind'], time_s=event['time_s'], **{k: v for k, v in event.items()
                                                                       if k not in ('kind', 'time_s')})
