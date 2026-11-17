"""Source-only terrain baseline metrics; importing this module loads no backend.

NumPy is injected by callers (capture uses the already constructed Core's np).
These are declared engineering diagnostics, never original D1--D8 certification.
"""
import math

LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
VERSION = 'REGISTERED_TERRAIN_NORMAL_TANGENT_BASELINE_V1'
ROW_SHAPES = {
    'time': (), 'com_world_xyz': (3,), 'com_ground_height': (),
    'com_relative_height': (), 'padz': (6,), 'padz_world': (6,),
    'pad_center_world': (6, 3, 3), 'pad_ground_height': (6, 3),
    'loads_world_z_positive': (6,), 'loads_normal': (6,),
    'tangent_speed': (6,), 'legacy_plane_loads': (6,),
    'legacy_plane_xy_speed': (6,),
    'legacy_plane_nonfoot_world_z_positive': (),
    'nonfoot_world_z_positive': (), 'nonfoot_normal': (),
    'registered_contact_count': (), 'legacy_plane_contact_count': (),
    'world_z_force_signed': (6,), 'control_source': (),
}
INT_FIELDS = ('registered_contact_count', 'legacy_plane_contact_count', 'control_source')


def capture(core, source):
    """Copy one post-step measurement; no references to mutable Core arrays escape."""
    np = core._terrain_np
    m = core.measure()
    legacy = m['legacy_plane']
    centers = np.asarray([[core.d.geom_xpos[g].copy() for g in core.pads[leg]]
                          for leg in LEGS], dtype=float).copy()
    ground = np.asarray([[sample['ground_height'] for sample in leg]
                         for leg in m['pad_ground_samples']], dtype=float).copy()
    return dict(
        time=float(core.d.time), com_world_xyz=core.d.subtree_com[core.th].copy(),
        com_ground_height=float(m['com_ground_height']), com_relative_height=float(m['com']),
        padz=m['padz'].copy(), padz_world=m['padz_world'].copy(),
        pad_center_world=centers, pad_ground_height=ground,
        loads_world_z_positive=m['loads'].copy(), loads_normal=m['normal_loads'].copy(),
        tangent_speed=m['tangent_speed'].copy(), legacy_plane_loads=legacy['loads'].copy(),
        legacy_plane_xy_speed=legacy['speed'].copy(),
        legacy_plane_nonfoot_world_z_positive=float(legacy['nonfoot']),
        nonfoot_world_z_positive=float(m['nonfoot']), nonfoot_normal=float(m['nonfoot_normal']),
        registered_contact_count=int(m['terrain_contact_count']),
        legacy_plane_contact_count=int(m['legacy_plane_contact_count']),
        world_z_force_signed=m['world_z_force_signed'].copy(), control_source=int(source))


def analytical_height(xy, terrain_ledger):
    """Highest registered horizontal top at XY; finite footprints include edges.

    Supports the per-case ledger {'terrain': [...]} or the terrain list itself.
    This is a vertical clearance oracle, not a shortest distance to side walls.
    """
    x, y = float(xy[0]), float(xy[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError('Nonfinite terrain query')
    terrain = terrain_ledger['terrain'] if isinstance(terrain_ledger, dict) else terrain_ledger
    tops = []
    for item in terrain:
        footprint = item['footprint']
        top = float(item['top_z'])
        if not math.isfinite(top):
            raise ValueError('Nonfinite terrain top')
        if footprint == 'infinite':
            hit = True
        else:
            if len(footprint) != 4:
                raise ValueError('Expected [minx,maxx,miny,maxy] footprint')
            minx, maxx, miny, maxy = map(float, footprint)
            if not all(map(math.isfinite, (minx, maxx, miny, maxy))) or minx > maxx or miny > maxy:
                raise ValueError('Invalid axis-aligned terrain footprint')
            hit = minx <= x <= maxx and miny <= y <= maxy
        if hit:
            tops.append(top)
    if not tops:
        raise ValueError('No registered analytic terrain hit; clearance UNKNOWN')
    return max(tops)


def _arrays(np, extra_arrays):
    """Require the complete declared 1 ms capture, with no silent short-window use."""
    arrays = {}
    for name, shape in ROW_SHAPES.items():
        value = np.asarray(extra_arrays[name])
        if value.shape != (15400,) + shape:
            raise ValueError('%s must have shape %s' % (name, (15400,) + shape))
        if value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
            raise ValueError('Expected finite numeric array: ' + name)
        if name in INT_FIELDS and not np.equal(value, np.floor(value)).all():
            raise ValueError('Expected integer values: ' + name)
        arrays[name] = value
    times = arrays['time']
    expected = np.arange(1, 15401, dtype=float) * .001
    if not np.all(np.abs(times - expected) <= 1e-8):
        raise ValueError('Expected post-step times 0.001..15.400 at 1 ms intervals')
    for name in ('loads_world_z_positive', 'loads_normal', 'tangent_speed',
                 'legacy_plane_loads', 'legacy_plane_xy_speed',
                 'legacy_plane_nonfoot_world_z_positive', 'nonfoot_world_z_positive',
                 'nonfoot_normal', 'registered_contact_count', 'legacy_plane_contact_count'):
        if np.any(arrays[name] < 0):
            raise ValueError('Negative positive-magnitude measurement: ' + name)
    return arrays


def _five(np, times, com, loads, slip_loads, speed, nonfoot, clearance, W):
    duration = float(times[-1] - times[0])
    delta = com[-1, :2] - com[0, :2]
    xy_speed = float(np.linalg.norm(delta) / duration)
    signed_x = float(delta[0] / duration)
    support = loads.sum(axis=1) / W
    denominator = slip_loads.sum(axis=0)
    weighted_speed = (slip_loads * speed).sum(axis=0) / np.maximum(denominator, 1e-12)
    ratio = weighted_speed / max(xy_speed, 1e-9)
    result = dict(duration_s=duration, sample_count=int(len(times)),
                  foot_support_mean=float(support.mean()), foot_support_min=float(support.min()),
                  nonfoot_force_max_native=float(nonfoot.max()),
                  com_height_min=float(clearance.min()),
                  forward_signed_world_x_native_per_s=signed_x,
                  net_com_xy_speed_native_per_s=xy_speed,
                  net_com_xy_displacement=float(np.linalg.norm(delta)),
                  per_leg_load_weighted_speed=dict(zip(LEGS, map(float, weighted_speed))),
                  per_leg_slip_ratio=dict(zip(LEGS, map(float, ratio))),
                  slip_ratio_max=float(ratio.max()),
                  slip_load_denominator_native=denominator.tolist(),
                  slip_valid_per_leg=(denominator > 1e-12).tolist())
    result['gates'] = dict(support=result['foot_support_mean'] >= .9,
                           nonfoot=result['nonfoot_force_max_native'] <= .2 * W,
                           com=result['com_height_min'] >= 1.2,
                           forward=signed_x >= .05, slip=result['slip_ratio_max'] <= .5)
    result['five_gates_pass'] = bool(all(result['gates'].values()))
    return result


def summarize(np, extra_arrays, native, terrain_ledger, W=24.446354755878005):
    """Exact [5800:15400] post window; coverage is separate from behavior."""
    if not math.isfinite(float(W)) or W <= 0:
        raise ValueError('W must be finite and positive')
    a = _arrays(np, extra_arrays)
    post = slice(5800, 15400)
    warm = slice(1000, 5800)
    t, com = a['time'][post], a['com_world_xyz'][post]
    relative = _five(np, t, com, a['loads_world_z_positive'][post],
                     a['loads_normal'][post], a['tangent_speed'][post],
                     a['nonfoot_normal'][post], a['com_relative_height'][post], W)
    relative['meanings'] = dict(support='registered terrain positive world-z foot loads / W',
        nonfoot='registered terrain compressive normal nonfoot force',
        com='world COM z minus registered terrain ray height at COM XY',
        forward='signed world-x net COM displacement / sampled duration',
        slip='per-leg normal-load-weighted tangent speed / net COM XY speed')
    legacy = _five(np, t, com, a['legacy_plane_loads'][post],
                   a['legacy_plane_loads'][post], a['legacy_plane_xy_speed'][post],
                   a['legacy_plane_nonfoot_world_z_positive'][post], com[:, 2], W)
    legacy['meanings'] = dict(support='plane-only positive world-z foot loads / W',
        nonfoot='plane-only positive world-z nonfoot force', com='world COM z',
        forward='signed world-x net COM displacement / sampled duration',
        slip='per-leg plane-only positive-world-z-load-weighted XY speed / net COM XY speed')
    strict_nf = bool(native['post_nonfoot_normal_peak_native'] == 0
                     and native['post_nonfoot_peak_native'] == 0)
    noforce = bool(native['extra_force_max'] == 0)
    warm_pad_flat = bool(np.all(np.abs(a['pad_ground_height'][warm]) <= 1e-10))
    warm_com_flat = bool(np.all(np.abs(a['com_ground_height'][warm]) <= 1e-10))
    warm_contact_flat = bool(native['warm_nonzero_height_contact_count'] == 0)
    warmflat = bool(warm_pad_flat and warm_com_flat and warm_contact_flat)
    full_native = bool(native['post_native_steps'] == 768000)
    contacts = native['post_native_contacts_by_terrain']
    for name, count in contacts.items():
        if not isinstance(name, str) or isinstance(count, bool) or int(count) != count or count < 0:
            raise ValueError('Invalid native terrain contact count')
    block_count = sum(int(count) for name, count in contacts.items()
                      if name.startswith('terrain_block_'))
    terrain_contact_tested = bool(block_count > 0)
    behavior = bool(relative['five_gates_pass'] and strict_nf and noforce and full_native and warmflat)
    normal_support = a['loads_normal'][post].sum(axis=1) / W
    terrain = terrain_ledger['terrain'] if isinstance(terrain_ledger, dict) else terrain_ledger
    return dict(new_version=VERSION, post_array_slice=[5800, 15400],
        post_time_first=float(t[0]), post_time_last=float(t[-1]),
        weight_native=float(W), relative_five=relative, legacy_plane_five=legacy,
        allterrain_legacy_five_source='separate original sw summary from the same runtime audit; not duplicated here',
        normal_support_diagnostic=dict(mean=float(normal_support.mean()), min=float(normal_support.min()),
                                       gate_applied=False),
        strict_post_nonfoot_normal_zero=strict_nf, no_extra_force=noforce,
        post_native_steps_exact_768000=full_native,
        warmflat=dict(pad_rays_flat=warm_pad_flat, com_rays_flat=warm_com_flat,
                      native_nonzero_height_contacts_zero=warm_contact_flat, covered=warmflat,
                      array_slice=[1000, 5800], force_flatwarm_claim=False),
        coverage=dict(terrain_contact_tested=terrain_contact_tested,
            obstacle_status='CONTACTED' if terrain_contact_tested else 'NOT_TESTED',
            post_terrain_block_contact_count=block_count,
            post_native_contacts_by_terrain=dict(contacts),
            post_pad_ray_nonzero_height_sample_count=int(np.count_nonzero(
                np.abs(a['pad_ground_height'][post]) > 1e-10)),
            ray_over_foot_is_actual_contact=False,
            body_com_x_interval=[float(com[:, 0].min()), float(com[:, 0].max())],
            max_pad_ground_height=float(a['pad_ground_height'][post].max()),
            max_pad_center_world_z=float(a['pad_center_world'][post, :, :, 2].max()),
            max_pad_relative_clearance=float(a['padz'][post].max()),
            registered_contact_count_1ms=int(a['registered_contact_count'][post].sum()),
            legacy_plane_contact_count_1ms=int(a['legacy_plane_contact_count'][post].sum()),
            registered_contact_count_native=int(native['registered_contact_count']),
            legacy_plane_contact_count_native=int(native['legacy_plane_contact_count']),
            registered_terrain_names=[item['name'] for item in terrain]),
        behavior_diagnostic_pass=behavior,
        terrain_contact_tested=terrain_contact_tested,
        rough_terrain_pass=bool(behavior and terrain_contact_tested),
        all_pass=bool(behavior and terrain_contact_tested),
        original_D1_D8='NOT_CLAIMED; changed meanings are declared engineering diagnostics',
        force_units='native; absolute SI NOT_ESTABLISHED')


def audit_observation(np, extra_arrays, decisions, initial_row, *, terrain_ledger=None):
    """Audit all geometry rows and exact actor clearance channels without model calls.

    decisions is a stacked mapping containing X238 (3080,238), or a list of
    decision dicts. The per-case ledger is a keyword or extra_arrays metadata.
    Initial capture must precede the very first 1 ms step at time zero.
    """
    a = _arrays(np, extra_arrays)
    if terrain_ledger is None:
        terrain_ledger = extra_arrays['terrain_ledger']
    if float(initial_row['time']) != 0.:
        raise ValueError('Expected initial pre-first-step capture at time zero')
    initial_padz = np.asarray(initial_row['padz'])
    if initial_padz.shape != (6,) or not np.isfinite(initial_padz).all():
        raise ValueError('Initial padz must be finite shape (6,)')
    x = np.asarray(decisions['X238'] if hasattr(decisions, 'keys')
                   else [row['X238'] for row in decisions])
    if x.shape != (3080, 238) or not np.isfinite(x).all():
        raise ValueError('Expected complete finite decision X238 shape (3080,238)')
    checks = {}
    errors = {}
    analytic_pad = np.asarray([analytical_height(point[:2], terrain_ledger)
                               for point in a['pad_center_world'].reshape(-1, 3)]).reshape(15400, 6, 3)
    analytic_com = np.asarray([analytical_height(point[:2], terrain_ledger)
                               for point in a['com_world_xyz']])
    errors['pad_ray_vs_analytic_max_abs'] = float(np.max(np.abs(a['pad_ground_height'] - analytic_pad)))
    errors['com_ray_vs_analytic_max_abs'] = float(np.max(np.abs(a['com_ground_height'] - analytic_com)))
    errors['com_relative_clearance_max_abs'] = float(np.max(np.abs(
        a['com_relative_height'] - (a['com_world_xyz'][:, 2] - a['com_ground_height']))))
    # The original feet are three unchanged radius-.008 spheres per leg.
    expected_padz = (a['pad_center_world'][:, :, :, 2] - .008 - a['pad_ground_height']).min(axis=2)
    expected_world = (a['pad_center_world'][:, :, :, 2] - .008).min(axis=2)
    errors['pad_relative_clearance_max_abs'] = float(np.max(np.abs(a['padz'] - expected_padz)))
    errors['pad_world_clearance_max_abs'] = float(np.max(np.abs(a['padz_world'] - expected_world)))
    initial_centers = np.asarray(initial_row['pad_center_world'])
    initial_ground = np.asarray(initial_row['pad_ground_height'])
    if initial_centers.shape != (6, 3, 3) or initial_ground.shape != (6, 3):
        raise ValueError('Initial geometry shape mismatch')
    initial_analytic = np.asarray([[analytical_height(point[:2], terrain_ledger)
                                    for point in leg] for leg in initial_centers])
    initial_com = np.asarray(initial_row['com_world_xyz'])
    errors['initial_pad_ray_vs_analytic_max_abs'] = float(np.max(np.abs(initial_ground - initial_analytic)))
    errors['initial_com_ray_vs_analytic_max_abs'] = abs(float(initial_row['com_ground_height'])
        - analytical_height(initial_com[:2], terrain_ledger))
    errors['initial_pad_relative_clearance_max_abs'] = float(np.max(np.abs(initial_padz
        - (initial_centers[:, :, 2] - .008 - initial_ground).min(axis=1))))
    errors['initial_com_relative_clearance_max_abs'] = abs(float(initial_row['com_relative_height'])
        - (float(initial_com[2]) - float(initial_row['com_ground_height'])))
    for name, error in errors.items():
        checks[name.removesuffix('_max_abs')] = bool(math.isfinite(error) and error <= 1e-10)
    height_channels, rate_channels = [], []
    for k in range(3080):
        ms = k * 5
        padz = initial_padz if k == 0 else a['padz'][ms - 1]
        if k == 0:
            rate = np.zeros(6)
        else:
            rate = (padz - a['padz'][ms - 2]) / .001
        height_channels.append(np.clip(padz / .05, -10., 10.))
        rate_channels.append(np.clip(rate / 10., -10., 10.))
    expected_h, expected_v = np.asarray(height_channels), np.asarray(rate_channels)
    checks['actor_pad_height_channels_exact'] = bool(np.array_equal(x[:, 183:189], expected_h))
    checks['actor_pad_rate_channels_exact'] = bool(np.array_equal(x[:, 189:195], expected_v))
    errors['actor_pad_height_channels_max_abs'] = float(np.max(np.abs(x[:, 183:189] - expected_h)))
    errors['actor_pad_rate_channels_max_abs'] = float(np.max(np.abs(x[:, 189:195] - expected_v)))
    return dict(checks=checks, diagnostics=errors, all_checks_passed=bool(all(checks.values())),
                geometry_tolerance=1e-10, actor_channels_require_array_equal=True,
                radius_native=.008, decision_period_ms=5,
                initial_row_required=True, no_Core_or_physical_reintegration=True)
