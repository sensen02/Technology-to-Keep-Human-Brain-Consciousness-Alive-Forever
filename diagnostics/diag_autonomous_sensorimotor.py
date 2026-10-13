#!/usr/bin/env python3
"""Strict fail-closed audit; optional unvalidated sensory -> network -> muscle experiment.

No target, tonic current, background current, CPG, external drive permission or
body-state-to-action controller. Passive physical movement is allowed. Exploratory
receptor and name-Jaccard motor translations are NOT validated anatomy or autonomy.
Contact presence only: engine/embodied/force_channel.py forbids physiological
interpretation of the project's contact force magnitudes.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import os
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MUJOCO_GL', 'egl')
OUT = ROOT / 'outputs/autonomous_sensorimotor_round1'
XML = ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def motor_rate(spikes, indices, window_s):
    """Per-neuron Hz, not summed-population Hz. Missing groups produce zero."""
    if not len(indices):
        return 0.0
    return float(spikes[:, indices].sum()) / (len(indices) * window_s)


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exploratory', action='store_true')
    ap.add_argument('--sensory-mode', choices=('fecc-classes','front-club-motion','legacy-broad'),
                    default='fecc-classes',
                    help='fecc-classes (default) drives ALL three identified front-leg chordotonal '
                         'subclasses with their documented CLASS-LEVEL property: claw = joint POSITION '
                         'deviation, hook and club = joint motion magnitude. front-club-motion is the '
                         'earlier club-only variant kept for comparison. All are uncalibrated class-level '
                         'proxies, not measured receptor currents, and none of them carries a target.')
    ap.add_argument('--design-overlay', type=Path, default=None,
                    help='Optional DESIGN motor-pool overlay from tools_build_design_runtime.py. It '
                         'REPLACES each actuator pool with the pool the design assigns, and applies '
                         'its per-actuator command scale (coupled fibre-cluster pairs are driven from '
                         'one pool at half strength each). Every row carries a support level, and rows '
                         'marked DESIGN_HYPOTHESIS / MODEL_GEOMETRY_BUNDLE_COUPLING are declared '
                         'wiring decisions rather than connectome-confirmed pools.')
    ap.add_argument('--motor-overlay', type=Path, default=None,
                    help='Optional BANC v888 motor overlay from tools_build_v888_motor_overlay.py. Its '
                         'EXACT_MUSCLE_NAME part is merged into the actuator mapping. The '
                         'PAIR_LEVEL_CLUSTER_ONLY part is merged only with --include-pair-level, because '
                         'driving both a/b clusters with one pool double counts muscle capacity.')
    ap.add_argument('--include-pair-level', action='store_true',
                    help='Also enable the pair-level part of the overlay (explicit opt-in, caveat noted in '
                         'the report).')
    ap.add_argument('--motor-map', choices=('anatomy-candidate','exact-candidate','legacy-fuzzy'), default='anatomy-candidate',
                    help='Exact annotation candidates remain unvalidated; unmatched outputs stay zero.')
    ap.add_argument('--seconds', type=float, default=.3)
    ap.add_argument('--interventions', action='store_true',
                    help='Add neural interventions: silence the mapped motor neurons IN the network '
                         '(zero their inbound and outbound synapses, the standard silencing '
                         'convention) plus a size-matched random-neuron control. Interventions only; '
                         'nothing here drives the body.')
    ap.add_argument('--stance-file', type=Path, default=None,
                    help='Optional JSON from diagnostics/diag_stance_search_with_adhesion.py (or any '
                         'JSON with a joints_rad map). Its per-leg joint angles are written into qpos '
                         'and the root is seated from the recorded root height, so the run starts from a '
                         'SEARCHED, tarsus-only stance instead of the XML keyframe. Only joint angles '
                         'and root height are used; no controller state is imported.')
    ap.add_argument('--sensory-battery', choices=('fecc','leg-full'), default='fecc',
                    help='fecc = front-leg chordotonal only (previous behaviour). leg-full = all six '
                         'legs with joint position (claw / hair plate), joint motion (hook / club), '
                         'leg load (campaniform) and tarsus touch (tactile bristle). Every channel is a '
                         'documented CLASS-LEVEL property driven by a body-available variable with an '
                         'uncalibrated normalisation, and none of them carries a target.')
    ap.add_argument('--record-contacts', action='store_true',
                    help='Record the ground contact set each interval (which parts of the body are '
                         'bearing on the floor, and whether the contact set is tarsus-only). This is '
                         'bookkeeping only and does not drive anything.')
    ap.add_argument('--sensory-cut-after', type=float, default=None,
                    help='Seconds after which the body-derived sensory current is cut to zero for the '
                         'whole run (sensory deafferentation: the fly loses its leg sense). An '
                         'intervention on the animal, not a control input.')
    ap.add_argument('--spike-storage', choices=('full','counts'), default='full',
                    help='full keeps per-interval spike index lists (default, comparable with earlier '
                         'rounds); counts keeps only counts, for longer runs.')
    ap.add_argument('--split-propagation-cuts', action='store_true',
                    help='Add isolated We-only and Wi-only cuts; not normal behavior settings.')
    ap.add_argument('--voltage-method', choices=('backward_euler','frozen_exact'), default='backward_euler')
    ap.add_argument('--neural-dt-ms', type=float, choices=(.5,.25,.1), default=.5,
                    help='Numerical convergence probe; topology and physical clocks unchanged.')
    ap.add_argument('--sensory-max-na', type=float, default=.05)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--adhesion-const', type=float, default=0.0,
                    help='Constant command held on any non-muscle ADHESION actuators present in a '
                         'variant model. It is a declared contact property of the body, held constant '
                         'for the whole run, NOT driven by the network, so it cannot steer or decide '
                         'anything. Must be 0.0 for the canonical model, which has no adhesion.')
    ap.add_argument('--xml', type=Path, default=None,
                    help='Body model to use. Default is the canonical asset; any other value is a '
                         'VARIANT and is recorded as such. The canonical hash is never modified.')
    args = ap.parse_args()
    OUT = args.out_dir.resolve()
    global XML
    canonical_xml = XML
    if args.xml is not None:
        XML = args.xml.resolve()
        if not XML.exists():
            ap.error(f'XML not found: {XML}')
        if XML != canonical_xml:
            print(f'VARIANT MODEL IN USE: {XML}  (canonical asset untouched: {canonical_xml})',
                  flush=True)
    if not 0 < args.seconds <= 3.0 or not np.isfinite(args.sensory_max_na) or not 0 <= args.sensory_max_na <= .1:
        ap.error('bounded duration (0,3.0] and sensory current [0,.1] nA required')
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), XML, ROOT/'muscle_wiring.py', ROOT/'tools_mn_muscle_map.py',
               ROOT/'outputs/mn_muscle_map_per_leg.json', ROOT/'engine/embodied/adapters.py', ROOT/'engine/neural_cond.py',
               ROOT/'data/flywire/banc_connectome.npz', ROOT/'engine/embodied/neural_command_audit.py', ROOT/'tools_audit_autonomous_mapping.py', ROOT/'engine/embodied/feco_identity.py', ROOT/'data/flywire/banc_neurons.csv.gz']
    report = {'overall_status': 'PARTIAL_TRANSLATION_NOT_VALIDATED_AUTONOMY',
              'body_model': str(XML),
              'canonical_body_model': str(canonical_xml),
              'is_variant_model': bool(XML != canonical_xml),
              'mode': 'exploratory' if args.exploratory else 'strict',
              'validation': {'receptor_mapping_validated': False, 'actuator_mapping_validated': False},
              'source_sha256': {str(p): digest(p) for p in sources},
              'uncertainties': [
                  'Leg nerve mechanosensory membership does not validate receptor modality, joint specificity, gain or sign.',
                  f'Actuator/MN policy {args.motor_map}; lexical annotation candidates are NOT validated anatomical registration. Legacy fuzzy mode is explicit opt-in only.',
                  'Bounded subnetwork excludes some motor neurons; no activity fabricated for missing MNs.',
                  'Conductance parameters, current gain and 100 Hz activation normalization are declared, not biological calibration.',
                  'No visual input implemented; body angle/velocity and binary foot contact are simulator proprio/contact quantities.',
                  'Controls zero does not remove passive muscle forces or gravity; observed passive motion is permitted.'],
              'forbidden_paths': {'target': False, 'CPG': False, 'tonic': False, 'external_forward_permission': False},
              'what_this_does_not_establish': 'Validated autonomous fly, standing, walking, biological behavior, or neural lesion effects.'}
    if not args.exploratory:
        report['execution_status'] = 'REFUSED_MISSING_VALIDATED_RECEPTOR_AND_ACTUATOR_MAPPING'
        (OUT/'strict.json').write_text(json.dumps(report, indent=2))
        print(report['execution_status'])
        return 2

    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    from muscle_wiring import build_mn_muscle_wiring
    from tools_mn_muscle_map import tokens, canon
    from engine.embodied.adapters import NeuralTier, TierConfig, OwnershipLedger, LEG_NERVE

    wiring, stats, built = build_mn_muscle_wiring(verbose=False)  # EXACTLY ONCE; reuse built graph.
    cfg = TierConfig(background_mean_nA=0., background_sd_nA=0., dt_ms=args.neural_dt_ms, voltage_method=args.voltage_method)
    report.update(tier_config=asdict(cfg), tier_fingerprint=built['fingerprint'],
                  wiring_stats=stats, build_mn_muscle_wiring_calls=1, mujoco_version=mj.__version__)
    model = _load_mjcf(str(XML)).compile()
    if model.neq:
        raise ValueError('unlocked model required: any equality causes refusal')
    if int(np.sum(model.jnt_type == int(mj.mjtJoint.mjJNT_FREE))) != 1:
        raise ValueError('expected exactly one free joint')
    whitelist = []
    mapping = []
    # Non-muscle actuators are admitted ONLY when they are adhesion actuators of a variant
    # model.  They are held at a declared constant for the whole run and are excluded from
    # the neural command audit, because the audit is about MUSCLE command origin.
    adhesion_actuators = []
    for a in range(model.nu):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, a) or ''
        is_muscle = (name[:2] in LEGS and model.actuator_dyntype[a] == int(mj.mjtDyn.mjDYN_MUSCLE)
                     and model.actuator_gaintype[a] == int(mj.mjtGain.mjGAIN_MUSCLE)
                     and model.actuator_trntype[a] == int(mj.mjtTrn.mjTRN_TENDON)
                     and model.actuator_actnum[a] == 1)
        if not is_muscle:
            if 'adhesion' not in name.lower():
                raise ValueError('muscle-only whitelist rejected actuator '+name)
            if args.adhesion_const <= 0.:
                raise ValueError('variant adhesion actuator present but --adhesion-const is 0: '
                                 'refusing to run a variant whose adhesion is silently inactive')
            adhesion_actuators.append(a)
            continue
        whitelist.append(a)
        nm_tokens = {canon(t) for t in tokens(name)}
        candidates = []
        for typ, group in wiring.get(name[:2], {}).items():
            tt = {canon(t) for t in tokens(typ)}
            score = len(nm_tokens & tt)/len(nm_tokens | tt) if nm_tokens | tt else 0.
            candidates.append((score, typ, group))
        candidates.sort(key=lambda c: (-c[0], c[1]))
        score, typ, group = candidates[0] if candidates else (0., None, {})
        accepted = score >= .4
        mapping.append({'actuator': a, 'name': name, 'mn_type': typ if accepted else None,
                        'name_jaccard': score, 'all_top_ties': [c[1] for c in candidates if c[0] == score],
                        'local': sorted(set(group.get('local', []))) if accepted else [],
                        'root_ids': [str(x) for x in group.get('root_ids', [])] if accepted else [],
                        'missing_MNs': [str(x) for x in group.get('missing', [])],
                        'translation_validated': False})
    if len(whitelist) != 90 or model.na != 90:
        raise ValueError('expected 90 one-state muscles')
    if args.motor_map == 'exact-candidate':
        from tools_audit_autonomous_mapping import model_name, resolve, NERVES
        source=built['data']
        for row in mapping:
            leg,key=model_name(row['name'])
            mask=(source['ann_nerve']==NERVES[leg]) & (source['ann_super_class']=='motor') & (source['ann_class']=='leg_motor_neuron')
            match=resolve(key,source['ann_primary_type'][mask])
            glob=np.flatnonzero(mask & (source['ann_primary_type']==match.get('label','__UNMATCHED__')))
            loc=built['local_of_global'][glob]
            row.update(mn_type=match.get('label'),local=loc[loc>=0].astype(int).tolist(),
                root_ids=[str(x) for x in source['root_ids'][glob[loc>=0]]],
                missing_MNs=[str(x) for x in source['root_ids'][glob[loc<0]]],
                resolution=match,legacy_name_jaccard_not_used=row.pop('name_jaccard'))
        # A normalized annotation shared by distinct actuator compartments is not unique.
        from collections import Counter
        labels=Counter((r['name'][:2],r['mn_type']) for r in mapping if r['mn_type'])
        for row in mapping:
            if row['mn_type'] and labels[(row['name'][:2],row['mn_type'])]>1:
                row.update(local=[],root_ids=[],resolution={'status':'AMBIGUOUS_SHARED_ACTUATOR_LABEL'})
    if args.motor_map=='anatomy-candidate':
        manifest_path=ROOT/'outputs/muscle_anatomy_resolution/resolution.json'
        manifest=json.loads(manifest_path.read_text())
        for path,expected_hash in manifest['source_sha256'].items():
            if digest(path)!=expected_hash:raise ValueError('Anatomy evidence source changed: '+path)
        by_name={r['actuator']:r for r in manifest['actuators']}
        if len(manifest['actuators'])!=len(by_name) or set(by_name)!={r['name'] for r in mapping}:
            raise ValueError('Anatomy manifest duplicate actuator or coverage mismatch')
        root_to_local={str(int(r)):i for i,r in enumerate(built['tier']['root_ids'])}
        for row in mapping:
            src=by_name[row['name']]
            candidates=src['candidates'] if src['status']=='CANDIDATE_SUPPORTED_BY_LABEL_NERVE_ROUTE' else []
            roots=[r['root_id'] for r in candidates]
            if len(set(roots))!=len(roots):raise ValueError('Duplicate root in anatomy candidate group')
            pairs=sorted((root_to_local[r],r) for r in roots if r in root_to_local)
            row.update(mn_type=src['target_primary'] if candidates else None,
                local=[li for li,r in pairs],
                root_ids=[r for li,r in pairs],
                missing_MNs=[r for r in roots if r not in root_to_local],
                resolution={'status':src['status'],'rule':src['normalization_rule']},
                legacy_name_jaccard_not_used=row.pop('name_jaccard'))
        report['source_sha256'][str(manifest_path)]=digest(manifest_path)
        report['anatomy_manifest_caveat']='Experimental candidates only; runtime_authorization false for biologically validated mode. Strict mode remains refused.'
    report['motor_mapping_policy']=args.motor_map
    if args.design_overlay is not None:
        # A DESIGN overlay is an explicit, labelled wiring hypothesis: it replaces each
        # actuator's pool with the pool the design assigns it, and it may scale the command of a
        # coupled fibre-cluster pair down so that one pool driving two actuators does not double
        # the muscle's force.  Nothing here is an anatomical validation; every row carries its
        # support level in the report.
        dov=json.loads(Path(args.design_overlay).read_text())
        by_name={r['name']:r for r in mapping}
        scales=np.ones(len(mapping))
        applied={'replaced':0,'scaled':0,'unknown_actuators':[],'held_zero':[]}
        for row in mapping:
            row['local']=[]; row['root_ids']=[]; row['missing_MNs']=[]
        row_index={id(r):i for i,r in enumerate(mapping)}
        for act,e in dov['entries'].items():
            row=by_name.get(act)
            if row is None:
                applied['unknown_actuators'].append(act); continue
            ri=row_index[id(row)]
            row['local']=sorted(set(int(x) for x in e['local_indices']))
            row['root_ids']=sorted(set(e['pool_root_ids']))
            row['missing_MNs']=[]
            row['design_support']=e['support']
            row['design_pool_source']=e.get('pool_source')
            row['design_note']=e.get('pool_note')
            scales[ri]=float(e.get('command_scale',1.0))
            applied['replaced']+=1
            if scales[ri]!=1.0:
                applied['scaled']+=1
            if not row['local']:
                applied['held_zero'].append(act)
        report['design_overlay']={'path':str(Path(args.design_overlay).resolve()),
            'sha256':digest(args.design_overlay),'applied':applied,
            'status':dov.get('status'),
            'caveat':'DESIGN HYPOTHESIS: rows marked DESIGN_HYPOTHESIS or MODEL_GEOMETRY_BUNDLE_'
                     'COUPLING are declared wiring decisions, not connectome-confirmed pools. '
                     'Scaled rows are coupled fibre-cluster pairs driven from one pool at half '
                     'strength each so the muscle is not given double force.'}
        report['mapped_actuator_count']=sum(bool(r['local']) for r in mapping)
        report['command_scale_by_mapping_order']=scales.tolist()
    if args.motor_overlay is not None:
        ov=json.loads(Path(args.motor_overlay).read_text())
        by_name={r['name']:r for r in mapping}
        merged_exact=merged_pair=0
        for act,locals_ in ov['exact_muscle_name'].items():
            row=by_name.get(act)
            if row is None:continue
            row['local']=sorted(set(row['local'])|set(int(x) for x in locals_));merged_exact+=1
            row['v888_overlay']='EXACT_MUSCLE_NAME'
        if args.include_pair_level:
            for act,locals_ in ov['pair_level_cluster_only'].items():
                row=by_name.get(act)
                if row is None:continue
                row['local']=sorted(set(row['local'])|set(int(x) for x in locals_));merged_pair+=1
                row['v888_overlay']='PAIR_LEVEL_CLUSTER_ONLY'
        report['motor_overlay']={'path':str(Path(args.motor_overlay).resolve()),
            'exact_merged':merged_exact,'pair_level_merged':merged_pair,
            'include_pair_level':bool(args.include_pair_level),
            'caveat':'pair-level actuators share one documented pool across the model a/b clusters, so '
                     'their combined drive double counts muscle capacity; exact-only is the defensible set'}
    report['mapped_actuator_count']=sum(bool(r['local']) for r in mapping)
    old_ranges = model.actuator_ctrlrange.copy()
    model.actuator_ctrlrange[:, 0] = 0  # memory ONLY: allow genuine zero command
    adr = model.actuator_actadr.astype(int)
    # Variant models may carry adhesion actuators with no activation state, so the
    # coverage check is applied to the MUSCLE channels only and the adhesion channels
    # are asserted to have no activation state at all.
    if sorted(adr[whitelist].tolist()) != list(range(model.na)):
        raise ValueError('activation address coverage mismatch over muscle actuators')
    if adhesion_actuators and np.any(adr[adhesion_actuators] != -1):
        raise ValueError('adhesion actuators must have no activation state')
    n_muscles = len(whitelist)
    muscle_ctrl_index = np.zeros(model.nu, bool)
    muscle_ctrl_index[whitelist] = True
    report['actuator_classification'] = {
        'n_muscle_actuators': int(n_muscles),
        'n_adhesion_actuators': int(len(adhesion_actuators)),
        'adhesion_actuator_names': [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, a) or ''
                                    for a in adhesion_actuators],
        'adhesion_const_declared': float(args.adhesion_const),
        'adhesion_is_a_constant_contact_property':
            'held constant for the whole run, never driven by the network, audited below',
        'scope': 'only the muscle channels are the neural translation; adhesion cannot steer or decide'}
    report['actuator_mapping'] = mapping
    report['wiring_complete_listing'] = wiring
    report['in_memory_model_changes'] = {'ctrl_lower_before': old_ranges[:,0].tolist(), 'ctrl_lower_after': model.actuator_ctrlrange[:,0].tolist(), 'canonical_written': False}
    sensory = {leg: np.asarray(built['sensory'][leg.lower()], int) for leg in LEGS}
    sensory_groups = {}
    if args.sensory_mode in ('front-club-motion','fecc-classes'):
        from engine.embodied.feco_identity import resolve_feco
        feco=resolve_feco(ROOT/'data/flywire/banc_neurons.csv.gz',built)
        report['feco_identity']=feco
        def locals_of(subtype, leg):
            return np.array([r['local_index'] for r in feco[subtype]
                             if r['local_index'] is not None and r['nerve']==LEG_NERVE[leg.lower()]],int)
        if args.sensory_mode == 'front-club-motion':
            sensory_groups={leg:{'club':locals_of('club',leg)} if leg in ('LF','RF') else {}
                            for leg in LEGS}
        else:
            # All three documented subclasses.  claw cells are documented as POSITION-tuned and each
            # single cell prefers flexion OR extension, and that per-root preference is NOT available
            # locally, so claw is driven by the SIGN-FREE position deviation from the joint's range
            # midpoint - the same class-level, uncalibrated standard already used for motion.  hook
            # and club are documented motion-responsive, so they get the motion magnitude.
            sensory_groups={leg:{s:locals_of(s,leg) for s in ('claw','hook','club')}
                            if leg in ('LF','RF') else {} for leg in LEGS}
        sensory={leg:np.unique(np.concatenate([v for v in sensory_groups[leg].values()])
                               if sensory_groups[leg] else np.array([],int)) for leg in LEGS}
        report['sensory_subtype_groups']={leg:{s:v.tolist() for s,v in sensory_groups[leg].items()}
                                          for leg in LEGS}
        report['sensory_subtype_counts']={leg:{s:int(v.size) for s,v in sensory_groups[leg].items()}
                                          for leg in LEGS}
        total_sensory=int(sum(len(v) for v in sensory.values()))
        if args.sensory_mode == 'front-club-motion':
            assert total_sensory==96
        report['sensory_cell_count']=total_sensory
    report['sensory_mode']=args.sensory_mode
    leg_channels=None
    if args.sensory_battery=='leg-full':
        ch_path=ROOT/'outputs/leg_sensory_channels/leg_sensory_channels.json'
        if not ch_path.exists():
            raise ValueError('leg-full battery needs tools_resolve_leg_sensory_channels.py output')
        leg_channels=json.loads(ch_path.read_text())
        report['leg_sensory_battery']={
            'source':leg_channels['source'],'counts':leg_channels['counts'],
            'modality_variable':leg_channels['modality_variable'],
            'limitations':leg_channels['limitations'],
            'legs_driven':{leg:{m:len(v) for m,v in leg_channels['channels'][leg].items()}
                           for leg in leg_channels['legs']}}
        # channels actually injected: the three chordotonal sub-classes plus hair plate,
        # campaniform and tactile.  The generic 'chordotonal' entry is the union of the
        # sub-classes, so it is not injected to avoid driving the same cell twice.
        drive_modalities=('chordotonal_claw','chordotonal_hook','chordotonal_club',
                          'hair_plate','campaniform','tactile_bristle')
        sensory={leg:np.unique(np.concatenate([np.asarray(leg_channels['channels'][leg][m],int)
            for m in drive_modalities if leg_channels['channels'][leg].get(m,[])]) ) if any(
                leg_channels['channels'][leg].get(m,[]) for m in drive_modalities) else np.array([],int)
            for leg in LEGS}
        report['sensory_cell_count']=int(sum(len(v) for v in sensory.values()))
    allowed_sensory = np.unique(np.concatenate(list(sensory.values())))
    mapped_mn = np.unique([i for row in mapping for i in row['local']]).astype(int)
    assert not np.intersect1d(allowed_sensory, mapped_mn).size, 'sensory injection must not directly target mapped motor neurons'
    if 'ann_super_class' in built['data']:
        mn_classes = np.asarray(built['data']['ann_super_class'])[built['global_index'][mapped_mn]]
        assert np.all(mn_classes == 'motor'), 'mapped MN has non-motor superclass'
        report['mn_class_check'] = {'status':'ANNOTATED_MOTOR_CONFIRMED','count':int(mapped_mn.size)}
    else:
        report['mn_class_check'] = {'status':'UNKNOWN_NO_SUPERCLASS_FIELD','count':int(mapped_mn.size)}
    report['sensory_motor_overlap_count'] = 0
    nonsensory = np.ones(built['n'], bool); nonsensory[allowed_sensory] = False
    report['sensory_mapping'] = {leg: {'nerve': LEG_NERVE[leg.lower()], 'local': sensory[leg].tolist(),
        'root_ids': [str(x) for x in built['data']['root_ids'][built['global_index'][sensory[leg]]]],
        'translation_validated': False} for leg in LEGS}
    leg_joints = {}
    for leg in LEGS:
        joints = [j for j in range(model.njnt) if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT,j) or '').startswith('joint_'+leg)]
        if len(joints) != 7: raise ValueError('expected seven joints for '+leg)
        leg_joints[leg] = [(int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]), model.jnt_range[j].copy()) for j in joints]
    th = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'Thorax')
    ground = set(np.flatnonzero(model.geom_type == int(mj.mjtGeom.mjGEOM_PLANE)))
    geom_names = [mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,g) or '' for g in range(model.ngeom)]
    chunk = .005
    neural_dt = cfg.dt_ms/1000
    nsub = round(chunk/neural_dt); body_steps = round(chunk/model.opt.timestep)
    if not np.isclose(body_steps*model.opt.timestep, chunk) or not np.isclose(args.seconds/chunk,round(args.seconds/chunk)):
        raise ValueError('clocks must be integral')
    report['translation_laws'] = {'sensory': 'max_nA * clip(.4*binary_foot_contact + .3*mean(abs(angle-midrange)/halfrange) + .3*mean(abs(joint_velocity)/10rad_s),0,1)',
                                  'motor': 'sum(spikes over selected MNs)/(N_selected * window_s) /100Hz clipped [0,1]',
                                  'chunk_s':chunk,'neural_dt_s':neural_dt,'body_dt_s':float(model.opt.timestep),
                                  'max_sensory_nA':args.sensory_max_na}
    if args.sensory_mode=='front-club-motion':
        report['translation_laws']['sensory']='ONLY identified front-leg club: max_nA*clip(abs(actual femur-tibia angular velocity)/10rad_s,0,1); no rest current, no target, no vibration; uncalibrated engineering motion proxy'
    if args.sensory_mode=='fecc-classes' and args.sensory_battery=='leg-full':
        report['translation_laws']['sensory']=('SIX-LEG battery, class-level, all uncalibrated: '
            'chordotonal claw = |femur-tibia angle - midrange|/halfrange; chordotonal hook/club = '
            '|femur-tibia angular velocity|/10rad_s; hair plate = mean |joint angle deviation| over that '
            'leg joints; campaniform = that leg tarsus ground contact NORMAL FORCE / body weight; '
            'tactile bristle = binary tarsus contact. No rest current, no target, no schedule, no '
            'per-cell tuning or preferred direction (not available locally).')
    elif args.sensory_mode=='fecc-classes':
        report['translation_laws']['sensory']=('identified front-leg chordotonal subclasses, class-level: '
            'claw = max_nA*clip(|actual femur-tibia angle - range midpoint|/(half range),0,1); '
            'hook and club = max_nA*clip(|actual femur-tibia angular velocity|/10rad_s,0,1). '
            'No rest current, no target, no schedule, no per-cell preferred direction (not available '
            'locally); uncalibrated prox[y] of the documented class property, not a measured current')
        report['sensory_evidence_url']='https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/'
    # Regression: identical firing per neuron must not double activation for a doubled group.
    fake=np.ones((2,2),dtype=bool)
    assert motor_rate(fake,[0],.01)==motor_rate(fake,[0,1],.01)==200
    body_weight=float(mj.mj_getTotalmass(model)*abs(float(model.opt.gravity[2])))
    cases = {}
    # per-actuator command scale: 1.0 everywhere unless a DESIGN overlay declares a coupled
    # fibre-cluster pair (one pool driving two actuators) that must be driven at half strength.
    if args.design_overlay is not None:
        # report the scale in MAPPING ORDER (muscle-channel order), which is the order the ctrl
        # vector uses after `full_ctrl[muscle_ctrl_index]=applied_ctrl`
        assert len(whitelist)==len(mapping), 'whitelist and mapping must be in the same order'
        design_scale=np.asarray(report['command_scale_by_mapping_order'],float)
    else:
        design_scale=np.ones(len(mapping))
    assert design_scale.shape==(len(mapping),), 'command scale must be per muscle actuator'
    case_names=['intact','sensory_cut','motor_readout_cut','propagation_cut']
    if args.split_propagation_cuts:case_names+=['excitatory_cut','inhibitory_cut']
    # Neural interventions.  The convention is the standard one used for connectome
    # models: silencing a neuron means removing ALL of its inbound and outbound
    # synapses, so it neither influences nor is influenced by the rest of the network.
    # The random control silences a size-matched random neuron set, so any effect can
    # be attributed to the specific neurons rather than to removing synapses in general.
    intervention_sets={}
    if args.interventions:
        mn_local=sorted({int(i) for r in mapping for i in r['local']})
        assert mn_local, 'no mapped motor neurons to silence'
        rng_ctl=np.random.default_rng(4242)
        pool=[i for i in range(built['n']) if i not in set(mn_local)]
        control_local=sorted(rng_ctl.choice(pool,size=len(mn_local),replace=False).tolist())
        intervention_sets={'motor_neuron_silenced':mn_local,'random_control_silenced':control_local}
        case_names+=list(intervention_sets)
        report['neural_interventions']={
            'convention':'silencing removes all inbound and outbound synapses (rows and columns '
                         'of both We and Wi) for the selected neurons - the standard connectome-model '
                         'silencing convention; the network, body, sensory translation and motor '
                         'decoder are otherwise untouched',
            'n_motor_neurons_silenced':len(mn_local),
            'random_control_size':len(control_local),
            'random_control_seed':4242,
            'scope':'interventions only; nothing here drives the body'}
    for case in case_names:
        case_built=built
        cut_keys={'propagation_cut':('We','Wi'),'excitatory_cut':('We',),'inhibitory_cut':('Wi',)}.get(case,())
        if case in intervention_sets:
            silenced=intervention_sets[case]
            case_built=dict(built);case_built['edges']=dict(built['edges'])
            for key in ('We','Wi'):
                mat=built['edges'][key].tolil().copy()
                mat[silenced,:]=0.0
                mat[:,silenced]=0.0
                mat=mat.tocsr();mat.eliminate_zeros()
                case_built['edges'][key]=mat
        if cut_keys:
            case_built=dict(case_built);case_built['edges']=dict(case_built['edges'])
            for key in cut_keys:
                mat=case_built['edges'][key].copy();mat.data[:]=0;mat.eliminate_zeros()
                assert mat.nnz==0
                case_built['edges'][key]=mat
        tier=NeuralTier(case_built,cfg,OwnershipLedger())
        assert np.count_nonzero(tier.background_nA)==0
        d=mj.MjData(model);mj.mj_resetDataKeyframe(model,d,0)
        if args.stance_file is not None:
            stance=json.loads(Path(args.stance_file).read_text())
            pose=stance.get('best',stance)
            joints=pose['joints_rad'] if 'joints_rad' in pose else pose
            applied=0
            for nm,val in joints.items():
                jid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,nm)
                if jid<0:raise ValueError('stance file joint not in model: '+nm)
                d.qpos[int(model.jnt_qposadr[jid])]=float(val);applied+=1
            if 'root_z_mm' in pose:d.qpos[2]=float(pose['root_z_mm'])
            report['stance_file']={'path':str(Path(args.stance_file).resolve()),
                                   'joints_applied':applied,
                                   'root_z_model':float(d.qpos[2]),
                                   'note':'searched stance used as the initial condition; the XML keyframe '
                                          'is NOT used in this run'}
        d.ctrl[:]=0;d.act[:]=0;d.qvel[:]=0
        d.qfrc_applied[:]=0;d.xfrc_applied[:]=0
        mj.mj_forward(model,d)
        commands=[];body_trace=[{'time_s':0.,'height_model':float(d.xpos[th,2]),'ctrl_max':0.,'act_max':0.}]
        max_non_sensory=max_cut_ctrl=max_cut_act=0.
        pending_ctrl=np.zeros(n_muscles)
        pending_available_s=0.
        for interval in range(round(args.seconds/chunk)):
            foot={leg:False for leg in LEGS}
            for contact in d.contact:
                g1,g2=int(contact.geom1),int(contact.geom2)
                other=g2 if g1 in ground else g1 if g2 in ground else None
                if other is not None:
                    nm=geom_names[other]
                    if nm[:2] in foot and 'Tarsus' in nm:foot[nm[:2]]=True
            currents=np.zeros(tier.n)
            features={}
            for leg in (LEGS if args.sensory_mode=='legacy-broad' else ()):
                angles=[];vels=[]
                for qa,va,rg in leg_joints[leg]:
                    angles.append(abs(float(d.qpos[qa])-.5*sum(rg))/max(.5*(rg[1]-rg[0]),1e-6))
                    vels.append(abs(float(d.qvel[va]))/10.)
                f=float(np.clip(.4*foot[leg]+.3*np.mean(angles)+.3*np.mean(vels),0,1))
                features[leg]={'binary_foot_contact':foot[leg], 'angle_feature':float(np.mean(angles)), 'velocity_feature':float(np.mean(vels)), 'current_nA':0. if case=='sensory_cut' else args.sensory_max_na*f}
                currents[sensory[leg]]=features[leg]['current_nA']
            if args.sensory_mode in ('front-club-motion','fecc-classes'):
                currents[:]=0
                features={}
                for leg in ('LF','RF'):
                    jid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,'joint_'+leg+'Tibia_pitch')
                    if jid<0:raise ValueError('front femur-tibia joint missing')
                    dof=int(model.jnt_dofadr[jid])
                    velocity=float(d.qvel[dof])
                    lo,hi=model.jnt_range[jid]
                    mid=.5*(lo+hi);half=max(.5*(hi-lo),1e-6)
                    angle=float(d.qpos[int(model.jnt_qposadr[jid])])
                    motion=min(abs(velocity)/10.,1.)          # 10 rad/s = declared normalisation
                    position=min(abs(angle-mid)/half,1.)      # sign-free position deviation
                    if case=='sensory_cut':
                        motion=position=0.
                    for subtype,level in sensory_groups[leg].items():
                        value=position if subtype=='claw' else motion
                        currents[level]=args.sensory_max_na*value
                    features[leg]={'joint':'joint_'+leg+'Tibia_pitch','angular_velocity_rad_s':velocity,
                                   'angle_rad':angle,'motion_level':motion,'position_level':position,
                                   'current_nA_injected':float(args.sensory_max_na*motion) if case!='sensory_cut' else 0.,
                                   'subtypes_driven':list(sensory_groups[leg]),
                                   'calibrated':False,'per_cell_direction_available':False,
                                   'vibration_implemented':False}
            if args.sensory_mode=='fecc-classes' and args.sensory_battery=='leg-full':
                # SIX-LEG battery: each channel is a documented CLASS-LEVEL property driven by a
                # body-available variable with a declared normalisation. No target, no schedule and
                # no per-cell tuning (per-root direction/tuning is not available locally).
                currents[:]=0
                features={}
                res=np.zeros(6)
                foot_force={leg:0.0 for leg in LEGS}
                foot_touch={leg:False for leg in LEGS}
                for _k in range(d.ncon):
                    _c=d.contact[_k]
                    _g1,_g2=int(_c.geom1),int(_c.geom2)
                    _o=_g2 if _g1 in ground else _g1 if _g2 in ground else None
                    if _o is None:
                        continue
                    _nm=geom_names[_o]
                    if not ('Tarsus' in _nm and _nm[:2] in LEGS):
                        continue
                    mj.mj_contactForce(model,d,_k,res)
                    foot_force[_nm[:2]]+=abs(float(res[0]))   # normal component of that contact
                    foot_touch[_nm[:2]]=True
                for leg in LEGS:
                    jid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,'joint_'+leg+'Tibia_pitch')
                    if jid<0:raise ValueError('femur-tibia joint missing for '+leg)
                    dof=int(model.jnt_dofadr[jid]);lo,hi=model.jnt_range[jid]
                    mid=.5*(lo+hi);half=max(.5*(hi-lo),1e-6)
                    angle=float(d.qpos[int(model.jnt_qposadr[jid])])
                    motion=min(abs(float(d.qvel[dof]))/10.,1.)
                    position=min(abs(angle-mid)/half,1.)
                    devs=[]
                    for qa,va,rg in leg_joints[leg]:
                        m2=.5*float(rg[0]+rg[1]);h2=max(.5*float(rg[1]-rg[0]),1e-6)
                        devs.append(min(abs(float(d.qpos[qa])-m2)/h2,1.))
                    hair=float(np.mean(devs)) if devs else 0.
                    load=min(foot_force[leg]/max(body_weight,1e-12),1.)
                    touch=1. if foot_touch[leg] else 0.
                    levels={'chordotonal_claw':position,'chordotonal_hook':motion,
                            'chordotonal_club':motion,'hair_plate':hair,
                            'campaniform':load,'tactile_bristle':touch}
                    if case=='sensory_cut':
                        levels={k:0. for k in levels}
                    for modal,level in levels.items():
                        idx=leg_channels['channels'][leg].get(modal,[])
                        if idx:
                            currents[np.asarray(idx,int)]=args.sensory_max_na*float(level)
                    features[leg]={'femur_tibia_angle_rad':angle,'motion_level':motion,
                                   'position_level':position,'hair_plate_level':hair,
                                   'campaniform_load_level':load,'tactile_level':touch,
                                   'tarsus_contact_force':float(foot_force[leg]),
                                   'calibrated':False,'per_cell_direction_available':False}
            if args.sensory_cut_after is not None and float(d.time) >= args.sensory_cut_after:
                currents[:] = 0.
                for k in features:
                    features[k]['current_nA'] = 0.
                features['_sensory_cut_after_s'] = args.sensory_cut_after
            assert not np.any(currents[nonsensory])
            max_non_sensory=max(max_non_sensory,float(np.max(np.abs(currents[nonsensory]),initial=0)))
            spikes=tier.advance(nsub,currents)
            if case=='propagation_cut':
                assert not np.any(spikes[:,nonsensory]), 'disconnected nonsensory neuron fired'
            if case in intervention_sets:
                # A silenced neuron can still spike if it is a SENSORY cell that receives
                # injected external current: that current is an environmental input, not a
                # synapse, so it is expected to survive.  Neurons without injected current
                # must be silent.
                silenced_wo_input=[i for i in intervention_sets[case] if i not in set(allowed_sensory)]
                assert not np.any(spikes[:,silenced_wo_input]), \
                    'a neuron with all its synapses removed and no injected current still spiked'
            rates=np.array([motor_rate(spikes,r['local'],chunk) for r in mapping])
            ctrl=np.clip(rates/100.,0,1)
            if args.design_overlay is not None:
                ctrl=np.clip(ctrl*design_scale,0,1)
            if case=='motor_readout_cut':ctrl[:]=0
            # End-of-neural-window spikes cannot act at its beginning.
            # Body [t,t+chunk] uses only the prior completed window's command.
            assert pending_available_s <= float(d.time)+1e-10
            applied_ctrl=pending_ctrl.copy()
            applied_available_s=pending_available_s
            full_ctrl=np.zeros(model.nu)
            full_ctrl[muscle_ctrl_index]=applied_ctrl
            if adhesion_actuators:
                full_ctrl[adhesion_actuators]=args.adhesion_const
            d.ctrl[:]=full_ctrl
            assert np.all(d.ctrl[adhesion_actuators]==args.adhesion_const), \
                'adhesion command must stay at the declared constant'
            pending_ctrl=ctrl.copy()
            pending_available_s=(interval+1)*chunk
            from engine.embodied.neural_command_audit import audit_interval
            origin = audit_interval(spikes=spikes, groups=[r['local'] for r in mapping],
                window_s=chunk, reference_hz=100., actual_ctrl=ctrl,
                sensory_current=currents, sensory_indices=allowed_sensory,
                background_current=tier.background_nA, readout_connected=case!='motor_readout_cut',
                command_scale=design_scale)
            if origin['status'] != 'PASS':
                raise ValueError('Runtime neural command origin audit failed: '+str(origin))
            if case=='motor_readout_cut':
                assert np.count_nonzero(d.ctrl[muscle_ctrl_index])==0, \
                    'MUSCLE command must stay zero under the readout cut'
                assert np.count_nonzero(d.act)==0, \
                    'activation must stay zero naturally under zero muscle command'
            contact_record=None
            if args.record_contacts:
                touched={}
                for contact in d.contact:
                    g1,g2=int(contact.geom1),int(contact.geom2)
                    other=g2 if g1 in ground else g1 if g2 in ground else None
                    if other is None: continue
                    nm=geom_names[other]
                    touched[nm]=touched.get(nm,0)+1
                names_now=sorted(touched)
                tarsus_only=bool(names_now) and all('Tarsus' in n for n in names_now)
                contact_record={'n_ground_contacts':int(sum(touched.values())),
                                'ground_geoms':names_now,'tarsus_only':tarsus_only,
                                'non_tarsus_geoms':[n for n in names_now if 'Tarsus' not in n],
                                'legs_with_foot_contact':sorted({n[:2] for n in names_now if 'Tarsus' in n and n[:2] in LEGS})}
            commands.append({'time_s':float(d.time),'features':features,'contacts':contact_record,
                'injected_current_sum_nA':float(currents.sum()), 'nonsensory_injected_current_max_nA':max_non_sensory,
                'network_spike_count':int(spikes.sum()),
                'spikes_sparse_time_neuron':(np.argwhere(spikes).tolist()
                                             if args.spike_storage=='full' else None),
                'mn_rate_hz':rates.tolist(), 'ctrl':ctrl.tolist(), 'command_origin_audit':origin,
                'ctrl_semantics':'computed at window end; pending for next physical window',
                'computed_available_s':pending_available_s,
                'applied_ctrl':applied_ctrl.tolist(), 'applied_available_s':applied_available_s,
                'causal_application_audit':bool(applied_available_s <= float(d.time)+1e-10 and
                    np.array_equal(applied_ctrl,commands[-1]['ctrl'] if commands else np.zeros(n_muscles))),
                'adhesion_const_applied':float(args.adhesion_const) if adhesion_actuators else None})
            assert commands[-1]['causal_application_audit']
            for _ in range(body_steps):
                if case in ('motor_readout_cut','propagation_cut'):
                    assert np.count_nonzero(d.ctrl[muscle_ctrl_index])==np.count_nonzero(d.act)==0, \
                        'muscle command and activation must both stay zero'
                    if adhesion_actuators:
                        assert np.all(d.ctrl[adhesion_actuators]==args.adhesion_const), \
                            'adhesion must remain at its declared constant in every branch'
                mj.mj_step(model,d);mj.mj_forward(model,d)
                if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):raise ValueError('nonfinite body state')
                if case in ('motor_readout_cut','propagation_cut'):
                    max_cut_ctrl=max(max_cut_ctrl,float(np.max(np.abs(d.ctrl[muscle_ctrl_index]))))
                    max_cut_act=max(max_cut_act,float(np.max(np.abs(d.act))))
                    assert max_cut_ctrl==max_cut_act==0
                _q=np.asarray(d.qpos[3:7],float)
                _w,_x,_y,_z=_q
                _upz=float(1.0-2.0*(_x*_x+_y*_y))     # world z of the body z-axis
                _foot={}
                for _c in d.contact:
                    _g1,_g2=int(_c.geom1),int(_c.geom2)
                    _o=_g2 if _g1 in ground else _g1 if _g2 in ground else None
                    if _o is None:continue
                    _nm=geom_names[_o]
                    if 'Tarsus' in _nm and _nm[:2] in LEGS:_foot[_nm[:2]]=True
                body_trace.append({'time_s':float(d.time),'height_model':float(d.xpos[th,2]),
                                   'root_x_model':float(d.qpos[0]),'root_y_model':float(d.qpos[1]),
                                   'torso_up_z':_upz,
                                   'foot_contact':{k:bool(_foot.get(k,False)) for k in LEGS},
                                   'ctrl_max':float(np.max(np.abs(d.ctrl))),'act_max':float(np.max(np.abs(d.act))),
                                   'actuator_force_absmax':float(np.max(np.abs(d.actuator_force))),
                                   'applied_command_available_s':applied_available_s})
        cases[case]={'semantics':{'intact':'exploratory sensory injection and MN readout',
                        'sensory_cut':'external sensory input cut only, no neuron lesion',
                        'motor_readout_cut':'MN readout cut only; neural spikes retained, ctrl and act zero; passive muscle force permitted',
                         'propagation_cut':'All We/Wi zero; same nodes/physical sensory input, no readout masking or forced activation overwrite',
                         'excitatory_cut':'Only We zero, Wi preserved; isolated diagnostic, no readout masking',
                         'inhibitory_cut':'Only Wi zero, We preserved; isolated diagnostic, no readout masking',
                         'motor_neuron_silenced':'All synapses of the mapped motor neurons removed '
                             '(inbound and outbound); the network still runs, so muscle command can '
                             'only fall to zero if those neurons were the source',
                         'random_control_silenced':'Size-matched random neurons silenced the same way; '
                             'controls for the effect of removing synapses in general'}[case],
                      'synaptic_intervention':{'zeroed_matrices':list(cut_keys),
                          'remaining_We_nnz':int(case_built['edges']['We'].nnz),
                          'remaining_Wi_nnz':int(case_built['edges']['Wi'].nnz)},
                     'commands':commands,'body_per_step':body_trace,
                     'invariants':{'background_max_nA':float(np.max(np.abs(tier.background_nA))),
                          'nonsensory_current_max_nA':max_non_sensory,'cut_ctrl_max':max_cut_ctrl,'cut_act_max':max_cut_act},
                     'summary':{'final_height_mm':body_trace[-1]['height_mm'],'min_height_mm':min(x['height_mm'] for x in body_trace),
                                'network_spikes':sum(x['network_spike_count'] for x in commands),
                                'max_ctrl':max(max(x['ctrl']) for x in commands)}}
        print(case,cases[case]['summary'],flush=True)
    report['cases']=cases
    report['execution_status']='EXPLORATORY_MEASUREMENT_COMPLETED'
    assert digest(XML)==report['source_sha256'][str(XML)], 'canonical XML changed during diagnostic'
    (OUT/'exploratory.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,1,figsize=(10,10),sharex=True)
    for name,c in cases.items():
        b=c['body_per_step'];r=c['commands']
        axs[0].plot([x['time_s'] for x in b],[x['height_mm'] for x in b],label=name)
        axs[1].plot([x['time_s'] for x in r],[x['injected_current_sum_nA'] for x in r],label=name)
        axs[2].plot([x['time_s'] for x in r],[max(x['applied_ctrl']) for x in r],label=name)
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    axs[0].set_ylabel('Thorax height (mm)');axs[1].set_ylabel('Sensory injected sum (nA)');axs[2].set_ylabel('Max muscle control');axs[2].set_xlabel('Time (s)')
    fig.suptitle('PARTIAL_TRANSLATION_NOT_VALIDATED_AUTONOMY\nNo target / CPG / tonic / background. Input and readout cuts, NOT lesions.')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'exploratory.png',dpi=140)
    print('PARTIAL_TRANSLATION_NOT_VALIDATED_AUTONOMY',OUT)
    return 0


if __name__=='__main__':raise SystemExit(main())
