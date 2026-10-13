"""Focused stdlib audit contracts; no simulation imports/builds."""
import json
import unittest
from collections import Counter
from run_sensory_tuning_annotation_audit import OUT, exact_groups, unique_join, polarity, summarize, root, sha

class AnnotationAuditTests(unittest.TestCase):
    def test_unique_root_join(self):
        self.assertEqual(unique_join(['720575941467386710','2'],{'720575941467386710'}),{'720575941467386710':0})
        with self.assertRaises(ValueError): unique_join(['1','1'],{'1'})
        with self.assertRaises(ValueError): unique_join(['1'],{'2'})
        with self.assertRaises(ValueError): root(720575941467386710.0)

    def test_missing_and_classification_finite(self):
        for v in (None,'','NA','NaN',float('nan')):
            self.assertEqual(polarity([v]),'missing_annotation')
        self.assertEqual(polarity(['SNpp39','left','position']),'no_explicit_polarity_tag')
        self.assertEqual(polarity(['claw_flexion']),'flexion_term_requires_review')
        self.assertEqual(polarity(['extension','flexion']),'both_terms_requires_review')
        json.dumps(summarize([None,float('nan'),'direction']),allow_nan=False)

    def test_group_integrity_rejects_wrong_or_missing(self):
        identity={'tier_root_ids':[101,102], 'sensory_groups':{'lf':{'claw_position':[0]}}}
        lookup={'leg_channel_by_root':{'101':['lf','claw_position']}}
        self.assertEqual(len(exact_groups(identity,lookup)),1)
        lookup['leg_channel_by_root']['102']=['lf','claw_position']
        with self.assertRaises(ValueError): exact_groups(identity,lookup)
        identity['sensory_groups']['lf']['claw_position']=[0,0]
        with self.assertRaises(ValueError): exact_groups(identity,lookup)

    def test_actual_artifact_exact_totals_and_provenance(self):
        d=json.loads((OUT/'sensory_tuning_annotation_audit.json').read_text(),
                     parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        self.assertEqual(d['tier_neurons'],12000)
        self.assertEqual(d['exact_groups_counts'],dict(claw_position=149,hook_direction=136,
            hairplate_angle=207,campaniform_strain=60,club_vibration=323,
            bristle_touch=2879,multidendritic_noxious=34))
        self.assertEqual(d['exact_claw_hook_rows'],285)
        self.assertEqual(d['exact_groups_total'],3788)
        self.assertEqual(d['exact_groups_total'],sum(d['exact_groups_counts'].values()))
        self.assertEqual(d['exact_groups_total'],len(d['exact_groups']))
        self.assertEqual(d['all_sensory_roots_joined_once'],d['exact_groups_total'])
        identity=json.loads(__import__('pathlib').Path(d['identity_path']).read_text())
        for rec in d['exact_groups']:
            self.assertEqual(str(identity['tier_root_ids'][rec['tier_index']]),rec['root_id'])
            self.assertIn(rec['tier_index'],identity['sensory_groups'][rec['leg']][rec['channel']])
        self.assertEqual(d['exact_claw_hook_rows'],d['exact_groups_counts']['claw_position']+d['exact_groups_counts']['hook_direction'])
        self.assertEqual(Counter(r['polarity_classification'] for r in d['per_root_claw_hook']),d['exact_claw_hook_polarity_counts'])
        self.assertTrue(all(r['assigned_polarity'] is None for r in d['per_root_claw_hook']))
        self.assertEqual(d['source_sha256_before'],d['source_sha256_after'])
        for path,digest in d['source_sha256_after'].items(): self.assertEqual(sha(path),digest,path)

if __name__=='__main__': unittest.main()
