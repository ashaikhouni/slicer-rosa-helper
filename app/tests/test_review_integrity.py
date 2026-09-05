"""Review integrity across geometry edits, failed saves, and cohort reuse."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from rosa_service.app import create_app
from rosa_service.models import ReviewEdit, ReviewOp
from rosa_service.review import ReviewStore, ReviewPersistenceError


class ReviewIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.case = self.root / 'case'
        self.case.mkdir()
        self.contact_header = 'trajectory\tlabel\tcontact_index\tx\ty\tz\telectrode_model\tregion\n'
        self.contacts = self.contact_header + ''.join(
            f'{s}\t{s}{i}\t{i}\t0\t0\t{i}\tTest\tOriginal region\n'
            for s in ('A', 'B') for i in (1, 2))
        (self.case / 'contacts.tsv').write_text(self.contacts)
        self.traj = 'name\tstart_x\tstart_y\tstart_z\tend_x\tend_y\tend_z\nA\t0\t0\t0\t0\t0\t3\n'
        (self.case / 'trajectories.tsv').write_text(self.traj)
        self.manifest = {'id': 'case', 'kind': 'pipeline', 'state': 'succeeded', 'params': {}}
        self.save_manifest()
        self.store = ReviewStore()

    def save_manifest(self):
        (self.case / 'manifest.json').write_text(json.dumps(self.manifest))

    def reject(self):
        self.store.apply('case', self.case, [
            ReviewEdit(op=ReviewOp.reject_contact, shank='A', index=1),
            ReviewEdit(op=ReviewOp.reject_shank, shank='B')])

    def test_geometry_edit_preserves_exclusions_and_invalidates_moved_labels(self):
        self.reject()
        (self.case / 'contacts.tsv').write_text(self.contacts.replace('A\tA1\t1\t0', 'A\tA1\t1\t10'))
        doc = self.store.rebuild_preserving_labels('case', self.case)
        self.assertFalse(doc.shanks[0].contacts[0].accepted)
        self.assertFalse(doc.shanks[1].accepted)
        self.assertIsNone(doc.shanks[0].contacts[0].region)
        self.assertTrue(doc.shanks[0].contacts[0].region_stale)
        self.assertEqual(doc.shanks[0].contacts[1].region, 'Original region')
        restored = ReviewStore().get_or_build('case', self.case)
        self.assertEqual(doc, restored)

    def test_rename_preserves_inclusion_and_unchanged_labels(self):
        self.reject()
        (self.case / 'contacts.tsv').write_text(self.contacts.replace('A\tA', 'NEW\tNEW'))
        doc = self.store.rebuild_preserving_labels('case', self.case, {'NEW': 'A'})
        self.assertEqual(doc.shanks[0].name, 'NEW')
        self.assertFalse(doc.shanks[0].contacts[0].accepted)
        self.assertEqual(doc.shanks[0].contacts[0].region, 'Original region')

    def test_failed_save_preserves_disk_and_memory(self):
        self.store.get_or_build('case', self.case)
        saved = (self.case / 'review.json').read_bytes()
        with patch('rosa_service.review.os.replace', side_effect=OSError('disk full')):
            with self.assertRaises(ReviewPersistenceError):
                self.reject()
        self.assertEqual(saved, (self.case / 'review.json').read_bytes())
        self.assertTrue(self.store.get_or_build('case', self.case).shanks[0].contacts[0].accepted)
        self.assertEqual(list(self.case.glob('.review.json.*')), [])

    def test_invalid_batch_does_not_partially_apply(self):
        self.store.get_or_build('case', self.case)
        with self.assertRaises(ValueError):
            self.store.apply('case', self.case, [
                ReviewEdit(op=ReviewOp.reject_contact, shank='A', index=1),
                ReviewEdit(op=ReviewOp.reject_shank, shank='missing')])
        self.assertTrue(self.store.get_or_build('case', self.case).shanks[0].contacts[0].accepted)

    def test_corrupt_review_is_not_overwritten(self):
        (self.case / 'review.json').write_text('{broken')
        with self.assertRaises(ReviewPersistenceError):
            self.store.get_or_build('case', self.case)
        self.assertEqual((self.case / 'review.json').read_text(), '{broken')

    def test_cohort_respects_both_contact_and_shank_rejection(self):
        self.reject()
        rows = [{'trajectory': s, 'contact_index': str(i), 'name': f'{s}{i}',
                 'mni_x': 0, 'mni_y': 0, 'mni_z': i, 'hemisphere': 'L'}
                for s in ('A', 'B') for i in (1, 2)]
        with patch('rosa_core.cohort.mni_transforms_present', return_value=True), \
             patch('rosa_core.cohort.ensure_contacts_mni', return_value=rows):
            result = TestClient(create_app(work_root=self.root)).get('/api/v1/cohort/contacts')
        self.assertEqual(result.status_code, 200)
        contacts = result.json()['subjects'][0]['contacts']
        self.assertEqual([c[-1] for c in contacts], ['A2'])

    def test_save_failure_is_visible_http_error(self):
        self.store.get_or_build('case', self.case)
        client = TestClient(create_app(work_root=self.root))
        with patch('rosa_service.review.os.replace', side_effect=OSError('disk full')):
            result = client.patch('/api/v1/jobs/case/review', json={'ops': [
                {'op': 'reject_contact', 'shank': 'A', 'index': 1}]})
        self.assertEqual(result.status_code, 507)
        self.assertIn('could not be saved', result.json()['detail'])

    def test_failed_geometry_review_save_restores_case(self):
        self.reject()
        saved = {n: (self.case / n).read_bytes() for n in
                 ('contacts.tsv', 'trajectories.tsv', 'review.json')}
        plan = {'origin': [0, 0, 0], 'models': {'Test': {'offsets': [2, 1]}},
                'trajectories': [{'name': 'A', 'model': 'Test',
                                  'entry': [10, 0, 0], 'target': [10, 0, 3]}]}
        app = create_app(work_root=self.root)
        with patch.object(app.state.reviews, '_persist', side_effect=ReviewPersistenceError('disk full')):
            result = TestClient(app).post('/api/v1/jobs/case/editor/plan', json=plan)
        self.assertEqual(result.status_code, 507)
        for n, original in saved.items():
            self.assertEqual((self.case / n).read_bytes(), original)

    def test_accuracy_requires_independent_plan_and_valid_hash(self):
        (self.case / 'ros_plan.tsv').write_text(self.traj)
        client = TestClient(create_app(work_root=self.root))
        self.assertFalse(client.get('/api/v1/jobs/case/plan-accuracy').json()['has_plan'])
        self.manifest['params'] = {'plan_source': 'rosa_import',
                                  'plan_sha256': hashlib.sha256(self.traj.encode()).hexdigest()}
        self.save_manifest()
        self.assertTrue(client.get('/api/v1/jobs/case/plan-accuracy').json()['has_plan'])
        (self.case / 'ros_plan.tsv').write_text(self.traj.replace('A\t0', 'A\t1'))
        self.assertFalse(client.get('/api/v1/jobs/case/plan-accuracy').json()['has_plan'])

    def test_legacy_import_verified_against_original_seeds(self):
        seeds = self.root / 'seeds.tsv'
        seeds.write_text(self.traj)
        (self.case / 'ros_plan.tsv').write_text(self.traj)
        self.manifest['params'] = {'seeds': str(seeds)}
        self.save_manifest()
        client = TestClient(create_app(work_root=self.root))
        self.assertTrue(client.get('/api/v1/jobs/case/plan-accuracy').json()['has_plan'])
        seeds.unlink()
        self.assertFalse(client.get('/api/v1/jobs/case/plan-accuracy').json()['has_plan'])

    def test_near_identity_naming_fit_cannot_create_accuracy_plan(self):
        from types import SimpleNamespace
        import importlib
        matcher = importlib.import_module('rosa_core.cross_volume_match')
        import numpy as np
        lines = [{'name': name, 'start': [i * 10, 0, 0], 'end': [i * 10, 0, 30]}
                 for i, name in enumerate(('A', 'B', 'C'))]
        result = SimpleNamespace(pairs=[(name, name, 0, 0) for name in ('A', 'B', 'C')],
                                 transform_4x4=np.eye(4), refined_inliers=3)
        with patch('rosa_service.app._read_traj_lines', return_value=lines), \
             patch('rosa_service.app._ros_plan_lines', return_value=lines), \
             patch.object(matcher, 'cross_volume_match', return_value=result):
            response = TestClient(create_app(work_root=self.root)).post(
                '/api/v1/jobs/case/match-ros', json={'ros_text': 'synthetic'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['inliers'], 3)
        self.assertFalse((self.case / 'ros_plan.tsv').exists())

    def test_unrecordable_new_job_is_rejected(self):
        client = TestClient(create_app(work_root=self.root))
        with patch('rosa_service.jobs.atomic_write_text', side_effect=OSError('disk full')):
            response = client.post('/api/v1/jobs', json={'kind': 'selftest'})
        self.assertEqual(response.status_code, 507)

    def test_interrupted_job_reopens_as_failed(self):
        self.manifest['state'] = 'running'
        self.save_manifest()
        response = TestClient(create_app(work_root=self.root)).get('/api/v1/jobs/case')
        self.assertEqual(response.json()['state'], 'failed')
        self.assertIn('interrupted', response.json()['error'])
