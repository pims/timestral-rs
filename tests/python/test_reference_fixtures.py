"""Check categorical fixtures against the pinned Python reference (no Rust or weights).

From the repository root:
  source .venv/bin/activate
  python tests/python ReferenceFixtures
"""
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import timestral.covariate_encoding as encoding
import timestral.forecaster as forecasting
import timestral.scaler as scaler

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures'


def values(items):
    return np.asarray([np.nan if v is None else v for v in items], dtype=np.float32)


class ReferenceFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.encoding = json.loads((FIXTURES / 'categorical-encoding.json').read_text())
        cls.normalization = json.loads((FIXTURES / 'categorical-known-normalization.json').read_text())

    def test_reference_revision_and_sources(self):
        metadata = importlib.metadata.distribution('timestral')
        provenance = json.loads(metadata.read_text('direct_url.json'))
        self.assertEqual(provenance['vcs_info']['commit_id'], self.encoding['reference_commit'])
        self.assertEqual(np.__version__, self.encoding['numpy_version'])
        self.assertEqual(torch.__version__.split('+')[0], self.normalization['torch_version'])
        for module in (encoding, forecasting):
            digest = hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
            self.assertEqual(digest, self.encoding['source_sha256'][module.__name__])
        digest = hashlib.sha256(Path(inspect.getfile(scaler)).read_bytes()).hexdigest()
        self.assertEqual(digest, self.normalization['scaler_sha256'])

    @torch.inference_mode()
    def test_categorical_encoding_cases(self):
        self.assertEqual(len(self.encoding['cases']), 23)
        for case in self.encoding['cases']:
            with self.subTest(case=case['name']):
                reference = SimpleNamespace(device=torch.device('cpu'), max_context=case['max_context'])
                inputs, mask = forecasting.TimestralForecaster._as_context(
                    reference, values(case['history']), case.get('observation_mask'))
                target = forecasting.TimestralForecaster._encoding_target(reference, inputs, mask)
                np.testing.assert_array_equal(target[0].numpy(), values(case['encoding_target']))
                labels = case['past'] + case.get('future', [])
                identities = dict.fromkeys(label for label in labels if label is not None)
                codes = {label: float(i) for i, label in enumerate(identities)}
                row = [np.nan if label is None else codes[label] for label in labels]
                actual = forecasting.TimestralForecaster._encode_categorical(
                    reference, torch.tensor(row, dtype=torch.float64)[None, None], target)[0, 0].numpy()
                n = len(case['history'])
                np.testing.assert_allclose(actual[:n], values(case['encoded_past']),
                                           rtol=1e-6, atol=1e-6, equal_nan=True)
                if case['encoded_future'] is not None:
                    np.testing.assert_allclose(actual[n:], values(case['encoded_future']),
                                               rtol=1e-6, atol=1e-6, equal_nan=True)
                else:
                    self.assertEqual(len(actual), n)

    def test_exact_float64_target_sums(self):
        self.assertEqual(len(self.encoding['mean_reductions']), 28)
        pattern = values(self.encoding['reduction_pattern'])
        for case in self.encoding['mean_reductions']:
            with self.subTest(length=case['length']):
                target = np.resize(pattern, case['length']).astype(np.float64)
                target = np.where(np.isfinite(target), target, 0.0)
                self.assertEqual(f'{target.sum().view(np.uint64):016x}', case['expected_sum_bits'])

    def test_known_normalization_cases(self):
        self.assertEqual(len(self.normalization['cases']), 22)
        for case in self.normalization['cases']:
            with self.subTest(length=case['length'], constant=case['constant']):
                n = case['length']
                row = np.asarray([104.45187 + (0 if case['constant'] else (i % 17)*0.001)
                                  for i in range(n)], dtype=np.float32)
                observed = np.asarray([i % 7 != 0 for i in range(n)])
                x, mask = torch.from_numpy(row)[None], torch.from_numpy(observed)[None]
                loc, scale, _ = scaler.instance_norm_stats(x, mask, minimum_scale=1e-6)
                self.assertEqual((x*mask).sum().item(), case['sum'])
                self.assertEqual(loc.item(), case['mean'])
                self.assertEqual(scale.item(), case['scale'])
                normalized = scaler.apply_affine_asinh_masked(x, loc, scale, mask, use_asinh=True)[0]
                np.testing.assert_allclose(normalized[:min(n, 32)].numpy(), case['normalized_prefix'],
                                           rtol=0, atol=2e-6)
