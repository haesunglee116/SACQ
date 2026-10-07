import copy
import tempfile
import unittest
from pathlib import Path

import torch

from sacq.vector_quantization import quantize_vector_codebook
from sacq.index_correction import (
    assert_only_indices_changed, compensate_packed, decode_fixed, inverse_factor,
)
from sacq.statistics import FullMoment
from sacq.metrics import apply_swiglu_down_diag_curvature, gelu_derivative
from sacq.calibration_layers import QuantLinear, pack_indices
import sacq.calibration_layers as calibration


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)

    def fitted(self, metric='euclidean'):
        weight = torch.randn(3, 64)
        extra = {}
        if metric == 'qk_diag':
            extra['scalar_importance'] = torch.rand_like(weight).add_(0.2)
        elif metric == 'qk_full':
            extra.update(input_metric=torch.eye(64), output_importance=torch.ones(3))
        extra.setdefault('scalar_importance', None)
        dense, packed = quantize_vector_codebook(
            weight, group_size=64, vector_dim=4, codebook_size=8,
            sample_size=48, iters=3, seed=123, metric=metric,
            return_packed=True, scale_squared_weighting=False, **extra)
        return weight, dense, packed

    def test_fitting_paths(self):
        for metric in ('euclidean', 'qk_diag', 'qk_full'):
            with self.subTest(metric=metric):
                weight, dense, packed = self.fitted(metric)
                self.assertEqual(dense.shape, weight.shape)
                self.assertTrue(torch.isfinite(dense).all())
                self.assertEqual(packed['indices'].shape, (3, 1, 16))
                self.assertLess(int(packed['indices'].max()), 8)
                self.assertFalse(packed['codebook_fit_scale_squared'])

    def test_fixed_payload_and_correction(self):
        weight, _, packed = self.fitted()
        result, report = compensate_packed(packed, weight, torch.eye(64), block_columns=64)
        assert_only_indices_changed(packed, result)
        self.assertEqual(report['status'], 'PASS')
        # Identity H decouples vectors; nearest representable candidate cannot
        # worsen this particular objective. No general monotonicity is asserted.
        self.assertLessEqual(report['layer_output_mse_after'], report['layer_output_mse_before'] + 1e-5)
        changed = copy.deepcopy(result)
        changed['scales'] = changed['scales'] + 1
        with self.assertRaises(ValueError):
            assert_only_indices_changed(packed, changed)

    def test_input_moment_preserves_all_rows(self):
        x = torch.randn(2, 5, 8)
        moment = FullMoment(8, 'cpu', 3)
        moment.add(x)
        self.assertEqual(moment.rows, 10)
        torch.testing.assert_close(moment.total, x.reshape(-1, 8).T @ x.reshape(-1, 8))

    def test_factor(self):
        h = torch.eye(8)
        u = inverse_factor(h, 0.1)
        torch.testing.assert_close(u.T @ u, torch.linalg.inv(h * 1.1))
        with self.assertRaises(ValueError):
            inverse_factor(torch.zeros(8, 8), 0.1)

    def test_correlated_factor(self):
        x = torch.randn(20, 8)
        h = x.T @ x / len(x)
        u = inverse_factor(h, 0.1)
        regularized = h + torch.eye(8) * (0.1 * h.diag().mean())
        torch.testing.assert_close(u.T @ u, torch.linalg.inv(regularized))

    def test_resume_and_provenance_guard(self):
        weight = torch.randn(3, 128)
        _, packed = quantize_vector_codebook(
            weight, group_size=64, vector_dim=4, codebook_size=8,
            sample_size=96, iters=3, seed=123, metric='euclidean',
            scalar_importance=None, return_packed=True)
        x = torch.randn(160, 128)
        h = x.T @ x / len(x)
        expected, _ = compensate_packed(packed, weight, h, block_columns=64)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(InterruptedError):
                compensate_packed(packed, weight, h, checkpoint_dir=d,
                                  block_columns=64, stop_after_blocks=1)
            actual, report = compensate_packed(packed, weight, h,
                                               checkpoint_dir=d, block_columns=64)
            torch.testing.assert_close(expected['indices'], actual['indices'], rtol=0, atol=0)
            self.assertEqual(report['roles']['whole']['restored_from_column'], 64)
            with self.assertRaisesRegex(ValueError, 'provenance/inputs mismatch'):
                compensate_packed(packed, weight + 0.1, h,
                                  checkpoint_dir=d, block_columns=64)

    def test_fused_qkv_keeps_separate_books(self):
        roles, weights = {}, []
        for role in ('q', 'k', 'v'):
            weight, _, packed = self.fitted()
            roles[role] = packed
            weights.append(weight)
        payload = {'format': 'packed_separate_qkv_codebooks_v1', 'roles': roles}
        result, report = compensate_packed(payload, torch.cat(weights), torch.eye(64))
        assert_only_indices_changed(payload, result)
        self.assertEqual(set(report['roles']), {'q', 'k', 'v'})
        self.assertEqual(decode_fixed(result, 'cpu').shape, (9, 64))

    def test_deterministic_cpu_fitting(self):
        torch.manual_seed(123)
        _, _, first = self.fitted('qk_full')
        torch.manual_seed(123)
        _, _, second = self.fitted('qk_full')
        for key in ('codebook', 'scales', 'indices'):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_optional_fp32_gather(self):
        book = torch.randn(5, 4, requires_grad=True)
        index = torch.tensor([0, 0, 3, 4, 3])
        result = calibration._FP32CodebookGather.apply(book, index)
        torch.testing.assert_close(result, book.to(torch.bfloat16)[index], rtol=0, atol=0)
        result.float().sum().backward()
        torch.testing.assert_close(book.grad, torch.tensor([2, 0, 0, 2, 1])[:, None].expand(5, 4).float())

    def test_downstream_weighting(self):
        g, u, w = torch.ones(3, 4), torch.ones(3, 4) * 2, torch.randn(5, 3)
        a, b = apply_swiglu_down_diag_curvature(g, u, w)
        torch.testing.assert_close(a, g * w.square().sum(0)[:, None])
        torch.testing.assert_close(b, 2 * a)

    def test_gelu_derivative(self):
        for mode in ('none', 'tanh'):
            x = torch.linspace(-3, 3, 21, requires_grad=True)
            actual, = torch.autograd.grad(torch.nn.functional.gelu(x, approximate=mode).sum(), x)
            torch.testing.assert_close(gelu_derivative(x.detach(), mode), actual)

    def test_bit_packing(self):
        # Independent scalar unpacking checks cross-byte widths, not the source
        # runtime decoder. Inputs deliberately reside on CPU.
        for bits in (1, 2, 3, 7, 8, 9, 11, 16):
            indices = torch.randint(0, 1 << bits, (79,))
            payload = bytes(pack_indices(indices, bits).tolist())
            integer = int.from_bytes(payload, 'little')
            recovered = [(integer >> (j * bits)) & ((1 << bits) - 1) for j in range(len(indices))]
            self.assertEqual(recovered, indices.tolist())
            self.assertEqual(len(payload), (len(indices) * bits + 7) // 8)

    def test_calibration_gradients_and_decode(self):
        _, _, packed = self.fitted()
        module = QuantLinear(packed, None, 'cpu')
        before = module.idx.clone()
        torch.testing.assert_close(module.weight_().float(), decode_fixed(packed, 'cpu'), rtol=0, atol=0)
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4, weight_decay=0)
        for _ in range(2):
            module._wc = None
            optimizer.zero_grad()
            module(torch.randn(2, 64, dtype=torch.bfloat16)).float().square().mean().backward()
            self.assertIsNotNone(module.codebook.grad)
            self.assertIsNotNone(module.scales.grad)
            self.assertTrue(torch.isfinite(module.codebook.grad).all())
            optimizer.step()
        torch.testing.assert_close(module.idx, before, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
