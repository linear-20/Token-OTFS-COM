"""Tests for masked normalized DD correlation and shift-orbit correlation.

Covers masked_normalized_dd_correlation, masked_shift_orbit_correlation,
integration with TokenDDCodebook, autograd, and quality checks.
"""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

from transmitter import (
    TokenDDCodebook,
    TransmitterConfig,
    build_transmitter_pilot_masks,
)
from transmitter.ablations import (
    dd_circular_shift,
    dd_circular_shift_bank,
    masked_normalized_dd_correlation,
    masked_shift_orbit_correlation,
)


# -- helpers -------------------------------------------------------------------

def _complex_randn(*shape, dtype=torch.complex64):
    rd = {torch.complex64: torch.float32, torch.complex128: torch.float64}[dtype]
    return torch.randn(*shape, dtype=rd) + 1j * torch.randn(*shape, dtype=rd)


def _real_dtype(cdtype):
    return {torch.complex64: torch.float32, torch.complex128: torch.float64}[cdtype]


def _tx_config(**kwargs):
    values = dict(
        M=8, N=10, vocab_size=6,
        pilot_delay=4, pilot_doppler=5,
        pilot_guard_delay=3, pilot_guard_doppler=3,
        pilot_obs_delay_radius=1, pilot_obs_doppler_radius=1,
        max_channel_delay=2, max_channel_doppler=2,
        data_power=1.5,
        pilot_value_real=2.0, pilot_value_imag=-1.0,
        complex_dtype="complex64",
    )
    values.update(kwargs)
    return TransmitterConfig(**values)


# ============================================================================
# A. masked_normalized_dd_correlation legal paths
# ============================================================================

class MaskedCorrLegalTests(unittest.TestCase):

    def setUp(self):
        self.ref = _complex_randn(3, 8, 10)
        self.mask = torch.ones(1, 8, 10)

    def test_identical_tensors_score_is_one(self):
        score = masked_normalized_dd_correlation(self.ref, self.ref, self.mask)
        self.assertTrue(torch.allclose(
            score, torch.ones(3, dtype=torch.float32), atol=1e-5,
        ))

    def test_global_phase_rotation_score_is_one(self):
        phase = torch.tensor(1.2, dtype=torch.float32)
        rot = torch.complex(torch.cos(phase), torch.sin(phase))
        cand = self.ref * rot
        score = masked_normalized_dd_correlation(self.ref, cand, self.mask)
        self.assertTrue(torch.allclose(
            score, torch.ones(3, dtype=torch.float32), atol=1e-5,
        ))

    def test_orthogonal_one_hot_score_is_zero(self):
        ref = torch.zeros(1, 4, 4, dtype=torch.complex64)
        cand = torch.zeros(1, 4, 4, dtype=torch.complex64)
        ref[0, 0, 0] = 1.0 + 0j
        cand[0, 0, 1] = 1.0 + 0j
        mask = torch.ones(1, 4, 4)
        score = masked_normalized_dd_correlation(ref, cand, mask)
        self.assertAlmostEqual(score[0].item(), 0.0, places=5)

    def test_manual_complex_inner_product_matches(self):
        ref = _complex_randn(2, 4, 4)
        cand = _complex_randn(2, 4, 4)
        mask = torch.ones(1, 4, 4)
        score = masked_normalized_dd_correlation(ref, cand, mask)
        for b in range(2):
            r = ref[b]
            c = cand[b]
            num = abs((torch.conj(r) * c).sum()).item()
            denom = (r.abs().pow(2).sum().sqrt() * c.abs().pow(2).sum().sqrt()).item()
            expected = num / denom
            self.assertAlmostEqual(score[b].item(), expected, places=5)

    def test_conj_ref_times_cand_distinguishes_wrong_formula(self):
        """abs(sum(conj(ref)*cand)) != abs(sum(ref*cand)) for this pair.

        ref  = [1+0j, 0+1j]   (a=1, a'=i)
        cand = [1+0j, 0-1j]   (b=1, b'=-i)

        Correct conj formula:
            conj(1)*(1) + conj(i)*(-i) = 1 + (-i)*(-i) = 1-1 = 0 -> |0| = 0
        Wrong non-conj formula:
            1*1 + i*(-i) = 1 + 1 = 2 -> |2| = 2
        Expected score: 0 (orthogonal).  Wrong score: 2/(sqrt(2)*sqrt(2)) = 1.
        """
        ref = torch.tensor([1.0 + 0j, 0.0 + 1j, 0.0 + 0j, 0.0 + 0j],
                           dtype=torch.complex64).reshape(1, 1, 4)
        cand = torch.tensor([1.0 + 0j, 0.0 - 1j, 0.0 + 0j, 0.0 + 0j],
                            dtype=torch.complex64).reshape(1, 1, 4)
        mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])

        score = masked_normalized_dd_correlation(ref, cand, mask)
        expected_correct = 0.0

        # What a wrong implementation without conj would produce:
        r = (ref * mask).reshape(-1)
        c = (cand * mask).reshape(-1)
        wrong_without_conj = abs((r * c).sum()).item() / (
            r.abs().pow(2).sum().sqrt().item()
            * c.abs().pow(2).sum().sqrt().item()
        )

        self.assertAlmostEqual(score[0].item(), expected_correct, places=5,
                               msg="correct conj formula gives 0")
        self.assertGreater(wrong_without_conj, 0.99,
                           msg="wrong formula without conj gives ~1")
        self.assertNotAlmostEqual(score[0].item(), wrong_without_conj, places=3)

    def test_mask_1MN_broadcasts(self):
        mask = torch.ones(1, 8, 10)
        score = masked_normalized_dd_correlation(self.ref, self.ref, mask)
        self.assertEqual(score.shape, (3,))

    def test_mask_BMN_works_independently(self):
        mask = torch.ones(3, 8, 10)
        score = masked_normalized_dd_correlation(self.ref, self.ref, mask)
        self.assertEqual(score.shape, (3,))

    def test_masked_out_bins_excluded_from_correlation(self):
        """Non-uniform mask: excluded bins hold strong overlap but
        active bins cancel perfectly -> masked score ~0 while
        full-mask score > 0.

        ref  = [1, 1, 3] (bins 0,1,2)
        cand = [1, -1, 3]
        Bins 0+1: conj(1)*1 + conj(1)*(-1) = 1 - 1 = 0 -> cancel.
        Bin 2:    conj(3)*3 = 9 -> strong overlap.
        Partial mask (bins 0+1): score = 0.
        Full mask (bins 0+1+2): score = 9/11 > 0.5.
        """
        M, N = 1, 3
        ref = torch.zeros(1, M, N, dtype=torch.complex64)
        cand = torch.zeros(1, M, N, dtype=torch.complex64)
        ref[0, 0, 0] = 1.0 + 0j
        ref[0, 0, 1] = 1.0 + 0j
        ref[0, 0, 2] = 3.0 + 0j
        cand[0, 0, 0] = 1.0 + 0j
        cand[0, 0, 1] = -1.0 + 0j
        cand[0, 0, 2] = 3.0 + 0j

        # Full mask: bins 0,1 cancel, bin 2 contributes -> score > 0.
        mask_full = torch.ones(1, M, N)
        s_full = masked_normalized_dd_correlation(ref, cand, mask_full)
        self.assertGreater(s_full[0].item(), 0.7,
                           "full mask includes bin 2 overlap")

        # Partial mask: only bins 0,1 active -> perfect cancellation.
        mask_partial = torch.zeros(1, M, N)
        mask_partial[0, 0, 0] = 1.0
        mask_partial[0, 0, 1] = 1.0
        s_partial = masked_normalized_dd_correlation(ref, cand, mask_partial)
        self.assertAlmostEqual(s_partial[0].item(), 0.0, places=5,
                               msg="excluded bin 2 must not contribute")

    def test_output_shape_B(self):
        score = masked_normalized_dd_correlation(self.ref, self.ref, self.mask)
        self.assertEqual(score.shape, (3,))

    def test_complex64_output_float32(self):
        score = masked_normalized_dd_correlation(self.ref, self.ref, self.mask)
        self.assertEqual(score.dtype, torch.float32)

    def test_complex128_output_float64(self):
        ref = _complex_randn(2, 8, 10, dtype=torch.complex128)
        score = masked_normalized_dd_correlation(ref, ref, torch.ones(1, 8, 10))
        self.assertEqual(score.dtype, torch.float64)

    def test_device_preserved(self):
        score = masked_normalized_dd_correlation(self.ref, self.ref, self.mask)
        self.assertEqual(score.device, self.ref.device)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_mask_cuda_dd_tensors(self):
        ref = _complex_randn(2, 8, 10).to("cuda")
        mask = torch.ones(1, 8, 10)
        score = masked_normalized_dd_correlation(ref, ref, mask)
        self.assertTrue(score.is_cuda)

    def test_scores_within_tolerance_of_0_1(self):
        ref = _complex_randn(3, 8, 10)
        cand = _complex_randn(3, 8, 10)
        mask = torch.ones(1, 8, 10)
        score = masked_normalized_dd_correlation(ref, cand, mask)
        self.assertTrue((score >= -1e-5).all(),
                        f"score below 0: {score.min().item()}")
        self.assertTrue((score <= 1.0 + 1e-5).all(),
                        f"score above 1: {score.max().item()}")


# ============================================================================
# B. masked_normalized_dd_correlation reject paths
# ============================================================================

class MaskedCorrRejectTests(unittest.TestCase):

    def setUp(self):
        self.ref = _complex_randn(3, 8, 10)
        self.cand = _complex_randn(3, 8, 10)
        self.mask = torch.ones(1, 8, 10)

    def test_ref_non_tensor_raises_type_error(self):
        with self.assertRaises(TypeError):
            masked_normalized_dd_correlation([1.0], self.cand, self.mask)

    def test_cand_non_tensor_raises_type_error(self):
        with self.assertRaises(TypeError):
            masked_normalized_dd_correlation(self.ref, [1.0], self.mask)

    def test_ref_non_complex_raises_type_error(self):
        with self.assertRaises(TypeError):
            masked_normalized_dd_correlation(torch.randn(3, 8, 10), self.cand, self.mask)

    def test_cand_non_complex_raises_type_error(self):
        with self.assertRaises(TypeError):
            masked_normalized_dd_correlation(self.ref, torch.randn(3, 8, 10), self.mask)

    def test_wrong_ndim_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(
                _complex_randn(8, 10), self.cand, self.mask,
            )

    def test_B_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(
                _complex_randn(0, 8, 10), _complex_randn(0, 8, 10), self.mask,
            )

    def test_M_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(
                _complex_randn(1, 0, 10), _complex_randn(1, 0, 10),
                torch.ones(1, 0, 10),
            )

    def test_N_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(
                _complex_randn(1, 8, 0), _complex_randn(1, 8, 0),
                torch.ones(1, 8, 0),
            )

    def test_shape_mismatch_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(self.ref, _complex_randn(3, 4, 10), self.mask)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_device_mismatch_raises_value_error(self):
        cand_gpu = _complex_randn(3, 8, 10).to("cuda")
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(self.ref, cand_gpu, self.mask)

    def test_dtype_mismatch_raises_type_error(self):
        cand_diff = _complex_randn(3, 8, 10, dtype=torch.complex128)
        with self.assertRaises(TypeError):
            masked_normalized_dd_correlation(self.ref, cand_diff, self.mask)

    def test_ref_nonfinite_raises_value_error(self):
        ref_bad = self.ref.clone()
        ref_bad[0, 0, 0] = complex(float("nan"), 0.0)
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(ref_bad, self.cand, self.mask)

    def test_cand_nonfinite_raises_value_error(self):
        cand_bad = self.cand.clone()
        cand_bad[0, 0, 0] = complex(0.0, float("inf"))
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(self.ref, cand_bad, self.mask)

    def test_mask_non_tensor_raises_type_error(self):
        with self.assertRaises(TypeError):
            masked_normalized_dd_correlation(self.ref, self.cand, [1.0])

    def test_mask_complex_raises_type_error(self):
        with self.assertRaises(TypeError):
            masked_normalized_dd_correlation(
                self.ref, self.cand,
                torch.ones(1, 8, 10, dtype=torch.complex64),
            )

    def test_mask_wrong_ndim_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(self.ref, self.cand, torch.ones(8, 10))

    def test_mask_wrong_shape_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(self.ref, self.cand,
                                              torch.ones(1, 4, 10))

    def test_mask_nonfinite_raises_value_error(self):
        m = torch.ones(1, 8, 10)
        m[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(self.ref, self.cand, m)

    def test_mask_fractional_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(
                self.ref, self.cand, torch.full((1, 8, 10), 0.5),
            )

    def test_mask_contains_0_5_raises_value_error(self):
        m = torch.ones(1, 8, 10)
        m[0, 0, 0] = 0.5
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(self.ref, self.cand, m)

    def test_mask_contains_0_999_raises_value_error(self):
        m = torch.ones(1, 8, 10)
        m[0, 0, 0] = 0.999
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(self.ref, self.cand, m)

    def test_empty_active_mask_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(
                self.ref, self.cand, torch.zeros(1, 8, 10),
            )

    def test_zero_masked_ref_energy_raises_value_error(self):
        mask = torch.ones(1, 8, 10)
        mask[0, :, :] = 0.0
        mask[0, 0, 0] = 1.0
        ref = torch.zeros(3, 8, 10, dtype=torch.complex64)
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(ref, self.cand, mask)

    def test_zero_masked_cand_energy_raises_value_error(self):
        mask = torch.ones(1, 8, 10)
        mask[0, :, :] = 0.0
        mask[0, 0, 0] = 1.0
        cand = torch.zeros(3, 8, 10, dtype=torch.complex64)
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(self.ref, cand, mask)


# ============================================================================
# C. masked_shift_orbit_correlation legal paths
# ============================================================================

class ShiftOrbitCorrLegalTests(unittest.TestCase):

    def setUp(self):
        self.ref = _complex_randn(2, 8, 10)
        self.cand = _complex_randn(2, 8, 10)
        self.shifts = torch.tensor([[0, 0], [1, 2], [-1, 3]], dtype=torch.long)
        self.mask = torch.ones(1, 8, 10)

    def test_returns_shape_B_S(self):
        score = masked_shift_orbit_correlation(
            self.ref, self.cand, self.shifts, self.mask,
        )
        self.assertEqual(score.shape, (2, 3))

    def test_each_column_matches_single_shift_correlation(self):
        score = masked_shift_orbit_correlation(
            self.ref, self.cand, self.shifts, self.mask,
        )
        for s in range(self.shifts.shape[0]):
            d = int(self.shifts[s, 0].item())
            v = int(self.shifts[s, 1].item())
            shifted = dd_circular_shift(self.cand, d, v)
            expected = masked_normalized_dd_correlation(
                self.ref, shifted, self.mask,
            )
            self.assertTrue(torch.allclose(score[:, s], expected, atol=1e-5),
                            f"mismatch at s={s} shift=({d},{v})")

    def test_zero_shift_included(self):
        shifts = torch.tensor([[0, 0], [3, 0]], dtype=torch.long)
        score = masked_shift_orbit_correlation(
            self.ref, self.ref, shifts, self.mask,
        )
        self.assertTrue(torch.allclose(score[:, 0], torch.ones(2), atol=1e-5))

    def test_shift_order_preserved(self):
        s1 = torch.tensor([[0, 0], [1, 0]], dtype=torch.long)
        s2 = torch.tensor([[1, 0], [0, 0]], dtype=torch.long)
        sc1 = masked_shift_orbit_correlation(self.ref, self.cand, s1, self.mask)
        sc2 = masked_shift_orbit_correlation(self.ref, self.cand, s2, self.mask)
        self.assertTrue(torch.equal(sc1[:, 0], sc2[:, 1]))
        self.assertTrue(torch.equal(sc1[:, 1], sc2[:, 0]))

    def test_duplicate_shifts_kept(self):
        shifts = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)
        score = masked_shift_orbit_correlation(
            self.ref, self.cand, shifts, self.mask,
        )
        self.assertTrue(torch.allclose(score[:, 0], score[:, 1], atol=1e-5))

    def test_evidence_mask_applied_after_shift(self):
        """Mask-after-shift vs mask-before-shift: non-ones mask with
        excluded origin distinguishes the two orders.

        mask: (0,0)=0 (excluded), (0,1)=1 (active).
        cand has energy at (0,0) - excluded by mask.
        ref  has energy at (0,1) - active.
        Shift (0,1) moves cand from (0,0) to (0,1).

        Correct (mask AFTER shift):  shifted cand at (0,1) is in
            active region -> overlaps ref -> score ~1.0.
        Wrong   (mask BEFORE shift): mask zeros cand at (0,0), then
            shift moves zero to (0,1) -> zero energy -> ValueError.
        """
        ref = torch.zeros(1, 4, 4, dtype=torch.complex64)
        ref[0, 0, 1] = 1.0 + 0j            # active position
        cand = torch.zeros(1, 4, 4, dtype=torch.complex64)
        cand[0, 0, 0] = 1.0 + 0j            # initially in excluded position

        mask = torch.zeros(1, 4, 4)          # non-uniform
        mask[0, 0, 1] = 1.0                  # only (0,1) is active

        shifts = torch.tensor([[0, 1]], dtype=torch.long)

        score = masked_shift_orbit_correlation(ref, cand, shifts, mask)
        self.assertGreater(score[0, 0].item(), 0.99,
                           "mask-after-shift: shifted cand at (0,1) overlaps ref")

    def test_no_implicit_mask_projection_on_shifted(self):
        """Shift can move energy into a bin that the evidence mask would
        later evaluate."""
        cand = torch.zeros(1, 4, 4, dtype=torch.complex64)
        cand[0, 0, 0] = 1.0 + 0j
        ref = torch.zeros(1, 4, 4, dtype=torch.complex64)
        ref[0, 1, 0] = 1.0 + 0j  # shift by (1,0) moves cand here
        mask = torch.ones(1, 4, 4)
        shifts = torch.tensor([[1, 0]], dtype=torch.long)
        score = masked_shift_orbit_correlation(ref, cand, shifts, mask)
        self.assertGreater(score[0, 0].item(), 0.99)

    def test_scores_within_tolerance_of_0_1(self):
        score = masked_shift_orbit_correlation(
            self.ref, self.cand, self.shifts, self.mask,
        )
        self.assertTrue((score >= -1e-5).all(),
                        f"score below 0: {score.min().item()}")
        self.assertTrue((score <= 1.0 + 1e-5).all(),
                        f"score above 1: {score.max().item()}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_shifts_cuda_dd_tensors(self):
        ref = _complex_randn(2, 8, 10).to("cuda")
        cand = _complex_randn(2, 8, 10).to("cuda")
        mask = torch.ones(1, 8, 10)
        shifts = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        score = masked_shift_orbit_correlation(ref, cand, shifts, mask)
        self.assertTrue(score.is_cuda)

    def test_rejects_invalid_shifts_through_bank_contract(self):
        with self.assertRaises(TypeError):
            masked_shift_orbit_correlation(
                self.ref, self.cand,
                torch.tensor([[0.5, 1.0]]),  # wrong dtype
                self.mask,
            )

    def test_zero_masked_energy_after_shift_raises_value_error(self):
        ref = torch.zeros(2, 8, 10, dtype=torch.complex64)
        cand = torch.zeros(2, 8, 10, dtype=torch.complex64)
        mask = torch.ones(1, 8, 10)
        shifts = torch.tensor([[0, 0]], dtype=torch.long)
        with self.assertRaises(ValueError):
            masked_shift_orbit_correlation(ref, cand, shifts, mask)

    def test_zero_candidate_energy_after_specific_shift(self):
        """One shift is fine; another shift moves cand entirely into
        a masked-out region, triggering ValueError."""
        mask = torch.ones(1, 1, 4)
        mask[0, 0, 1] = 0.0  # bin (0,1) excluded

        ref = torch.zeros(1, 1, 4, dtype=torch.complex64)
        ref[0, 0, 0] = 1.0 + 0j  # always active

        cand = torch.zeros(1, 1, 4, dtype=torch.complex64)
        cand[0, 0, 0] = 1.0 + 0j  # at active bin

        # shift [0,0]: cand stays at (0,0)=active -> OK
        # shift [0,1]: cand moves to (0,1)=blocked -> zero energy
        shifts = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
        with self.assertRaises(ValueError):
            masked_shift_orbit_correlation(ref, cand, shifts, mask)


# ============================================================================
# D. Integration with TokenDDCodebook
# ============================================================================

class IntegrationTests(unittest.TestCase):

    def test_integration_with_codebook_and_data_mask(self):
        cfg = _tx_config()
        cb = TokenDDCodebook(cfg)
        masks = build_transmitter_pilot_masks(cfg)
        cw = cb.forward(data_mask=masks.data_mask)  # [V, M, N]
        evidence_mask = masks.data_mask  # [1, M, N]

        # Select token pair.
        ref = cw[[0, 1, 2]]   # [P, M, N]
        cand = cw[[3, 4, 5]]  # [P, M, N]

        shifts = torch.tensor([[0, 0], [1, 0], [0, 1], [-2, 3]], dtype=torch.long)

        scores = masked_shift_orbit_correlation(
            ref, cand, shifts, evidence_mask,
        )
        self.assertEqual(scores.shape, (3, 4))
        self.assertTrue(torch.isfinite(scores).all())
        # Scores within [-tol, 1+tol].
        self.assertTrue((scores >= -1e-5).all())
        self.assertTrue((scores <= 1.0 + 1e-5).all())

    def test_does_not_compute_full_V2_matrix(self):
        """We use P=3 pairs from V=6 tokens, not all 6*6=36 pairs."""
        cfg = _tx_config()
        cb = TokenDDCodebook(cfg)
        masks = build_transmitter_pilot_masks(cfg)
        cw = cb.forward(data_mask=masks.data_mask)
        ref = cw[[0, 1, 2]]
        cand = cw[[3, 4, 5]]
        shifts = torch.tensor([[0, 0]], dtype=torch.long)
        scores = masked_shift_orbit_correlation(
            ref, cand, shifts, masks.data_mask,
        )
        # Only P=3 rows, not V^2.
        self.assertEqual(scores.shape[0], 3)
        self.assertEqual(scores.shape[1], 1)


# ============================================================================
# E. Autograd tests
# ============================================================================

class AutogradTests(unittest.TestCase):

    def test_shift_orbit_correlation_preserves_grad(self):
        ref = _complex_randn(2, 8, 10)
        cand = _complex_randn(2, 8, 10)
        ref.requires_grad_(True)
        cand.requires_grad_(True)
        shifts = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        mask = torch.ones(1, 8, 10)

        score = masked_shift_orbit_correlation(ref, cand, shifts, mask)
        loss = score.sum()
        loss.backward()

        self.assertIsNotNone(ref.grad)
        self.assertIsNotNone(cand.grad)
        self.assertTrue(torch.isfinite(ref.grad).all())
        self.assertTrue(torch.isfinite(cand.grad).all())


# ============================================================================
# F. Quality tests
# ============================================================================

class QualityTests(unittest.TestCase):

    def test_orbit_correlation_py_is_ascii_only(self):
        path = TRANSMITTER_ROOT / "transmitter" / "orbit_correlation.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_corr_docstring_exists(self):
        doc = masked_normalized_dd_correlation.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(len(doc.strip()) > 0)

    def test_shift_orbit_docstring_exists(self):
        doc = masked_shift_orbit_correlation.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(len(doc.strip()) > 0)

    def test_corr_docstring_mentions_shapes(self):
        doc = masked_normalized_dd_correlation.__doc__
        self.assertIn("[B, M, N]", doc)
        self.assertIn("hard binary", doc)

    def test_shift_orbit_docstring_mentions_shapes(self):
        doc = masked_shift_orbit_correlation.__doc__
        self.assertIn("[B, S", doc)
        self.assertIn("hard binary", doc)

    def test_shift_orbit_docstring_mask_after_shift(self):
        doc = masked_shift_orbit_correlation.__doc__
        self.assertTrue(
            "mask-after-shift" in doc.lower()
            or "after shifting" in doc.lower(),
            "docstring should mention mask-after-shift semantics",
        )

    def test_exports_importable(self):
        from transmitter.ablations import (
            masked_normalized_dd_correlation as _mnc,
            masked_shift_orbit_correlation as _msoc,
        )
        self.assertTrue(callable(_mnc))
        self.assertTrue(callable(_msoc))


if __name__ == "__main__":
    unittest.main()
