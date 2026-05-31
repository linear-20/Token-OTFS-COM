"""TX/RX physical contract tests: artifact schema, mask alignment,
shift equivalence, and PhysicalDDProfile validation."""

import hashlib
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

# TX imports.
TX_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TX_ROOT))
from transmitter import (
    PHYSICAL_CODEWORD_BOOK_SCHEMA_VERSION,
    LearnableTokenDDTransmitter,
    TransmitterConfig,
    TransmitterPilotMasks,
    build_transmitter_pilot_masks,
    export_physical_codeword_book,
    load_exported_codeword_book,
    load_exported_physical_dd_config,
    load_exported_physical_dd_artifact,
    save_raw_tx_state,
)
tx_export = importlib.import_module("transmitter.export")

# RX imports -- path-based due to hyphenated directory.
RX_ROOT = str(ROOT / "Learnable_Receiver_DD-Signals-to-Tokens")
sys.path.insert(0, RX_ROOT)
try:
    from receiver.token_prior import TokenCodewordPrior
    dd_ops = importlib.import_module("receiver.dd_ops")
    try:
        from receiver.config import ReceiverConfig
        from receiver.pilot_ce import (
            EmbeddedPilotConfig,
            build_embedded_pilot_masks,
        )
    except ImportError:
        ReceiverConfig = None
        EmbeddedPilotConfig = None
        build_embedded_pilot_masks = None
finally:
    if RX_ROOT in sys.path:
        sys.path.remove(RX_ROOT)


def _tx_config(**kw):
    v = dict(M=8, N=10, vocab_size=6, pilot_delay=4, pilot_doppler=5,
             pilot_guard_delay=3, pilot_guard_doppler=3,
             pilot_obs_delay_radius=1, pilot_obs_doppler_radius=1,
             max_channel_delay=2, max_channel_doppler=2,
             data_power=1.5, pilot_value_real=2.0, pilot_value_imag=-1.0,
             complex_dtype="complex64")
    v.update(kw)
    return TransmitterConfig(**v)


def _asymmetric_x(B=2, M=4, N=6):
    m = torch.arange(M, dtype=torch.float32).view(1, M, 1).expand(B, M, N)
    n = torch.arange(N, dtype=torch.float32).view(1, 1, N).expand(B, M, N)
    return (10.0 * m + n + 1j * (m - n)).to(torch.complex64)


# ==============================================================================
# A. Artifact schema tests
# ==============================================================================

class ArtifactSchemaTests(unittest.TestCase):

    def setUp(self):
        self.cfg = _tx_config()
        self.tx = LearnableTokenDDTransmitter(self.cfg)

    def test_export_includes_schema_version(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pt"
            payload = export_physical_codeword_book(self.tx, path)
            self.assertEqual(
                payload["metadata"]["schema_version"],
                PHYSICAL_CODEWORD_BOOK_SCHEMA_VERSION,
            )

    def test_export_includes_contract_fields(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pt"
            payload = export_physical_codeword_book(self.tx, path)
            meta = payload["metadata"]
            self.assertEqual(meta["artifact_type"], "physical_codeword_book")
            self.assertEqual(meta["dd_axis_order"], ["delay", "doppler"])
            self.assertEqual(meta["positive_shift_semantics"], "torch.roll")
            self.assertFalse(meta["pilot_wrap_around"])

    def test_export_includes_artifact_sha256(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pt"
            payload = export_physical_codeword_book(self.tx, path)
            sha = payload["metadata"].get("artifact_sha256")
            self.assertIsNotNone(sha)
            self.assertEqual(len(sha), 64)
            self.assertTrue(all(c in "0123456789abcdef" for c in sha))

    def test_export_rejects_nan_codeword(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        # Infer NaN by running forward then corrupting.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pt"
            with torch.no_grad():
                masks = build_transmitter_pilot_masks(cfg)
            # Force a NaN in raw params, then try export.
            tx.codebook.raw_real.data[0, 0, 0] = float("nan")
            with self.assertRaises((ValueError, RuntimeError)):
                export_physical_codeword_book(tx, path)


# ==============================================================================
# B. Strict loader tests
# ==============================================================================

class StrictLoaderTests(unittest.TestCase):

    def setUp(self):
        self.cfg = _tx_config()
        self.tx = LearnableTokenDDTransmitter(self.cfg)

    def _export_and_load(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda d=tmpdir: _rmtree(d))
        path = Path(tmpdir) / "export.pt"
        payload = export_physical_codeword_book(self.tx, path)
        loaded = load_exported_physical_dd_artifact(path)
        return payload, loaded

    def _resign_and_save(self, payload, path):
        payload["metadata"]["artifact_sha256"] = hashlib.sha256(
            tx_export._canonical_hash_bytes(
                payload["codeword_book"],
                payload["data_mask"],
                payload["metadata"],
            )
        ).hexdigest()
        torch.save(payload, str(path))

    def test_load_returns_full_payload(self):
        payload, loaded = self._export_and_load()
        self.assertIn("codeword_book", loaded)
        self.assertIn("data_mask", loaded)
        self.assertIn("metadata", loaded)

    def test_load_codeword_book_matches_export(self):
        payload, loaded = self._export_and_load()
        self.assertTrue(
            torch.equal(payload["codeword_book"],
                        loaded["codeword_book"]),
        )

    def test_load_data_mask_matches_export(self):
        payload, loaded = self._export_and_load()
        self.assertTrue(
            torch.equal(payload["data_mask"], loaded["data_mask"]),
        )

    def test_load_sha256_match(self):
        payload, loaded = self._export_and_load()
        self.assertEqual(
            payload["metadata"]["artifact_sha256"],
            loaded["metadata"]["artifact_sha256"],
        )

    def test_legacy_load_codeword_book_still_works(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pt"
            export_physical_codeword_book(self.tx, path)
            cw = load_exported_codeword_book(path)
            self.assertTrue(torch.is_complex(cw))
            self.assertEqual(cw.shape,
                             (self.cfg.vocab_size, self.cfg.M, self.cfg.N))

    def test_load_exported_config_matches_source(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pt"
            export_physical_codeword_book(self.tx, path)
            loaded_cfg = load_exported_physical_dd_config(path)
        self.assertEqual(loaded_cfg, self.cfg)

    def test_rejects_schema_version_missing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pt"
            payload = export_physical_codeword_book(self.tx, path)
            del payload["metadata"]["schema_version"]
            torch.save(payload, str(path))
            with self.assertRaises(ValueError):
                load_exported_physical_dd_artifact(path)

    def test_rejects_no_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pt"
            payload = export_physical_codeword_book(
                self.tx, path, include_metadata=False,
            )
            with self.assertRaises(ValueError):
                load_exported_physical_dd_artifact(path)

    def test_rejects_wrong_sha256(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pt"
            payload = export_physical_codeword_book(self.tx, path)
            payload["metadata"]["artifact_sha256"] = "0" * 64
            torch.save(payload, str(path))
            with self.assertRaises(ValueError):
                load_exported_physical_dd_artifact(path)

    def test_rejects_contract_field_tampering_even_when_resigned(self):
        mutations = (
            ("dd_axis_order", ["doppler", "delay"]),
            ("positive_shift_semantics", "numpy.roll"),
            ("pilot_wrap_around", True),
        )
        for key, value in mutations:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as d:
                path = Path(d) / "test.pt"
                payload = export_physical_codeword_book(self.tx, path)
                payload["metadata"][key] = value
                self._resign_and_save(payload, path)
                with self.assertRaises(ValueError):
                    load_exported_physical_dd_artifact(path)

    def test_rejects_fractional_mask_without_float32_rounding(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pt"
            payload = export_physical_codeword_book(self.tx, path)
            data_mask = payload["data_mask"].double()
            index = tuple(data_mask.nonzero(as_tuple=False)[0].tolist())
            data_mask[index] = 1.0 + 2.0 ** -30
            payload["data_mask"] = data_mask
            self._resign_and_save(payload, path)
            with self.assertRaises(ValueError):
                load_exported_physical_dd_artifact(path)

    def test_rejects_missing_required_config_field(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pt"
            payload = export_physical_codeword_book(self.tx, path)
            del payload["metadata"]["pilot_guard_delay"]
            self._resign_and_save(payload, path)
            with self.assertRaises(ValueError):
                load_exported_physical_dd_artifact(path)


# ==============================================================================
# C. RX prior boundary tests
# ==============================================================================

class RXPriorBoundaryTests(unittest.TestCase):

    def _export_and_load_cw(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda d=tmpdir: _rmtree(d))
        path = Path(tmpdir) / "export.pt"
        export_physical_codeword_book(tx, path)
        return load_exported_codeword_book(path)

    def test_strict_loaded_codeword_accepted_by_prior(self):
        cw = self._export_and_load_cw()
        prior = TokenCodewordPrior(codeword_book=cw)
        self.assertIsInstance(prior, TokenCodewordPrior)

    def test_prior_rejects_bool_prior_strength(self):
        cw = self._export_and_load_cw()
        with self.assertRaises(TypeError):
            TokenCodewordPrior(codeword_book=cw, prior_strength=True)

    def test_prior_rejects_negative_prior_strength(self):
        cw = self._export_and_load_cw()
        with self.assertRaises(ValueError):
            TokenCodewordPrior(codeword_book=cw, prior_strength=-1.0)

    def test_prior_rejects_nonfinite_prior_strength(self):
        cw = self._export_and_load_cw()
        with self.assertRaises(ValueError):
            TokenCodewordPrior(codeword_book=cw,
                               prior_strength=float("inf"))

    def test_prior_rejects_bool_temperature(self):
        cw = self._export_and_load_cw()
        with self.assertRaises(TypeError):
            TokenCodewordPrior(codeword_book=cw, temperature=True)

    def test_prior_rejects_zero_temperature(self):
        cw = self._export_and_load_cw()
        with self.assertRaises(ValueError):
            TokenCodewordPrior(codeword_book=cw, temperature=0.0)

    def test_prior_rejects_nonfinite_temperature(self):
        cw = self._export_and_load_cw()
        with self.assertRaises(ValueError):
            TokenCodewordPrior(codeword_book=cw,
                               temperature=float("nan"))

    def test_prior_rejects_non_str_similarity(self):
        cw = self._export_and_load_cw()
        with self.assertRaises(TypeError):
            TokenCodewordPrior(codeword_book=cw, similarity=1)

    def test_prior_rejects_invalid_codeword_nan(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with torch.no_grad():
            cw = tx.codebook.forward(
                data_mask=build_transmitter_pilot_masks(cfg).data_mask,
            )
        cw[0, 0, 0] = complex(float("nan"), 0.0)
        with self.assertRaises(ValueError):
            TokenCodewordPrior(codeword_book=cw)

    def test_prior_rejects_empty_dims(self):
        cw = torch.empty(0, 4, 4, dtype=torch.complex64)
        with self.assertRaises(ValueError):
            TokenCodewordPrior(codeword_book=cw)


# ==============================================================================
# D. Mask alignment tests
# ==============================================================================

class MaskAlignmentTests(unittest.TestCase):

    def test_tx_masks_match_rx_masks_rebuilt_from_metadata(self):
        cfg = _tx_config()
        masks = build_transmitter_pilot_masks(cfg)

        # Export via codebook, then load metadata.
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pt"
            payload = export_physical_codeword_book(tx, path)
        meta = payload["metadata"]

        # Rebuild RX masks from the exported TX physical contract.
        rx_cfg = EmbeddedPilotConfig(
            pilot_delay=meta["pilot_delay"],
            pilot_doppler=meta["pilot_doppler"],
            guard_delay=meta["pilot_guard_delay"],
            guard_doppler=meta["pilot_guard_doppler"],
            obs_delay_radius=meta["pilot_obs_delay_radius"],
            obs_doppler_radius=meta["pilot_obs_doppler_radius"],
            pilot_value=complex(
                meta["pilot_value_real"], meta["pilot_value_imag"],
            ),
            wrap_around=meta["pilot_wrap_around"],
        )
        rx_masks = build_embedded_pilot_masks(meta["M"], meta["N"], rx_cfg)

        # Masks must match exactly.
        self.assertEqual(rx_cfg.pilot_value, cfg.pilot_value)
        self.assertTrue(torch.equal(masks.pilot_mask[0], rx_masks.pilot_mask))
        self.assertTrue(torch.equal(masks.guard_mask[0], rx_masks.guard_mask))
        self.assertTrue(torch.equal(masks.data_mask[0], rx_masks.data_mask))

    def test_contract_fields_are_fixed_values(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pt"
            payload = export_physical_codeword_book(tx, path)
        meta = payload["metadata"]
        self.assertEqual(meta["dd_axis_order"], ["delay", "doppler"])
        self.assertEqual(meta["positive_shift_semantics"], "torch.roll")
        self.assertFalse(meta["pilot_wrap_around"])


# ==============================================================================
# E. TX/RX operator equivalence tests
# ==============================================================================

class TXRXOperatorEquivalenceTests(unittest.TestCase):

    def test_sparse_operator_matches_rx_convolve(self):
        """TX apply_sparse_multipath_dd_operator == RX convolve."""
        from transmitter import (
            SparseMultipathDDChannel,
            apply_sparse_multipath_dd_operator,
        )
        x = _asymmetric_x(B=2)
        shifts = torch.tensor([[[2, 0], [0, -1]]], dtype=torch.long
                              ).expand(2, -1, -1)
        gains = torch.tensor([[1.0 + 0j, -0.5 + 0.5j]],
                             dtype=torch.complex64).expand(2, -1)
        ch = SparseMultipathDDChannel(
            path_shifts=shifts, path_gains=gains,
        )
        y_tx = apply_sparse_multipath_dd_operator(x, ch)
        y_rx = dd_ops.dd_circular_convolve_sparse(
            x, path_indices=shifts, path_gains=gains,
        )
        self.assertTrue(torch.allclose(y_tx, y_rx, atol=1e-5))

    def test_contract_no_cp_len_in_artifact(self):
        """PhysicalDDProfile must NOT include cp_len."""
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.pt"
            payload = export_physical_codeword_book(tx, path)
        self.assertNotIn("cp_len", payload["metadata"])


# ==============================================================================
# F. Quality tests
# ==============================================================================

class QualityTests(unittest.TestCase):

    def test_export_py_ascii_only(self):
        path = TX_ROOT / "transmitter" / "export.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_new_loaders_docstrings_exist(self):
        self.assertIsNotNone(load_exported_physical_dd_artifact.__doc__)
        self.assertTrue(
            len(load_exported_physical_dd_artifact.__doc__.strip()) > 0,
        )

    def test_schema_version_is_int_one(self):
        self.assertEqual(PHYSICAL_CODEWORD_BOOK_SCHEMA_VERSION, 1)
        self.assertIsInstance(PHYSICAL_CODEWORD_BOOK_SCHEMA_VERSION, int)


def _rmtree(path: str) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
