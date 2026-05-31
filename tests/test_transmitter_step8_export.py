import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

from transmitter import (
    LearnableTokenDDTransmitter,
    TokenDDCodebook,
    TransmitterConfig,
    build_transmitter_pilot_masks,
    export_physical_codeword_book,
    load_exported_codeword_book,
    save_raw_tx_state,
)


def _tx_config(**kwargs) -> TransmitterConfig:
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


class ExportPhysicalCodewordBookTests(unittest.TestCase):
    """Tests for export_physical_codeword_book()."""

    def test_with_learnable_tx_creates_file(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "exported_codeword_book.pt"
            payload = export_physical_codeword_book(tx, path)
            self.assertTrue(path.exists())
            self.assertIn("codeword_book", payload)
            self.assertIn("data_mask", payload)

    def test_with_codebook_creates_file(self):
        cfg = _tx_config()
        cb = TokenDDCodebook(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cb_codeword_book.pt"
            payload = export_physical_codeword_book(cb, path)
            self.assertTrue(path.exists())
            self.assertIn("codeword_book", payload)

    def test_invalid_transmitter_type_raises_type_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pt"
            with self.assertRaises(TypeError):
                export_physical_codeword_book("not_a_tx", path)  # type: ignore

    def test_payload_has_metadata_when_include_metadata_true(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "with_meta.pt"
            payload = export_physical_codeword_book(tx, path, include_metadata=True)
            self.assertIn("metadata", payload)

    def test_payload_no_metadata_when_include_metadata_false(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "no_meta.pt"
            payload = export_physical_codeword_book(tx, path, include_metadata=False)
            self.assertNotIn("metadata", payload)

    def test_saved_file_has_same_keys(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = export_physical_codeword_book(tx, path)
            loaded = torch.load(str(path))
            for key in ("codeword_book", "data_mask", "metadata"):
                self.assertIn(key, loaded)

    def test_exported_codeword_book_is_complex_V_M_N(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = export_physical_codeword_book(tx, path)
            cw = payload["codeword_book"]
            self.assertTrue(torch.is_complex(cw))
            self.assertEqual(cw.shape, (cfg.vocab_size, cfg.M, cfg.N))

    def test_exported_codeword_book_is_cpu(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = export_physical_codeword_book(tx, path)
            self.assertEqual(payload["codeword_book"].device.type, "cpu")

    def test_exported_data_mask_is_bool_cpu(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = export_physical_codeword_book(tx, path)
            dm = payload["data_mask"]
            self.assertEqual(dm.dtype, torch.bool)
            self.assertEqual(dm.device.type, "cpu")

    def test_pilot_guard_are_zero_when_data_mask_omitted(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        masks = build_transmitter_pilot_masks(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = export_physical_codeword_book(tx, path)
            cw = payload["codeword_book"]
            for v in range(cfg.vocab_size):
                self.assertTrue(torch.all(cw[v, masks.pilot_mask[0]].abs() < 1e-5))
                self.assertTrue(torch.all(cw[v, masks.guard_mask[0]].abs() < 1e-5))

    def test_data_region_power_approx_data_power(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        masks = build_transmitter_pilot_masks(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = export_physical_codeword_book(tx, path)
            cw = payload["codeword_book"]
            data_2d = payload["data_mask"][0]
            for v in range(cfg.vocab_size):
                avg = cw[v, data_2d].abs().pow(2).mean()
                self.assertTrue(
                    torch.allclose(avg, torch.tensor(cfg.data_power), atol=0.01),
                )

    def test_metadata_flags_correct(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = export_physical_codeword_book(tx, path)
            meta = payload["metadata"]
            self.assertEqual(meta["artifact_type"], "physical_codeword_book")
            self.assertTrue(meta["receiver_prior_ready"])
            self.assertTrue(meta["physical_codeword_book_exported"])
            self.assertFalse(meta["raw_tx_state_dict"])

    def test_metadata_includes_config_fields(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = export_physical_codeword_book(tx, path)
            meta = payload["metadata"]
            for key in ("M", "N", "vocab_size", "data_power", "complex_dtype",
                        "pilot_delay", "pilot_doppler"):
                self.assertIn(key, meta)

    def test_explicit_bool_data_mask_works(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        dm = build_transmitter_pilot_masks(cfg).data_mask
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = export_physical_codeword_book(tx, path, data_mask=dm)
            self.assertTrue(torch.equal(payload["data_mask"], dm.cpu()))

    def test_explicit_real_binary_mask_works(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        # Use config data_mask as float -- valid subset of config data region.
        dm = build_transmitter_pilot_masks(cfg).data_mask.float()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = export_physical_codeword_book(tx, path, data_mask=dm)
            self.assertEqual(payload["data_mask"].dtype, torch.bool)

    def test_fractional_data_mask_raises(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        dm = torch.full((1, cfg.M, cfg.N), 0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            with self.assertRaises(ValueError):
                export_physical_codeword_book(tx, path, data_mask=dm)

    def test_explicit_all_ones_mask_raises_value_error(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        dm = torch.ones(1, cfg.M, cfg.N)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            with self.assertRaises(ValueError):
                export_physical_codeword_book(tx, path, data_mask=dm)

    def test_explicit_mask_with_pilot_true_raises_value_error(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        dm = build_transmitter_pilot_masks(cfg).data_mask.clone()
        dm[0, cfg.pilot_delay, cfg.pilot_doppler] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            with self.assertRaises(ValueError):
                export_physical_codeword_book(tx, path, data_mask=dm)

    def test_explicit_mask_with_guard_true_raises_value_error(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        dm = build_transmitter_pilot_masks(cfg).data_mask.clone()
        guard_mask = build_transmitter_pilot_masks(cfg).guard_mask
        guard_positions = guard_mask[0].nonzero()
        dm[0, guard_positions[0, 0], guard_positions[0, 1]] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            with self.assertRaises(ValueError):
                export_physical_codeword_book(tx, path, data_mask=dm)

    def test_explicit_VMN_mask_with_one_token_guard_true_raises(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        dm = build_transmitter_pilot_masks(cfg).data_mask.expand(
            cfg.vocab_size, -1, -1,
        ).clone()
        guard_mask = build_transmitter_pilot_masks(cfg).guard_mask
        guard_positions = guard_mask[0].nonzero()
        dm[2, guard_positions[0, 0], guard_positions[0, 1]] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            with self.assertRaises(ValueError):
                export_physical_codeword_book(tx, path, data_mask=dm)

    def test_explicit_valid_subset_of_config_data_mask_works(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        full_data = build_transmitter_pilot_masks(cfg).data_mask.clone()
        data_positions = full_data[0].nonzero()
        half = len(data_positions) // 2
        for i in range(half, len(data_positions)):
            full_data[0, data_positions[i, 0], data_positions[i, 1]] = False
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = export_physical_codeword_book(tx, path, data_mask=full_data)
            self.assertIn("codeword_book", payload)

    def test_explicit_valid_mask_export_pilot_guard_zero(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        dm = build_transmitter_pilot_masks(cfg).data_mask
        masks = build_transmitter_pilot_masks(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = export_physical_codeword_book(tx, path, data_mask=dm)
            cw = payload["codeword_book"]
            for v in range(cfg.vocab_size):
                self.assertTrue(torch.all(
                    cw[v, masks.pilot_mask[0]].abs() < 1e-5,
                ))
                self.assertTrue(torch.all(
                    cw[v, masks.guard_mask[0]].abs() < 1e-5,
                ))

    def test_metadata_receiver_prior_ready_for_valid_explicit_export(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        dm = build_transmitter_pilot_masks(cfg).data_mask
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = export_physical_codeword_book(tx, path, data_mask=dm)
            self.assertTrue(payload["metadata"]["receiver_prior_ready"])

    def test_parent_dirs_auto_created(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "deep" / "test.pt"
            export_physical_codeword_book(tx, path)
            self.assertTrue(path.exists())

    def test_payload_no_state_dict(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = export_physical_codeword_book(tx, path)
            self.assertNotIn("state_dict", payload)
            self.assertNotIn("raw_real", payload)
            self.assertNotIn("raw_imag", payload)

    # --- zero guard (empty guard_mask) safety ---------------------------

    @staticmethod
    def _zero_guard_config() -> TransmitterConfig:
        """Config with all guard/channel/obs radii set to zero."""
        return TransmitterConfig(
            M=8, N=8, vocab_size=4,
            pilot_delay=3, pilot_doppler=3,
            pilot_guard_delay=0, pilot_guard_doppler=0,
            pilot_obs_delay_radius=0, pilot_obs_doppler_radius=0,
            max_channel_delay=0, max_channel_doppler=0,
            data_power=1.0,
            pilot_value_real=1.0, pilot_value_imag=0.0,
            complex_dtype="complex64",
        )

    def test_zero_guard_config_export_succeeds(self):
        cfg = self._zero_guard_config()
        tx = LearnableTokenDDTransmitter(cfg)
        masks = build_transmitter_pilot_masks(cfg)
        # guard_mask must be empty (all False) for zero guard radii.
        self.assertFalse(masks.guard_mask.any(),
                         "guard_mask should be empty for zero guard config")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "zero_guard.pt"
            payload = export_physical_codeword_book(tx, path)
            self.assertIn("codeword_book", payload)

    def test_zero_guard_export_codeword_book_shape_correct(self):
        cfg = self._zero_guard_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "zero_guard.pt"
            payload = export_physical_codeword_book(tx, path)
            cw = payload["codeword_book"]
            self.assertEqual(cw.shape, (cfg.vocab_size, cfg.M, cfg.N))
            self.assertTrue(torch.is_complex(cw))

    def test_zero_guard_export_metadata_receiver_prior_ready_true(self):
        cfg = self._zero_guard_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "zero_guard.pt"
            payload = export_physical_codeword_book(tx, path)
            self.assertTrue(payload["metadata"]["receiver_prior_ready"])


class SaveRawTxStateTests(unittest.TestCase):
    """Tests for save_raw_tx_state()."""

    def test_with_learnable_tx_creates_file(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw_state.pt"
            payload = save_raw_tx_state(tx, path)
            self.assertTrue(path.exists())
            self.assertIn("state_dict", payload)

    def test_with_codebook_creates_file(self):
        cfg = _tx_config()
        cb = TokenDDCodebook(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cb_raw.pt"
            payload = save_raw_tx_state(cb, path)
            self.assertTrue(path.exists())
            self.assertIn("state_dict", payload)

    def test_invalid_type_raises_type_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pt"
            with self.assertRaises(TypeError):
                save_raw_tx_state("bad", path)  # type: ignore

    def test_metadata_artifact_type_is_raw_tx_state(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = save_raw_tx_state(tx, path)
            meta = payload["metadata"]
            self.assertEqual(meta["artifact_type"], "raw_tx_state")
            self.assertTrue(meta["raw_tx_state_dict"])
            self.assertFalse(meta["physical_codeword_book_exported"])
            self.assertFalse(meta["receiver_prior_ready"])

    def test_payload_no_codeword_book(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = save_raw_tx_state(tx, path)
            self.assertNotIn("codeword_book", payload)

    def test_state_tensors_are_cpu(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            payload = save_raw_tx_state(tx, path)
            for v in payload["state_dict"].values():
                if torch.is_tensor(v):
                    self.assertEqual(v.device.type, "cpu")

    def test_parent_dirs_auto_created(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "a" / "b" / "test.pt"
            save_raw_tx_state(tx, path)
            self.assertTrue(path.exists())


class LoadExportedCodewordBookTests(unittest.TestCase):
    """Tests for load_exported_codeword_book()."""

    def _export_and_load(self, cfg=None):
        cfg = cfg or _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda d=tmpdir: _rmtree(d))
        path = Path(tmpdir) / "export.pt"
        payload = export_physical_codeword_book(tx, path)
        loaded = load_exported_codeword_book(path)
        return payload, loaded, path

    def test_returns_complex_V_M_N(self):
        payload, loaded, _ = self._export_and_load()
        self.assertTrue(torch.is_complex(loaded))
        cfg = _tx_config()
        self.assertEqual(loaded.shape, (cfg.vocab_size, cfg.M, cfg.N))

    def test_equals_exported_payload(self):
        payload, loaded, _ = self._export_and_load()
        self.assertTrue(torch.equal(loaded, payload["codeword_book"]))

    def test_rejects_missing_codeword_book_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pt"
            torch.save({"other": 1}, str(path))
            with self.assertRaises(ValueError):
                load_exported_codeword_book(path)

    def test_rejects_payload_with_state_dict(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw.pt"
            save_raw_tx_state(tx, path)
            with self.assertRaises(ValueError):
                load_exported_codeword_book(path)

    def test_rejects_non_complex_codeword_book(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pt"
            torch.save({"codeword_book": torch.randn(3, 8, 10)}, str(path))
            with self.assertRaises(ValueError):
                load_exported_codeword_book(path)

    def test_rejects_wrong_ndim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pt"
            torch.save({"codeword_book": torch.ones(2, 3, dtype=torch.complex64)}, str(path))
            with self.assertRaises(ValueError):
                load_exported_codeword_book(path)

    def test_rejects_empty_dimensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pt"
            torch.save({
                "codeword_book": torch.zeros(0, 8, 10, dtype=torch.complex64),
                "metadata": {"artifact_type": "physical_codeword_book"},
            }, str(path))
            with self.assertRaises(ValueError):
                load_exported_codeword_book(path)

    def test_rejects_raw_state_artifact_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pt"
            torch.save({
                "codeword_book": torch.ones(3, 8, 10, dtype=torch.complex64),
                "metadata": {"artifact_type": "raw_tx_state"},
            }, str(path))
            with self.assertRaises(ValueError):
                load_exported_codeword_book(path)

    def test_rejects_raw_tx_state_dict_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pt"
            torch.save({
                "codeword_book": torch.ones(3, 8, 10, dtype=torch.complex64),
                "metadata": {"raw_tx_state_dict": True},
            }, str(path))
            with self.assertRaises(ValueError):
                load_exported_codeword_book(path)

    def test_rejects_physical_codeword_book_exported_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pt"
            torch.save({
                "codeword_book": torch.ones(3, 8, 10, dtype=torch.complex64),
                "metadata": {"physical_codeword_book_exported": False},
            }, str(path))
            with self.assertRaises(ValueError):
                load_exported_codeword_book(path)

    def test_rejects_receiver_prior_ready_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pt"
            torch.save({
                "codeword_book": torch.ones(3, 8, 10, dtype=torch.complex64),
                "metadata": {"receiver_prior_ready": False},
            }, str(path))
            with self.assertRaises(ValueError):
                load_exported_codeword_book(path)

    def test_rejects_unknown_artifact_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pt"
            torch.save({
                "codeword_book": torch.ones(3, 8, 10, dtype=torch.complex64),
                "metadata": {"artifact_type": "what_is_this"},
            }, str(path))
            with self.assertRaises(ValueError):
                load_exported_codeword_book(path)

    def test_receiver_prior_compatible(self):
        payload, loaded, _ = self._export_and_load()
        self.assertTrue(torch.is_complex(loaded))
        self.assertEqual(loaded.ndim, 3)
        try:
            receiver_root = ROOT / "Learnable_Receiver_DD-Signals-to-Tokens"
            sys.path.insert(0, str(receiver_root))
            try:
                from receiver.token_prior import TokenCodewordPrior
            except ImportError:
                self.skipTest("TokenCodewordPrior not importable")
            finally:
                if str(receiver_root) in sys.path:
                    sys.path.remove(str(receiver_root))
            prior = TokenCodewordPrior(codeword_book=loaded)
            self.assertIsNotNone(prior)
        except ImportError:
            self.skipTest("Receiver package not available")


def _rmtree(path: str) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
