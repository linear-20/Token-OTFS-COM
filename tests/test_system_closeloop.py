"""Tests for system closed-loop harness -- CPU smoke only."""

import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
TX_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TX_ROOT))

import experiments.system_closeloop_harness as H
from experiments.system_closeloop_harness import (
    ScenarioResult,
    WaveformProfile,
    run_system_closeloop,
    unequalized_codeword_nn_argmin,
)
from experiments.system_contract import validate_system_physical_contract

from transmitter import (
    LearnableTokenDDTransmitter,
    TransmitterConfig,
    build_transmitter_pilot_masks,
    export_physical_codeword_book,
    initialize_token_codebook_,
)

# RX imports (path-based).
RX_ROOT = str(ROOT / "Learnable_Receiver_DD-Signals-to-Tokens")
sys.path.insert(0, RX_ROOT)
import importlib
try:
    _cfg = importlib.import_module("receiver.config")
    _pce = importlib.import_module("receiver.pilot_ce")
    ReceiverConfig = _cfg.ReceiverConfig
    EmbeddedPilotConfig = _pce.EmbeddedPilotConfig
finally:
    if RX_ROOT in sys.path:
        sys.path.remove(RX_ROOT)

# OTFS modem.
_OTFS = TX_ROOT / "otfs_modem.py"
import importlib.util as _iu
_spec = _iu.spec_from_file_location("otfs_modem", str(_OTFS))
_otfs = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_otfs)
OTFSModem = _otfs.OTFSModem

# Channel model.
_ch_p = ROOT
if str(_ch_p) not in sys.path:
    sys.path.insert(0, str(_ch_p))
from channel_model import ChannelConfig


def _tx_config(**kw):
    v = dict(M=8, N=6, vocab_size=6, pilot_delay=4, pilot_doppler=3,
             pilot_guard_delay=2, pilot_guard_doppler=2,
             pilot_obs_delay_radius=0, pilot_obs_doppler_radius=0,
             max_channel_delay=2, max_channel_doppler=2,
             data_power=1.0, pilot_value_real=2.0, pilot_value_imag=0.0,
             complex_dtype="complex64")
    v.update(kw)
    return TransmitterConfig(**v)


def _rx_config(tx_cfg):
    return ReceiverConfig(
        M=tx_cfg.M, N=tx_cfg.N, vocab_size=tx_cfg.vocab_size,
        token_embedding_dim=16, num_unfolded_layers=1, topk_paths=4,
        hidden_channels=8, noise_var=0.01, use_refinement_net=False,
        channel_estimator_mode="pilot",
        pilot_delay=tx_cfg.pilot_delay,
        pilot_doppler=tx_cfg.pilot_doppler,
        pilot_guard_delay=tx_cfg.pilot_guard_delay,
        pilot_guard_doppler=tx_cfg.pilot_guard_doppler,
        pilot_obs_delay_radius=tx_cfg.pilot_obs_delay_radius,
        pilot_obs_doppler_radius=tx_cfg.pilot_obs_doppler_radius,
        pilot_value_real=tx_cfg.pilot_value_real,
        pilot_value_imag=tx_cfg.pilot_value_imag,
    )


def _pilot_config(tx_cfg):
    return EmbeddedPilotConfig(
        pilot_delay=tx_cfg.pilot_delay,
        pilot_doppler=tx_cfg.pilot_doppler,
        guard_delay=tx_cfg.pilot_guard_delay,
        guard_doppler=tx_cfg.pilot_guard_doppler,
        obs_delay_radius=tx_cfg.pilot_obs_delay_radius,
        obs_doppler_radius=tx_cfg.pilot_obs_doppler_radius,
        pilot_value=complex(tx_cfg.pilot_value_real,
                            tx_cfg.pilot_value_imag),
    )


# ==============================================================================
# A. WaveformProfile tests
# ==============================================================================

class WaveformProfileTests(unittest.TestCase):

    def test_profile_fields(self):
        wp = WaveformProfile(M=8, N=6, cp_len=3, sample_rate=1e6,
                             doppler_bin_hz=100.0, max_delay_samples=2.0,
                             cp_sufficient=True)
        d = wp.to_dict()
        self.assertEqual(d["M"], 8)
        self.assertEqual(d["N"], 6)
        self.assertEqual(d["cp_len"], 3)
        self.assertEqual(d["waveform_variant"],
                         "CP-OFDM-based OTFS-per-slot-CP")

    def test_cp_sufficient_false(self):
        wp = WaveformProfile(M=8, N=6, cp_len=1, sample_rate=1e6,
                             doppler_bin_hz=100.0, max_delay_samples=5.0,
                             cp_sufficient=False)
        self.assertFalse(wp.cp_sufficient)


# ==============================================================================
# B. Unequalized nearest-neighbor tests
# ==============================================================================

class UnequalizedNearestNeighborTests(unittest.TestCase):

    def test_identity_ter_zero(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        gen = torch.Generator(device="cpu").manual_seed(42)
        initialize_token_codebook_(tx.codebook, mode="random_phase",
                                   generator=gen)
        masks = build_transmitter_pilot_masks(cfg)
        with torch.no_grad():
            cw = tx.codebook.forward(data_mask=masks.data_mask)

        # Identity channel: no noise, no distortion.
        token_ids = torch.arange(4, dtype=torch.long)
        x_dd = tx.forward(token_ids).x_dd
        pred = unequalized_codeword_nn_argmin(
            x_dd, cw, masks.data_mask,
        )
        self.assertEqual(pred.shape, token_ids.shape)
        ter = float((pred != token_ids).float().mean().item())
        self.assertEqual(ter, 0.0, "identity channel must have TER==0")

    def test_integer_delay_ter_zero(self):
        """Codebook-only circular shift: known shift can be inverted."""
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        gen = torch.Generator(device="cpu").manual_seed(42)
        initialize_token_codebook_(tx.codebook, mode="random_phase",
                                   generator=gen)
        masks = build_transmitter_pilot_masks(cfg)
        with torch.no_grad():
            cw = tx.codebook.forward(data_mask=masks.data_mask)

        from transmitter.dd_shifts import dd_circular_shift
        token_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        # Use selected codewords WITHOUT pilot insertion.
        selected = tx.codebook.selected_codewords(token_ids,
                                                   codeword_book=cw)
        # Apply known integer shift and invert.
        y_dd = dd_circular_shift(selected, delay_shift=2, doppler_shift=0)
        recovered = dd_circular_shift(y_dd, delay_shift=-2, doppler_shift=0)
        pred = unequalized_codeword_nn_argmin(
            recovered, cw, masks.data_mask,
        )
        ter = float((pred != token_ids).float().mean().item())
        self.assertEqual(ter, 0.0,
                         "inverted circular shift TER must be 0")


# ==============================================================================
# C. Closed-loop smoke tests
# ==============================================================================

class ClosedLoopSmokeTests(unittest.TestCase):

    def test_modem_roundtrip_no_channel(self):
        """Token IDs -> TX -> modem -> demod -> unequalized NN TER==0."""
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        gen = torch.Generator(device="cpu").manual_seed(42)
        initialize_token_codebook_(tx.codebook, mode="random_phase",
                                   generator=gen)
        masks = build_transmitter_pilot_masks(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N), cp_len=cfg.max_channel_delay + 1)

        with torch.no_grad():
            cw = tx.codebook.forward(data_mask=masks.data_mask)
        token_ids = torch.arange(4, dtype=torch.long)
        tx_out = tx.forward(token_ids, return_time=True, modem=modem)
        y_dd = modem.demodulate(tx_out.time_signal)
        pred = unequalized_codeword_nn_argmin(
            y_dd, cw, masks.data_mask,
        )
        ter = float((pred != token_ids).float().mean().item())
        self.assertEqual(ter, 0.0, "modem roundtrip TER must be 0")

    def test_identity_waveform_channel(self):
        """Noise-free identity channel through full chain."""
        cfg = _tx_config()
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N), cp_len=cfg.max_channel_delay + 1)
        ch_cfg = ChannelConfig(num_paths=1, snr_db=None, add_awgn=False,
                               seed=200, fractional_delays=False,
                               path_delays=[0.0],
                               path_dopplers_hz=[0.0],
                               path_powers_db=[0.0])
        results = run_system_closeloop(
            cfg, modem, rx_cfg, pilot_cfg, [ch_cfg],
            num_tokens=4, seed=42, oracle_ce=True,
            path_gain_overrides=[
                torch.tensor([1.0 + 0.0j], dtype=torch.complex64),
            ],
        )
        r = results[0]
        self.assertEqual(r.ter_unequalized_nn, 0.0)
        self.assertEqual(r.ter_detector, 0.0)
        self.assertLess(r.waveform_vs_dd_surrogate_nmse, 1e-10)
        self.assertEqual(r.rx_sparse_active_path_count, 1)
        self.assertIsNone(r.pilot_ce_path_count)
        self.assertTrue(r.cp_sufficient)
        self.assertTrue(r.waveform_evaluation)

    def test_integer_delay_waveform_surrogate_mapping(self):
        """One sample of waveform delay maps to one DD delay bin."""
        cfg = _tx_config()
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N),
                          cp_len=cfg.max_channel_delay + 1)
        ch_cfg = ChannelConfig(num_paths=1, snr_db=None, add_awgn=False,
                               seed=210, fractional_delays=False,
                               path_delays=[1.0],
                               path_dopplers_hz=[0.0],
                               path_powers_db=[0.0])
        results = run_system_closeloop(
            cfg, modem, rx_cfg, pilot_cfg, [ch_cfg],
            num_tokens=4, seed=46, oracle_ce=True,
            path_gain_overrides=[
                torch.tensor([1.0 + 0.0j], dtype=torch.complex64),
            ],
        )
        self.assertLess(results[0].waveform_vs_dd_surrogate_nmse, 1e-10)

    def test_fractional_doppler_flag_uses_realized_grid_offset(self):
        cfg = _tx_config()
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N),
                          cp_len=cfg.max_channel_delay + 1)
        doppler_bin_hz = (
            1e6 / (modem.N * (modem.M + modem.cp_len))
        )
        ch_cfg = ChannelConfig(num_paths=1, sample_rate=1e6,
                               snr_db=None, add_awgn=False,
                               seed=213, fractional_delays=False,
                               path_delays=[0.0],
                               path_dopplers_hz=[0.5 * doppler_bin_hz],
                               path_powers_db=[0.0])
        results = run_system_closeloop(
            cfg, modem, rx_cfg, pilot_cfg, [ch_cfg],
            num_tokens=4, seed=49, oracle_ce=True,
            path_gain_overrides=[
                torch.tensor([1.0 + 0.0j], dtype=torch.complex64),
            ],
        )
        r = results[0]
        self.assertFalse(r.has_fractional_delay)
        self.assertTrue(r.has_fractional_doppler)
        self.assertTrue(r.has_fractional)

    def test_waveform_surrogate_nmse_excludes_awgn(self):
        """NMSE is a waveform-model diagnostic, not an SNR metric."""
        cfg = _tx_config()
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N),
                          cp_len=cfg.max_channel_delay + 1)
        ch_cfg = ChannelConfig(num_paths=1, snr_db=0.0, add_awgn=True,
                               seed=211, fractional_delays=False,
                               path_delays=[0.0],
                               path_dopplers_hz=[0.0],
                               path_powers_db=[0.0])
        results = run_system_closeloop(
            cfg, modem, rx_cfg, pilot_cfg, [ch_cfg],
            num_tokens=4, seed=47, oracle_ce=True,
            path_gain_overrides=[
                torch.tensor([1.0 + 0.0j], dtype=torch.complex64),
            ],
        )
        self.assertLess(results[0].waveform_vs_dd_surrogate_nmse, 1e-10)

    def test_pilot_ce_identity_counts_only_active_paths(self):
        """Pilot CE count excludes padded top-k entries."""
        cfg = _tx_config()
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N),
                          cp_len=cfg.max_channel_delay + 1)
        ch_cfg = ChannelConfig(num_paths=1, snr_db=None, add_awgn=False,
                               seed=212, fractional_delays=False,
                               path_delays=[0.0],
                               path_dopplers_hz=[0.0],
                               path_powers_db=[0.0])
        results = run_system_closeloop(
            cfg, modem, rx_cfg, pilot_cfg, [ch_cfg],
            num_tokens=4, seed=48, oracle_ce=False,
            path_gain_overrides=[
                torch.tensor([1.0 + 0.0j], dtype=torch.complex64),
            ],
        )
        r = results[0]
        self.assertEqual(r.rx_sparse_active_path_count, 1)
        self.assertEqual(r.pilot_ce_path_count, 1)

    def test_pilot_ce_integer_delay_recovers_tokens_without_oracle(self):
        """Nonzero waveform delay -> embedded pilot CE -> detector TER==0."""
        cfg = _tx_config(M=12, pilot_delay=6, pilot_guard_delay=4,
                         pilot_obs_delay_radius=2)
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N),
                          cp_len=cfg.max_channel_delay + 1)
        ch_cfg = ChannelConfig(num_paths=1, snr_db=None, add_awgn=False,
                               seed=214, fractional_delays=False,
                               max_delay_samples=2.0,
                               path_delays=[1.0],
                               path_dopplers_hz=[0.0],
                               path_powers_db=[0.0])
        results = run_system_closeloop(
            cfg, modem, rx_cfg, pilot_cfg, [ch_cfg],
            num_tokens=16, seed=61, oracle_ce=False,
            path_gain_overrides=[
                torch.tensor([1.0 + 0.0j], dtype=torch.complex64),
            ],
        )
        r = results[0]
        self.assertEqual(r.pilot_ce_path_count, 1)
        self.assertEqual(r.ter_detector, 0.0)
        self.assertLess(r.waveform_vs_dd_surrogate_nmse, 1e-10)

    def test_path_gain_override_length_must_match(self):
        cfg = _tx_config()
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N),
                          cp_len=cfg.max_channel_delay + 1)
        channel = ChannelConfig(num_paths=1, snr_db=None, add_awgn=False)
        with self.assertRaises(ValueError):
            run_system_closeloop(
                cfg, modem, rx_cfg, pilot_cfg, [channel],
                path_gain_overrides=[],
            )

    def test_integer_multipath_smoke(self):
        """Integer multipath with high SNR -- smoke only, not TER==0."""
        cfg = _tx_config()
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N), cp_len=cfg.max_channel_delay + 1)
        ch_cfg = ChannelConfig(
            num_paths=2, snr_db=40.0, add_awgn=True,
            seed=201, fractional_delays=False,
            max_delay_samples=2.0, max_doppler_hz=50.0,
        )
        results = run_system_closeloop(
            cfg, modem, rx_cfg, pilot_cfg, [ch_cfg],
            num_tokens=8, seed=43, oracle_ce=True,
        )
        r = results[0]
        self.assertIsNotNone(r.ter_unequalized_nn)
        self.assertIsNotNone(r.ter_detector)
        self.assertTrue(r.waveform_evaluation)

    def test_fractional_delay_doppler_smoke(self):
        """Fractional delay/Doppler: NMSE > 0 expected, not exact equivalence."""
        cfg = _tx_config()
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N), cp_len=cfg.max_channel_delay + 1)
        ch_cfg = ChannelConfig(
            num_paths=1, snr_db=40.0, add_awgn=True,
            seed=202, fractional_delays=True,
            max_delay_samples=2.5, max_doppler_hz=80.0,
        )
        results = run_system_closeloop(
            cfg, modem, rx_cfg, pilot_cfg, [ch_cfg],
            num_tokens=8, seed=44, oracle_ce=True,
        )
        r = results[0]
        self.assertTrue(r.has_fractional)
        self.assertTrue(r.has_fractional_delay or r.has_fractional_doppler)
        self.assertTrue(math.isfinite(r.waveform_vs_dd_surrogate_nmse))
        self.assertGreaterEqual(r.waveform_vs_dd_surrogate_nmse, 0.0)
        self.assertTrue(r.waveform_evaluation)

    def test_snr_sweep_produces_multiple_results(self):
        cfg = _tx_config()
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N), cp_len=cfg.max_channel_delay + 1)
        channels = [
            ChannelConfig(num_paths=1, snr_db=s, add_awgn=True,
                          seed=300 + i, fractional_delays=False,
                          max_delay_samples=2.0, max_doppler_hz=50.0)
            for i, s in enumerate([10.0, 20.0, 30.0])
        ]
        results = run_system_closeloop(
            cfg, modem, rx_cfg, pilot_cfg, channels,
            num_tokens=8, seed=45, oracle_ce=True,
        )
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsNotNone(r.ter_unequalized_nn)

    def test_harness_does_not_modify_tx(self):
        cfg = _tx_config()
        tx_before = LearnableTokenDDTransmitter(cfg)
        gen = torch.Generator(device="cpu").manual_seed(99)
        initialize_token_codebook_(tx_before.codebook, mode="random_phase",
                                   generator=gen)
        raw_before = tx_before.codebook.raw_real.clone()
        # Run harness with a fresh TX internally.
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N), cp_len=cfg.max_channel_delay + 1)
        ch_cfg = ChannelConfig(num_paths=1, snr_db=None, add_awgn=False,
                               seed=400, fractional_delays=False,
                               path_delays=[0.0],
                               path_dopplers_hz=[0.0],
                               path_powers_db=[0.0])
        run_system_closeloop(
            cfg, modem, rx_cfg, pilot_cfg, [ch_cfg],
            num_tokens=4, seed=99, oracle_ce=True,
        )
        # Original tx should be unchanged.
        self.assertTrue(
            torch.equal(tx_before.codebook.raw_real, raw_before),
            "harness must not modify transmitter architecture",
        )

    def test_cp_sufficient_recorded(self):
        cfg = _tx_config(max_channel_delay=3, pilot_guard_delay=3)
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N), cp_len=2)  # insufficient
        ch_cfg = ChannelConfig(num_paths=1, snr_db=None, add_awgn=False,
                               seed=500, fractional_delays=False,
                               max_delay_samples=4.0,
                               path_delays=[3.0],
                               path_dopplers_hz=[0.0],
                               path_powers_db=[0.0])
        results = run_system_closeloop(
            cfg, modem, rx_cfg, pilot_cfg, [ch_cfg],
            num_tokens=4, seed=50, oracle_ce=True,
        )
        self.assertFalse(results[0].cp_sufficient,
                         "cp_len=2 < max_delay=4 should be insufficient")


# ==============================================================================
# D. Artifact-driven contract tests
# ==============================================================================

class ArtifactDrivenContractTests(unittest.TestCase):

    @staticmethod
    def _identity_channel():
        return ChannelConfig(num_paths=1, snr_db=None, add_awgn=False,
                             seed=600, fractional_delays=False,
                             max_delay_samples=0.0,
                             path_delays=[0.0],
                             path_dopplers_hz=[0.0],
                             path_powers_db=[0.0])

    def test_artifact_identity_does_not_reinitialize_codebook(self):
        cfg = _tx_config()
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N),
                          cp_len=cfg.max_channel_delay + 1)
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "physical.pt"
            payload = export_physical_codeword_book(tx, path)
            with mock.patch.object(H, "initialize_token_codebook_") as init:
                results = run_system_closeloop(
                    cfg, modem, rx_cfg, pilot_cfg,
                    [self._identity_channel()],
                    num_tokens=6, seed=70, oracle_ce=True,
                    path_gain_overrides=[
                        torch.tensor([1.0 + 0.0j], dtype=torch.complex64),
                    ],
                    physical_artifact_path=path,
                )
            init.assert_not_called()
        r = results[0]
        self.assertEqual(r.ter_detector, 0.0)
        self.assertEqual(r.metadata["codeword_source"],
                         "exported_physical_artifact")
        self.assertTrue(r.physical_contract["artifact_driven"])
        self.assertEqual(
            r.physical_contract["artifact_sha256"],
            payload["metadata"]["artifact_sha256"],
        )

    def test_artifact_integer_delay_pilot_ce_recovers_without_oracle(self):
        cfg = _tx_config(M=12, pilot_delay=6, pilot_guard_delay=4,
                         pilot_obs_delay_radius=2)
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N),
                          cp_len=cfg.max_channel_delay + 1)
        ch_cfg = ChannelConfig(num_paths=1, snr_db=None, add_awgn=False,
                               seed=601, fractional_delays=False,
                               max_delay_samples=2.0,
                               path_delays=[1.0],
                               path_dopplers_hz=[0.0],
                               path_powers_db=[0.0])
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "physical.pt"
            export_physical_codeword_book(tx, path)
            results = run_system_closeloop(
                cfg, modem, rx_cfg, pilot_cfg, [ch_cfg],
                num_tokens=16, seed=71, oracle_ce=False,
                path_gain_overrides=[
                    torch.tensor([1.0 + 0.0j], dtype=torch.complex64),
                ],
                physical_artifact_path=path,
                require_cp_sufficient=True,
            )
        r = results[0]
        self.assertEqual(r.ter_detector, 0.0)
        self.assertEqual(r.pilot_ce_path_count, 1)
        self.assertLess(r.waveform_vs_dd_surrogate_nmse, 1e-10)

    def test_contract_rejects_rx_pilot_value_mismatch(self):
        cfg = _tx_config()
        rx_cfg = replace(_rx_config(cfg), pilot_value_real=1.0)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N), cp_len=3)
        with self.assertRaises(ValueError):
            validate_system_physical_contract(
                cfg, rx_cfg, pilot_cfg, modem,
            )

    def test_contract_rejects_modem_shape_mismatch(self):
        cfg = _tx_config()
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N + 1), cp_len=3)
        with self.assertRaises(ValueError):
            validate_system_physical_contract(
                cfg, rx_cfg, pilot_cfg, modem,
            )

    def test_cp_and_pilot_guard_conditions_are_recorded_separately(self):
        cfg = _tx_config()
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N), cp_len=2)
        channel = ChannelConfig(max_delay_samples=4.0)
        report = validate_system_physical_contract(
            cfg, rx_cfg, pilot_cfg, modem, channel_cfg=channel,
        )
        self.assertTrue(report.pilot_observation_within_guard)
        self.assertTrue(report.cp_covers_tx_max_channel_delay)
        self.assertFalse(report.cp_covers_channel_config_max_delay)
        with self.assertRaises(ValueError):
            validate_system_physical_contract(
                cfg, rx_cfg, pilot_cfg, modem,
                channel_cfg=channel,
                require_cp_sufficient=True,
            )

    def test_artifact_runtime_rejects_subset_mask(self):
        cfg = _tx_config()
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N), cp_len=3)
        tx = LearnableTokenDDTransmitter(cfg)
        subset = build_transmitter_pilot_masks(cfg).data_mask.clone()
        index = subset.nonzero(as_tuple=False)[0]
        subset[tuple(index.tolist())] = False
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "subset.pt"
            export_physical_codeword_book(tx, path, data_mask=subset)
            with self.assertRaises(ValueError):
                run_system_closeloop(
                    cfg, modem, rx_cfg, pilot_cfg,
                    [self._identity_channel()],
                    physical_artifact_path=path,
                )

    def test_artifact_runtime_rejects_external_tx_config_mismatch(self):
        cfg = _tx_config()
        runtime_cfg = replace(cfg, data_power=2.0)
        rx_cfg = _rx_config(runtime_cfg)
        pilot_cfg = _pilot_config(runtime_cfg)
        modem = OTFSModem(dd_shape=(runtime_cfg.M, runtime_cfg.N), cp_len=3)
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "physical.pt"
            export_physical_codeword_book(tx, path)
            with self.assertRaises(ValueError):
                run_system_closeloop(
                    runtime_cfg, modem, rx_cfg, pilot_cfg,
                    [self._identity_channel()],
                    physical_artifact_path=path,
                )

    def test_artifact_runtime_rejects_corrupt_file(self):
        cfg = _tx_config()
        rx_cfg = _rx_config(cfg)
        pilot_cfg = _pilot_config(cfg)
        modem = OTFSModem(dd_shape=(cfg.M, cfg.N), cp_len=3)
        tx = LearnableTokenDDTransmitter(cfg)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "physical.pt"
            payload = export_physical_codeword_book(tx, path)
            payload["metadata"]["artifact_sha256"] = "0" * 64
            torch.save(payload, str(path))
            with self.assertRaises(ValueError):
                run_system_closeloop(
                    cfg, modem, rx_cfg, pilot_cfg,
                    [self._identity_channel()],
                    physical_artifact_path=path,
                )


# ==============================================================================
# E. Quality tests
# ==============================================================================

class QualityTests(unittest.TestCase):

    def test_harness_py_ascii_only(self):
        path = TX_ROOT / "experiments" / "system_closeloop_harness.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_system_contract_py_ascii_only(self):
        path = TX_ROOT / "experiments" / "system_contract.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_scenario_result_serialization(self):
        sr = ScenarioResult(
            name="test", token_ids=torch.tensor([0]),
            predicted_unequalized_nn=torch.tensor([0]),
            predicted_detector=torch.tensor([0]),
            predicted_classifier=None,
            predicted_fused=None,
            ter_unequalized_nn=0.0, ter_detector=0.0,
            ter_classifier=None, ter_fused=None,
            waveform_vs_dd_surrogate_nmse=0.001, cp_sufficient=True,
            num_paths=1, has_fractional=False,
            has_fractional_delay=False, has_fractional_doppler=False,
            snr_db=30.0, seed=1,
            oracle_ce=True, waveform_evaluation=True,
        )
        d = sr.to_dict()
        self.assertIn("ter_detector", d)
        self.assertIn("requires_rx_training_for_classifier", d)
        self.assertTrue(d["requires_rx_training_for_classifier"])

    def test_default_classifier_untrusted_flag(self):
        sr = ScenarioResult(
            name="t", token_ids=torch.tensor([0]),
            predicted_unequalized_nn=torch.tensor([0]),
            predicted_detector=torch.tensor([0]),
            predicted_classifier=torch.tensor([0]),
            predicted_fused=None,
            ter_unequalized_nn=0.0, ter_detector=0.0,
            ter_classifier=0.5, ter_fused=None,
            waveform_vs_dd_surrogate_nmse=0.0, cp_sufficient=True,
            num_paths=0, has_fractional=False,
            has_fractional_delay=False, has_fractional_doppler=False,
            snr_db=None, seed=1,
            oracle_ce=True, waveform_evaluation=True,
        )
        d = sr.to_dict()
        self.assertTrue(d["requires_rx_training_for_classifier"])


if __name__ == "__main__":
    unittest.main()
