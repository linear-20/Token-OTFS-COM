from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import (
    FROZEN_BASELINE_ABLATIONS,
    FROZEN_PUBLIC_APIS,
    FROZEN_RECEIVER_CHAIN,
    FROZEN_TX_RX_CONTRACT,
    RECEIVER_FREEZE_VERSION,
    LearnableOTFSReceiver,
    ReceiverConfig,
    named_receiver_paper_ablation,
)


class FreezeVersionTests(unittest.TestCase):
    def test_version_is_rx_v1(self):
        self.assertEqual(RECEIVER_FREEZE_VERSION, "rx_v1")


class FrozenChainTests(unittest.TestCase):
    def test_chain_has_correct_length(self):
        self.assertGreaterEqual(len(FROZEN_RECEIVER_CHAIN), 12)

    def test_chain_contains_core_stages(self):
        chain_str = " ".join(FROZEN_RECEIVER_CHAIN)
        for stage in ("Sparse CE", "Off-grid", "Gain Re-estimation",
                       "DD Operator", "Unfolded Detector",
                       "Candidate-Pruned", "Denoiser",
                       "Data-Consistency", "Token Classifier",
                       "Logit Fusion"):
            self.assertIn(stage, chain_str)

    def test_chain_ends_with_token_logits(self):
        self.assertIn("token_logits", FROZEN_RECEIVER_CHAIN[-1])


class PublicAPITests(unittest.TestCase):
    def test_apis_include_core_classes(self):
        api_set = set(FROZEN_PUBLIC_APIS)
        for name in ("LearnableOTFSReceiver.forward", "ReceiverConfig",
                     "ReceiverOutput", "TokenCodewordPrior"):
            self.assertIn(name, api_set)

    def test_apis_include_losses(self):
        api_set = set(FROZEN_PUBLIC_APIS)
        for name in ("paper_receiver_loss", "token_ce_loss", "total_receiver_loss"):
            self.assertIn(name, api_set)

    def test_apis_include_harness(self):
        api_set = set(FROZEN_PUBLIC_APIS)
        for name in ("validate_receiver_physics", "summarize_receiver_config",
                     "estimate_receiver_ops", "named_receiver_paper_ablation"):
            self.assertIn(name, api_set)


class TXRXContractTests(unittest.TestCase):
    def test_contract_has_codeword_shape(self):
        shared = FROZEN_TX_RX_CONTRACT["shared"]
        self.assertIn("codeword_book_shape", shared)
        self.assertEqual(shared["codeword_book_shape"], "[V, M, N]")

    def test_contract_has_rx_output_shape(self):
        self.assertEqual(
            FROZEN_TX_RX_CONTRACT["rx_output"],
            "token_logits [B, V] float (receiver decision)",
        )

    def test_contract_forbids_forbidden_architectures(self):
        forbidden = " ".join(str(x) for x in FROZEN_TX_RX_CONTRACT["forbidden"])
        for item in ("bit", "QAM", "Transformer", "GNN", "VAE", "dense MN"):
            self.assertIn(item, forbidden)


class BaselineAblationTests(unittest.TestCase):
    def test_all_baseline_ablations_are_recognized(self):
        config = ReceiverConfig(
            M=5, N=4, vocab_size=11, token_embedding_dim=7,
            num_unfolded_layers=2, topk_paths=3, hidden_channels=6,
            noise_var=0.1, use_refinement_net=True,
        )
        for name in FROZEN_BASELINE_ABLATIONS:
            try:
                result = named_receiver_paper_ablation(name, config)
                self.assertIsInstance(result, ReceiverConfig)
            except ValueError as e:
                self.fail(f"ablation '{name}' raised ValueError: {e}")


class BackwardCompatibilityTests(unittest.TestCase):
    def test_default_model_returns_tensor_batch_v(self):
        config = ReceiverConfig(
            M=5, N=4, vocab_size=11, token_embedding_dim=7,
            num_unfolded_layers=2, topk_paths=3, hidden_channels=6,
            noise_var=0.1, use_refinement_net=True, use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = torch.complex(
            torch.randn(2, config.M, config.N),
            torch.randn(2, config.M, config.N),
        )
        logits = model(y_dd)
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (2, config.vocab_size))

    def test_no_bit_qam_transformer_gnn_vae(self):
        config = ReceiverConfig(
            M=5, N=4, vocab_size=11, token_embedding_dim=7,
            num_unfolded_layers=2, topk_paths=3, hidden_channels=6,
            noise_var=0.1, use_refinement_net=True,
        )
        model = LearnableOTFSReceiver(config)
        names = {m.__class__.__name__.lower() for m in model.modules()}
        self.assertFalse(any("transformer" in n for n in names))
        self.assertFalse(any("graph" in n for n in names))
        self.assertFalse(any("vae" in n for n in names))
        self.assertFalse(hasattr(model, "bit_head"))
        self.assertFalse(hasattr(model, "qam_head"))


if __name__ == "__main__":
    unittest.main()
