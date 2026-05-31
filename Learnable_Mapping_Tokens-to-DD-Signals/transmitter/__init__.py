"""Core public API for direct token transmission over OTFS.

The package root intentionally exposes one compact physical chain and one
operator-aware shaping objective. Optional measurements live in
``transmitter.diagnostics``. Pre-17C2 orbit experiments live in
``transmitter.ablations`` and are not part of the main algorithm surface.
"""

from .codebook import TokenDDCodebook, initialize_token_codebook_
from .config import TransmitterConfig
from .export import (
    PHYSICAL_CODEWORD_BOOK_SCHEMA_VERSION,
    export_physical_codeword_book,
    load_exported_codeword_book,
    load_exported_physical_dd_config,
    load_exported_physical_dd_artifact,
    save_raw_tx_state,
)
from .modem_adapter import modulate_otfs_dd_frame
from .model import LearnableTokenDDTransmitter, TokenDDTransmitterOutput
from .multipath_margin import (
    sparse_multipath_operator_margin_loss,
    sparse_multipath_operator_separation_scores,
)
from .multipath_scenarios import (
    SparseMultipathScenarioBank,
    sample_normalized_sparse_multipath_scenario_bank,
)
from .pilot_frame import (
    TransmitterPilotMasks,
    build_transmitter_pilot_masks,
    insert_transmitter_pilot,
)
from .sampling import sample_uniform_cross_token_pairs
from .shaping import SparseShiftSet, default_integer_shift_set
from .sparse_multipath import (
    SparseMultipathDDChannel,
    apply_sparse_multipath_dd_operator,
)

__all__ = [
    "LearnableTokenDDTransmitter",
    "PHYSICAL_CODEWORD_BOOK_SCHEMA_VERSION",
    "SparseMultipathDDChannel",
    "SparseMultipathScenarioBank",
    "SparseShiftSet",
    "TokenDDCodebook",
    "TokenDDTransmitterOutput",
    "TransmitterConfig",
    "TransmitterPilotMasks",
    "apply_sparse_multipath_dd_operator",
    "build_transmitter_pilot_masks",
    "default_integer_shift_set",
    "export_physical_codeword_book",
    "initialize_token_codebook_",
    "insert_transmitter_pilot",
    "load_exported_codeword_book",
    "load_exported_physical_dd_config",
    "load_exported_physical_dd_artifact",
    "modulate_otfs_dd_frame",
    "sample_normalized_sparse_multipath_scenario_bank",
    "sample_uniform_cross_token_pairs",
    "save_raw_tx_state",
    "sparse_multipath_operator_margin_loss",
    "sparse_multipath_operator_separation_scores",
]
