"""Optional pre-17C2 orbit ablations.

The main transmitter algorithm is the sparse multipath operator margin loss.
The symbols below remain available for controlled ablation studies, but they
are intentionally absent from the package root and must not be combined into
the default training objective without evidence.
"""

from .dd_shifts import dd_circular_shift, dd_circular_shift_bank
from .hard_negative import (
    HardNegativeTokenPairs,
    mine_cross_token_hard_negatives,
)
from .orbit_correlation import (
    masked_normalized_dd_correlation,
    masked_shift_orbit_correlation,
)
from .orbit_visibility import masked_shift_orbit_visibility
from .regularizers import (
    cross_token_orbit_confusion_loss,
    cross_token_orbit_confusion_scores,
    self_shift_orbit_sidelobe_loss,
    self_shift_orbit_sidelobe_scores,
    shift_orbit_visibility_loss,
    shift_orbit_visibility_scores,
)
from .sampling import SampledSparseShiftSet, sample_weighted_sparse_shift_set
from .shaping import SparseShiftSet, default_integer_shift_set

__all__ = [
    "HardNegativeTokenPairs",
    "SampledSparseShiftSet",
    "SparseShiftSet",
    "cross_token_orbit_confusion_loss",
    "cross_token_orbit_confusion_scores",
    "dd_circular_shift",
    "dd_circular_shift_bank",
    "default_integer_shift_set",
    "masked_normalized_dd_correlation",
    "masked_shift_orbit_correlation",
    "masked_shift_orbit_visibility",
    "mine_cross_token_hard_negatives",
    "sample_weighted_sparse_shift_set",
    "self_shift_orbit_sidelobe_loss",
    "self_shift_orbit_sidelobe_scores",
    "shift_orbit_visibility_loss",
    "shift_orbit_visibility_scores",
]
