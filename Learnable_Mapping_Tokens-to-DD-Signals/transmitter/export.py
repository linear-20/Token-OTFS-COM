"""Export and load transmitter artifacts: raw state and physical codeword book.

Public API:
    PHYSICAL_CODEWORD_BOOK_SCHEMA_VERSION - current artifact schema version
    export_physical_codeword_book - save receiver-prior-ready codeword book
    save_raw_tx_state             - save raw learnable transmitter state_dict
    load_exported_codeword_book   - load and validate physical codeword book
    load_exported_physical_dd_artifact - strict schema-aware loader
    load_exported_physical_dd_config - recover config from strict artifact
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

from .codebook import TokenDDCodebook
from .config import TransmitterConfig
from .metrics import _validate_binary_support_mask, _complex_to_real_dtype
from .model import LearnableTokenDDTransmitter
from .pilot_frame import build_transmitter_pilot_masks

PHYSICAL_CODEWORD_BOOK_SCHEMA_VERSION = 1


def export_physical_codeword_book(
    transmitter: LearnableTokenDDTransmitter | TokenDDCodebook,
    output_path: str | Path,
    data_mask: torch.Tensor | None = None,
    *,
    include_metadata: bool = True,
) -> dict:
    """Export receiver-prior-ready physical normalized codeword_book [V, M, N].

    The exported codeword_book has been projected onto a hard binary data
    support mask, zeroed in pilot/guard regions, and equal-power normalized.
    It is the exact tensor the receiver TokenCodewordPrior should consume.

    Args:
        transmitter: LearnableTokenDDTransmitter or TokenDDCodebook.
        output_path: File path for the saved .pt artifact.
        data_mask: Optional hard binary support mask [1, M, N] or [V, M, N].
            If None, build_transmitter_pilot_masks(config).data_mask is used.
            Fractional masks are rejected.
        include_metadata: If True, include metadata dict in the payload.

    Returns:
        Dict with keys "codeword_book" (complex [V, M, N] CPU) and
        "data_mask" (bool [1, M, N] or [V, M, N] CPU). If include_metadata
        is True, "metadata" is also included.

    Raises:
        TypeError: If transmitter type is invalid or data_mask is complex.
        ValueError: If physical consistency checks fail.
    """
    config, codebook = _resolve_codebook(transmitter)

    if data_mask is None:
        masks = build_transmitter_pilot_masks(config)
        resolved_mask = masks.data_mask
    else:
        _validate_export_mask_subset_of_data_region(data_mask, config)
        resolved_mask = data_mask

    with torch.no_grad():
        codeword_book = codebook.forward(data_mask=resolved_mask)

    codeword_book = codeword_book.detach().cpu()
    bool_mask = _resolve_export_data_mask(resolved_mask, config)

    _validate_physical_export_constraints(
        codeword_book, bool_mask, config,
    )

    payload: dict = {
        "codeword_book": codeword_book,
        "data_mask": bool_mask,
    }
    if include_metadata:
        meta = _physical_export_metadata(config)
        # Compute SHA-256 over codeword_book, data_mask, and canonical
        # metadata (without the hash field itself).
        _canonical = _canonical_hash_bytes(
            codeword_book, bool_mask, meta,
        )
        meta["artifact_sha256"] = hashlib.sha256(_canonical).hexdigest()
        payload["metadata"] = meta

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output_path))
    return payload


def save_raw_tx_state(
    transmitter: LearnableTokenDDTransmitter | TokenDDCodebook,
    output_path: str | Path,
    *,
    include_config: bool = True,
) -> dict:
    """Save raw transmitter state_dict for training/optimization recovery.

    This saves the raw learnable parameters (raw_real, raw_imag). It does NOT
    save the physical codeword_book. The saved state_dict can be loaded via
    transmitter.load_state_dict().

    Args:
        transmitter: LearnableTokenDDTransmitter or TokenDDCodebook.
        output_path: File path for the saved .pt artifact.
        include_config: If True, include metadata dict in the payload.

    Returns:
        Dict with key "state_dict". If include_config is True, "metadata"
        is also included.

    Raises:
        TypeError: If transmitter type is invalid.
    """
    if isinstance(transmitter, LearnableTokenDDTransmitter):
        config = transmitter.config
        state_dict = transmitter.state_dict()
    elif isinstance(transmitter, TokenDDCodebook):
        config = transmitter.config
        state_dict = transmitter.state_dict()
    else:
        raise TypeError(
            "transmitter must be a LearnableTokenDDTransmitter or "
            f"TokenDDCodebook, got {type(transmitter).__name__}."
        )

    cpu_state = _to_cpu_state_dict(state_dict)
    payload: dict = {"state_dict": cpu_state}
    if include_config:
        payload["metadata"] = _raw_state_metadata(config)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output_path))
    return payload


def load_exported_physical_dd_artifact(
    path: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
) -> dict:
    """Strict schema-aware loader for physical codeword book artifacts.

    Uses torch.load(weights_only=True). Rejects artifacts with:
    - missing or non-dict metadata
    - missing or wrong schema_version
    - missing data_mask
    - invalid metadata flags
    - non-complex / non-3D / NaN / Inf codeword
    - metadata/codeword shape/dtype mismatch
    - non-binary mask / wrong mask shape / mask outside data region
    - non-equal-power codewords
    - non-zero pilot/guard regions
    - mismatched or missing artifact_sha256

    Args:
        path: File path of the exported .pt artifact.
        map_location: torch.load map_location parameter.

    Returns:
        Dict with keys "codeword_book" (complex [V, M, N]),
        "data_mask" (bool [1, M, N]), and "metadata" (dict).

    Raises:
        ValueError/TypeError: If any contract check fails.
    """
    payload = torch.load(str(path), map_location=map_location,
                         weights_only=True)

    if not isinstance(payload, dict):
        raise ValueError("Exported file must contain a dict payload.")

    if "state_dict" in payload:
        raise ValueError(
            "Payload contains 'state_dict'. This is a raw tx state file."
        )

    if "codeword_book" not in payload:
        raise ValueError("Payload missing required key 'codeword_book'.")

    if "data_mask" not in payload:
        raise ValueError("Payload missing required key 'data_mask'.")

    if "metadata" not in payload or not isinstance(payload["metadata"], dict):
        raise ValueError(
            "Payload must contain a 'metadata' dict. "
            "Artifacts without metadata are diagnostic-only and "
            "cannot be loaded by the strict contract loader."
        )

    metadata = payload["metadata"]
    schema_version = metadata.get("schema_version", None)
    if schema_version is None:
        raise ValueError("metadata missing 'schema_version'.")
    if schema_version != PHYSICAL_CODEWORD_BOOK_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {schema_version}. "
            f"This loader supports version {PHYSICAL_CODEWORD_BOOK_SCHEMA_VERSION}."
        )

    if metadata.get("artifact_type") != "physical_codeword_book":
        raise ValueError(
            f"metadata artifact_type must be 'physical_codeword_book', "
            f"got {metadata.get('artifact_type')!r}."
        )
    if metadata.get("raw_tx_state_dict", None) is not False:
        raise ValueError("metadata raw_tx_state_dict must be False.")
    if metadata.get("physical_codeword_book_exported", None) is not True:
        raise ValueError(
            "metadata physical_codeword_book_exported must be True."
        )
    if metadata.get("receiver_prior_ready", None) is not True:
        raise ValueError("metadata receiver_prior_ready must be True.")
    if metadata.get("hard_data_mask_projection", None) is not True:
        raise ValueError("metadata hard_data_mask_projection must be True.")
    if metadata.get("codeword_is_finite", None) is not True:
        raise ValueError("metadata codeword_is_finite must be True.")
    _validate_physical_export_metadata(metadata)

    codeword_book = payload["codeword_book"]
    data_mask = payload["data_mask"]

    # Validate codeword_book.
    if not torch.is_tensor(codeword_book):
        raise ValueError("codeword_book must be a torch.Tensor.")
    if not torch.is_complex(codeword_book):
        raise ValueError("codeword_book must be a complex tensor.")
    if not torch.isfinite(codeword_book).all():
        raise ValueError("codeword_book must contain only finite values.")
    if codeword_book.ndim != 3:
        raise ValueError(
            f"codeword_book must have 3 dimensions [V, M, N], "
            f"got ndim={codeword_book.ndim}."
        )
    V, M_loaded, N_loaded = codeword_book.shape
    if V <= 0 or M_loaded <= 0 or N_loaded <= 0:
        raise ValueError(
            f"codeword_book dimensions must be > 0, "
            f"got {list(codeword_book.shape)}."
        )

    # Validate metadata consistency.
    for key in ("M", "N", "vocab_size"):
        if metadata.get(key) is None:
            raise ValueError(f"metadata missing '{key}'.")
    if metadata["M"] != M_loaded or metadata["N"] != N_loaded:
        raise ValueError(
            f"metadata shape [{metadata['M']}, {metadata['N']}] "
            f"does not match codeword_book shape [{M_loaded}, {N_loaded}]."
        )
    if metadata["vocab_size"] != V:
        raise ValueError(
            f"metadata vocab_size {metadata['vocab_size']} "
            f"does not match codeword_book V={V}."
        )
    expected_dtype = {"complex64": torch.complex64,
                      "complex128": torch.complex128}.get(
                          metadata.get("complex_dtype"))
    if expected_dtype is None:
        raise ValueError(
            f"metadata complex_dtype '{metadata.get('complex_dtype')}' "
            "is not 'complex64' or 'complex128'."
        )
    if codeword_book.dtype != expected_dtype:
        raise ValueError(
            f"codeword_book dtype {codeword_book.dtype} does not match "
            f"metadata complex_dtype {metadata['complex_dtype']}."
        )

    # Validate data_mask.
    if not torch.is_tensor(data_mask):
        raise ValueError("data_mask must be a torch.Tensor.")
    if data_mask.ndim != 3:
        raise ValueError(
            f"data_mask must have 3 dimensions, got ndim={data_mask.ndim}."
        )
    if data_mask.shape != (1, M_loaded, N_loaded) and \
       data_mask.shape != (V, M_loaded, N_loaded):
        raise ValueError(
            f"data_mask shape {list(data_mask.shape)} must be "
            f"[1, {M_loaded}, {N_loaded}] or [{V}, {M_loaded}, {N_loaded}]."
        )
    # Rebuild config and validate physical constraints.
    config = _config_from_metadata(metadata)
    _validate_export_mask_subset_of_data_region(data_mask, config)
    _validate_physical_export_constraints(codeword_book, data_mask, config)

    # Verify SHA-256.
    stored_hash = metadata.get("artifact_sha256", None)
    if stored_hash is None:
        raise ValueError("metadata missing 'artifact_sha256'.")
    if not isinstance(stored_hash, str) or len(stored_hash) != 64 or \
       any(ch not in "0123456789abcdef" for ch in stored_hash):
        raise ValueError(
            "metadata artifact_sha256 must be a 64-character lowercase "
            "hexadecimal string."
        )
    canonical_bytes = _canonical_hash_bytes(
        codeword_book, data_mask, metadata,
    )
    computed_hash = hashlib.sha256(canonical_bytes).hexdigest()
    if computed_hash != stored_hash:
        raise ValueError(
            f"artifact_sha256 mismatch: stored {stored_hash[:16]}..., "
            f"computed {computed_hash[:16]}... "
            "The artifact may be corrupted or tampered with."
        )

    return payload


def load_exported_codeword_book(
    path: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
) -> torch.Tensor:
    """Load and validate an exported physical codeword_book [V, M, N].

    Delegates to the strict schema-aware loader.
    Returns the complex codeword_book tensor for receiver prior consumption.

    Args:
        path: File path of the exported .pt artifact.
        map_location: torch.load map_location parameter.

    Returns:
        Complex tensor [V, M, N] ready for receiver TokenCodewordPrior.

    Raises:
        ValueError: If payload fails strict contract validation.
    """
    payload = load_exported_physical_dd_artifact(
        path, map_location=map_location,
    )
    return payload["codeword_book"]


def load_exported_physical_dd_config(
    path: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
) -> TransmitterConfig:
    """Load a strict physical artifact and reconstruct its TX configuration."""

    payload = load_exported_physical_dd_artifact(
        path,
        map_location=map_location,
    )
    return _config_from_metadata(payload["metadata"])


# ---- private helpers -------------------------------------------------------


def _resolve_codebook(
    transmitter: LearnableTokenDDTransmitter | TokenDDCodebook,
) -> tuple[TransmitterConfig, TokenDDCodebook]:
    if isinstance(transmitter, LearnableTokenDDTransmitter):
        return transmitter.config, transmitter.codebook
    if isinstance(transmitter, TokenDDCodebook):
        return transmitter.config, transmitter
    raise TypeError(
        "transmitter must be a LearnableTokenDDTransmitter or "
        f"TokenDDCodebook, got {type(transmitter).__name__}."
    )


def _resolve_export_data_mask(
    data_mask: torch.Tensor,
    config: TransmitterConfig,
) -> torch.Tensor:
    """Resolve data_mask to a bool CPU tensor for export."""
    if data_mask.dtype == torch.bool:
        mask = data_mask.detach().cpu()
    else:
        V = config.vocab_size
        M = config.M
        N = config.N
        ref_shape = (V, M, N)
        # Validate as binary support mask (raises on fractional values)
        float_mask = _validate_binary_support_mask(
            data_mask, ref_shape, "data_mask",
            device=torch.device("cpu"), dtype=torch.float32,
        )
        mask = (float_mask > 0).cpu()
    return mask


def _validate_export_mask_subset_of_data_region(
    data_mask: torch.Tensor,
    config: TransmitterConfig,
) -> None:
    """Validate explicit data_mask is a hard binary subset of config data region.

    Every True/1 position in data_mask must lie within the config-defined
    data region. Pilot and guard positions must all be 0/False.

    Args:
        data_mask: Hard binary support mask, bool or real 0/1, shape
            [1, M, N] or [V, M, N].
        config: TransmitterConfig defining the no-wrap rectangular layout.

    Raises:
        TypeError: If data_mask is not a tensor or is complex.
        ValueError: If shape is invalid, mask is not hard binary, or any
            pilot/guard/outside-data position is non-zero.
    """
    if not torch.is_tensor(data_mask):
        raise TypeError(
            f"data_mask must be a torch.Tensor, got {type(data_mask).__name__}."
        )
    if torch.is_complex(data_mask):
        raise TypeError("data_mask must not be a complex tensor.")

    M = config.M
    N = config.N
    V = config.vocab_size

    if data_mask.ndim != 3:
        raise ValueError(
            f"data_mask must have 3 dimensions, got ndim={data_mask.ndim}."
        )
    if data_mask.shape not in ((1, M, N), (V, M, N)):
        raise ValueError(
            f"data_mask must have shape [1, {M}, {N}] or [{V}, {M}, {N}], "
            f"got {list(data_mask.shape)}."
        )

    # Convert to bool CPU for checks.
    if data_mask.dtype == torch.bool:
        mask_bool = data_mask.detach().cpu()
    else:
        mask_float = data_mask.detach().to(
            device=torch.device("cpu"), copy=True,
        )
        if not torch.isfinite(mask_float).all():
            raise ValueError("data_mask must contain only finite values.")
        invalid = (mask_float != 0) & (mask_float != 1)
        if invalid.any():
            raise ValueError(
                "data_mask must be a hard binary support mask "
                "(only 0 or 1 values). Fractional values are not allowed."
            )
        mask_bool = mask_float > 0

    config_masks = build_transmitter_pilot_masks(config)
    cfg_pilot = config_masks.pilot_mask[0]   # [M, N]
    cfg_guard = config_masks.guard_mask[0]   # [M, N]
    cfg_data = config_masks.data_mask[0]     # [M, N]

    num_tokens = mask_bool.shape[0]
    for t in range(num_tokens):
        token_mask = mask_bool[t]  # [M, N]

        # Pilot positions must be 0.
        if token_mask[cfg_pilot].any():
            raise ValueError(
                f"data_mask token {t} has pilot position(s) set to 1. "
                "Pilot positions must be 0 in data_mask."
            )

        # Guard positions must be 0.
        if token_mask[cfg_guard].any():
            raise ValueError(
                f"data_mask token {t} has guard position(s) set to 1. "
                "Guard positions must be 0 in data_mask."
            )

        # Must be a subset of config data region.
        outside_data = token_mask & ~cfg_data
        if outside_data.any():
            raise ValueError(
                f"data_mask token {t} has position(s) outside the "
                "config-defined data region. "
                "data_mask must be a subset of the config data region."
            )


def _validate_physical_export_constraints(
    codeword_book: torch.Tensor,
    data_mask: torch.Tensor,
    config: TransmitterConfig,
) -> None:
    """Run physical consistency checks before saving.

    Pilot/guard zero checks are always performed; there is no bypass path.

    Raises ValueError if any constraint is violated.
    """
    if not torch.is_tensor(codeword_book):
        raise ValueError("codeword_book must be a torch.Tensor.")
    if not torch.is_complex(codeword_book):
        raise ValueError("codeword_book must be a complex tensor.")
    if not torch.isfinite(codeword_book).all():
        raise ValueError(
            "codeword_book must contain only finite values "
            "(real and imag parts must both be finite)."
        )

    V, M_exp, N_exp = codeword_book.shape
    if V != config.vocab_size or M_exp != config.M or N_exp != config.N:
        raise ValueError(
            f"codeword_book shape [{V}, {M_exp}, {N_exp}] does not match "
            f"config [vocab={config.vocab_size}, M={config.M}, N={config.N}]."
        )

    from .metrics import data_codeword_power as _dcp
    dcp = _dcp(codeword_book, data_mask)
    target = config.data_power
    for v in range(V):
        val = dcp[v].item()
        if not (math.isclose(val, target, rel_tol=1e-4, abs_tol=1e-5)):
            raise ValueError(
                f"data-region power for token {v} is {val:.6f}, "
                f"expected {target}. Physical normalization check failed."
            )

    # Always check pilot/guard zero -- no bypass.
    # Uses .any() instead of .max() so empty guard regions
    # (pilot_guard_delay=0, pilot_guard_doppler=0) do not
    # raise RuntimeError on max() of an empty tensor.
    masks = build_transmitter_pilot_masks(config)
    pilot_2d = masks.pilot_mask[0]
    guard_2d = masks.guard_mask[0]
    for v in range(V):
        if (codeword_book[v, pilot_2d].abs() >= 1e-6).any():
            raise ValueError(
                f"codeword_book token {v} has non-zero pilot region."
            )
        if (codeword_book[v, guard_2d].abs() >= 1e-6).any():
            raise ValueError(
                f"codeword_book token {v} has non-zero guard region."
            )


def _validate_loaded_codeword_book(codeword_book: torch.Tensor) -> None:
    if not torch.is_tensor(codeword_book):
        raise ValueError("codeword_book must be a torch.Tensor.")
    if not torch.is_complex(codeword_book):
        raise ValueError("codeword_book must be a complex tensor.")
    if codeword_book.ndim != 3:
        raise ValueError(
            f"codeword_book must have 3 dimensions [V, M, N], "
            f"got ndim={codeword_book.ndim}."
        )
    if codeword_book.shape[0] <= 0 or codeword_book.shape[1] <= 0 or codeword_book.shape[2] <= 0:
        raise ValueError(
            f"codeword_book dimensions must be > 0, got {list(codeword_book.shape)}."
        )


def _validate_physical_export_metadata(metadata: dict) -> None:
    artifact = metadata.get("artifact_type", None)
    if artifact is not None and artifact != "physical_codeword_book":
        raise ValueError(
            f"metadata artifact_type is '{artifact}', expected "
            "'physical_codeword_book'. This file is not a valid physical "
            "codeword_book export."
        )
    if metadata.get("raw_tx_state_dict", None) is True:
        raise ValueError(
            "metadata raw_tx_state_dict is True; this is a raw state file, "
            "not a physical codeword_book export."
        )
    if metadata.get("physical_codeword_book_exported", None) is False:
        raise ValueError(
            "metadata physical_codeword_book_exported is False; "
            "this export is invalid."
        )
    if metadata.get("receiver_prior_ready", None) is False:
        raise ValueError(
            "metadata receiver_prior_ready is False; "
            "this export is not receiver-prior-ready."
        )
    # Contract fields.
    if metadata.get("dd_axis_order") != ["delay", "doppler"]:
        raise ValueError(
            "metadata dd_axis_order must be ['delay', 'doppler']."
        )
    if metadata.get("positive_shift_semantics") != "torch.roll":
        raise ValueError(
            "metadata positive_shift_semantics must be 'torch.roll'."
        )
    if metadata.get("pilot_wrap_around", None) is not False:
        raise ValueError("metadata pilot_wrap_around must be False.")


def _config_from_metadata(metadata: dict) -> TransmitterConfig:
    """Reconstruct TransmitterConfig from physical export metadata."""
    required_keys = (
        "M", "N", "vocab_size", "data_power", "complex_dtype",
        "pilot_value_real", "pilot_value_imag",
        "pilot_delay", "pilot_doppler", "pilot_guard_delay",
        "pilot_guard_doppler", "pilot_obs_delay_radius",
        "pilot_obs_doppler_radius", "max_channel_delay",
        "max_channel_doppler",
    )
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"metadata missing '{key}'.")
    return TransmitterConfig(
        M=metadata["M"],
        N=metadata["N"],
        vocab_size=metadata["vocab_size"],
        data_power=metadata["data_power"],
        complex_dtype=metadata["complex_dtype"],
        pilot_value_real=metadata["pilot_value_real"],
        pilot_value_imag=metadata["pilot_value_imag"],
        pilot_delay=metadata["pilot_delay"],
        pilot_doppler=metadata["pilot_doppler"],
        pilot_guard_delay=metadata["pilot_guard_delay"],
        pilot_guard_doppler=metadata["pilot_guard_doppler"],
        pilot_obs_delay_radius=metadata["pilot_obs_delay_radius"],
        pilot_obs_doppler_radius=metadata["pilot_obs_doppler_radius"],
        max_channel_delay=metadata["max_channel_delay"],
        max_channel_doppler=metadata["max_channel_doppler"],
    )


def _canonical_hash_bytes(
    codeword_book: torch.Tensor,
    data_mask: torch.Tensor,
    metadata: dict,
) -> bytes:
    """Produce canonical byte sequence for SHA-256 hashing.

    Includes codeword_book real/imag as contiguous native-dtype CPU bytes,
    data_mask as uint8 CPU bytes, and JSON-serialized metadata without
    the artifact_sha256 key.
    """
    import json
    # Codeword book: contiguous real + imag parts.
    cw_real = codeword_book.real.detach().cpu().contiguous()
    cw_imag = codeword_book.imag.detach().cpu().contiguous()
    cw_bytes = cw_real.numpy().tobytes() + cw_imag.numpy().tobytes()

    # Data mask: as uint8 CPU bytes.
    dm = data_mask.to(device=torch.device("cpu"),
                      dtype=torch.uint8).contiguous()
    dm_bytes = dm.numpy().tobytes()

    # Metadata: sorted JSON without artifact_sha256.
    meta_copy = {k: v for k, v in metadata.items()
                 if k != "artifact_sha256"}
    meta_json = json.dumps(meta_copy, sort_keys=True, ensure_ascii=True)
    meta_bytes = meta_json.encode("ascii")

    return cw_bytes + dm_bytes + meta_bytes


def _config_metadata(config: TransmitterConfig) -> dict:
    return {
        "M": config.M,
        "N": config.N,
        "vocab_size": config.vocab_size,
        "data_power": config.data_power,
        "complex_dtype": config.complex_dtype,
        "pilot_value_real": config.pilot_value_real,
        "pilot_value_imag": config.pilot_value_imag,
        "pilot_delay": config.pilot_delay,
        "pilot_doppler": config.pilot_doppler,
        "pilot_guard_delay": config.pilot_guard_delay,
        "pilot_guard_doppler": config.pilot_guard_doppler,
        "pilot_obs_delay_radius": config.pilot_obs_delay_radius,
        "pilot_obs_doppler_radius": config.pilot_obs_doppler_radius,
        "max_channel_delay": config.max_channel_delay,
        "max_channel_doppler": config.max_channel_doppler,
    }


def _physical_export_metadata(config: TransmitterConfig) -> dict:
    meta = _config_metadata(config)
    meta.update({
        "schema_version": PHYSICAL_CODEWORD_BOOK_SCHEMA_VERSION,
        "artifact_type": "physical_codeword_book",
        "physical_codeword_book_exported": True,
        "hard_data_mask_projection": True,
        "receiver_prior_ready": True,
        "raw_tx_state_dict": False,
        "dd_axis_order": ["delay", "doppler"],
        "positive_shift_semantics": "torch.roll",
        "pilot_wrap_around": False,
        "codeword_is_finite": True,
    })
    return meta


def _raw_state_metadata(config: TransmitterConfig) -> dict:
    meta = _config_metadata(config)
    meta.update({
        "artifact_type": "raw_tx_state",
        "raw_tx_state_dict": True,
        "physical_codeword_book_exported": False,
        "receiver_prior_ready": False,
    })
    return meta


def _to_cpu_state_dict(state_dict: dict) -> dict:
    out: dict = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu()
        else:
            out[key] = value
    return out
