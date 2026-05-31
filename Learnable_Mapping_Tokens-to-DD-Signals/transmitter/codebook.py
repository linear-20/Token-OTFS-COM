"""Token DD codebook: raw learnable params -> physical normalized codeword book.

Raw learnable parameters have shape [V, M, N]. The forward pass projects them
onto a hard binary data support mask and applies equal-power normalization so
the receiver prior can consume the resulting physical codeword_book [V, M, N].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import TransmitterConfig


class TokenDDCodebook(nn.Module):
    """Raw learnable DD token embeddings normalized to physical codewords.

    Raw params: real [V, M, N] (raw_real, raw_imag).
    Physical codeword_book: complex [V, M, N], returned by forward().
    Selected codewords: complex [B, M, N], returned by selected_codewords().
    Mask: bool or real-valued binary support mask [1, M, N] or [V, M, N].
        This is a hard support mask, NOT a soft/reliability mask.

    The forward() output is the physical normalized codeword_book that the
    receiver TokenCodewordPrior should consume. It does NOT insert pilots,
    compute PAPR, or implement shaping loss.
    """

    def __init__(
        self,
        config: TransmitterConfig,
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        _validate_init_scale(init_scale)
        self.config = config
        V = config.vocab_size
        M = config.M
        N = config.N
        real_dtype = config.torch_real_dtype

        self.raw_real = nn.Parameter(torch.randn(V, M, N, dtype=real_dtype) * init_scale)
        self.raw_imag = nn.Parameter(torch.randn(V, M, N, dtype=real_dtype) * init_scale)

    def forward(
        self,
        data_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalize raw params into physical complex codeword book [V, M, N].

        Args:
            data_mask: Optional hard binary support mask (bool or real-valued
                0/1) with shape [1, M, N] or [V, M, N]. Fractional values
                are rejected. If None, a grid-wide all-ones mask [1, M, N]
                is used.

        Returns:
            Physical complex codeword_book [V, M, N] with per-token
            equal-power normalization over its active data region. The
            average active-region power equals config.data_power.

        Raises:
            TypeError: If data_mask is complex.
            ValueError: If data_mask shape, spatial extent, finiteness,
                non-binary values, or active-region emptiness / zero-power
                are violated.
        """
        raw_codeword = torch.complex(self.raw_real, self.raw_imag)

        if data_mask is None:
            mask = torch.ones(
                1, self.config.M, self.config.N,
                device=raw_codeword.device,
                dtype=self.config.torch_real_dtype,
            )
        else:
            mask = _prepare_data_mask(data_mask, self.config, raw_codeword)

        projected = raw_codeword * mask

        active = mask > 0
        active_count = active.sum(dim=(-2, -1))
        if (active_count <= 0).any():
            raise ValueError(
                "data_mask has no active bins for at least one token. "
                "Each token must have at least one active data-region bin."
            )

        active_power_sum = projected.abs().pow(2).sum(dim=(-2, -1))
        power = active_power_sum / active_count.clamp_min(1)
        if (power <= 0).any():
            raise ValueError(
                "Active-region power is zero for at least one token. "
                "Ensure raw params produce non-zero signal in the active region."
            )

        scale = math.sqrt(self.config.data_power) / power.sqrt()
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        codeword_book = projected * scale
        return codeword_book.to(dtype=self.config.torch_complex_dtype)

    def selected_codewords(
        self,
        token_indices: torch.Tensor,
        codeword_book: torch.Tensor | None = None,
        data_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Look up selected codewords by token indices.

        Args:
            token_indices: Long tensor [B] of token indices in [0, V).
            codeword_book: Optional complex tensor [V, M, N]. If None,
                self.forward(data_mask=data_mask) is called.
            data_mask: Optional mask passed to forward() when codeword_book
                is None.

        Returns:
            Complex tensor [B, M, N] of selected codewords in token_indices
            order.

        Raises:
            TypeError: If token_indices is not a long tensor, or if
                codeword_book is not a complex tensor.
            ValueError: If shapes, batch size, or index range are invalid.
        """
        _validate_token_indices(token_indices, self.config.vocab_size)

        if codeword_book is None:
            codeword_book = self.forward(data_mask=data_mask)
        else:
            _validate_codeword_book(codeword_book, self.config)

        return codeword_book[token_indices]


def _validate_init_scale(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"init_scale must be a number, got {type(value).__name__}.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"init_scale must be a finite positive scalar, got {value}.")


def _prepare_data_mask(
    data_mask: torch.Tensor,
    config: TransmitterConfig,
    raw_codeword: torch.Tensor,
) -> torch.Tensor:
    """Validate and prepare data_mask for forward().

    data_mask must be a hard binary support mask (bool or real-valued 0/1).
    Fractional or out-of-binary values are rejected. This is NOT a soft
    reliability mask.

    Args:
        data_mask: Input mask tensor (bool or real-valued binary 0/1).
        config: TransmitterConfig for spatial reference.
        raw_codeword: Complex raw codeword [V, M, N] for device/dtype.

    Returns:
        Real 0/1 mask tensor on raw_codeword device and config real dtype.

    Raises:
        TypeError: If data_mask is complex.
        ValueError: If shape, finiteness, or non-binary values are detected.
    """
    if not torch.is_tensor(data_mask):
        raise TypeError(f"data_mask must be a torch.Tensor, got {type(data_mask).__name__}.")
    if torch.is_complex(data_mask):
        raise TypeError("data_mask must not be a complex tensor.")

    if data_mask.ndim != 3:
        raise ValueError(f"data_mask must have 3 dimensions, got ndim={data_mask.ndim}.")
    V = config.vocab_size
    M = config.M
    N = config.N
    if data_mask.shape not in ((1, M, N), (V, M, N)):
        raise ValueError(
            f"data_mask must have shape [1, {M}, {N}] or [{V}, {M}, {N}], "
            f"got {list(data_mask.shape)}."
        )

    mask_check = data_mask.detach().to(
        device=torch.device("cpu"), copy=True,
    )
    if not torch.isfinite(mask_check).all():
        raise ValueError("data_mask must contain only finite values.")
    invalid = (mask_check != 0) & (mask_check != 1)
    if invalid.any():
        raise ValueError(
            "data_mask must be a hard binary support mask (only 0 or 1 values). "
            "Fractional or out-of-binary values are not allowed."
        )
    mask = data_mask.to(device=raw_codeword.device, dtype=config.torch_real_dtype)
    return mask


def _validate_token_indices(token_indices: torch.Tensor, V: int) -> None:
    if not torch.is_tensor(token_indices):
        raise TypeError(
            f"token_indices must be a torch.Tensor, got {type(token_indices).__name__}."
        )
    if token_indices.dtype not in (torch.long, torch.int64):
        raise TypeError(
            f"token_indices dtype must be torch.long, got {token_indices.dtype}."
        )
    if token_indices.ndim != 1:
        raise ValueError(
            f"token_indices must be 1-D [B], got ndim={token_indices.ndim}."
        )
    B = token_indices.shape[0]
    if B <= 0:
        raise ValueError(f"token_indices batch size must be positive, got B={B}.")
    if (token_indices < 0).any() or (token_indices >= V).any():
        raise ValueError(f"token_indices must be in [0, V={V}).")


def initialize_token_codebook_(
    module: TokenDDCodebook,
    mode: str = "random_phase",
    init_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> TokenDDCodebook:
    """Initialize TokenDDCodebook raw params with DD-aware structured phases.

    This function writes module.raw_real and module.raw_imag in-place and
    returns the same module. It does NOT alter config, forward normalization,
    pilot insertion, or the physical codeword_book contract.

    Args:
        module: TokenDDCodebook whose raw params [V, M, N] will be initialized.
        mode: Initializer mode. Supported values: "random_phase",
            "separable_dft_phase", "chirp_phase".
        init_scale: Finite positive scalar controlling raw magnitude.
        generator: Optional torch.Generator for "random_phase" reproducibility.

    Returns:
        The same TokenDDCodebook module with updated raw_real / raw_imag.

    Raises:
        TypeError: If module is not a TokenDDCodebook, or init_scale is bool
            or non-number.
        ValueError: If mode is unsupported or init_scale is non-finite or
            non-positive.
    """
    if not isinstance(module, TokenDDCodebook):
        raise TypeError(
            f"module must be a TokenDDCodebook instance, got {type(module).__name__}."
        )
    if isinstance(init_scale, bool) or not isinstance(init_scale, (int, float)):
        raise TypeError(
            f"init_scale must be a number, got {type(init_scale).__name__}."
        )
    if not math.isfinite(init_scale) or init_scale <= 0:
        raise ValueError(
            f"init_scale must be a finite positive scalar, got {init_scale}."
        )

    SUPPORTED_MODES = {"random_phase", "separable_dft_phase", "chirp_phase"}
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"mode must be one of {sorted(SUPPORTED_MODES)}, got '{mode}'."
        )

    V = module.config.vocab_size
    M = module.config.M
    N = module.config.N
    device = module.raw_real.device
    dtype = module.raw_real.dtype

    with torch.no_grad():
        if mode == "random_phase":
            raw_real, raw_imag = _random_phase_init(V, M, N, init_scale, device, dtype, generator)
        elif mode == "separable_dft_phase":
            raw_real, raw_imag = _separable_dft_phase_init(V, M, N, init_scale, device, dtype)
        elif mode == "chirp_phase":
            raw_real, raw_imag = _chirp_phase_init(V, M, N, init_scale, device, dtype)
        else:
            raise ValueError(f"Unsupported mode: {mode}")  # pragma: no cover

        module.raw_real.copy_(raw_real)
        module.raw_imag.copy_(raw_imag)

    return module


def _random_phase_init(
    V: int,
    M: int,
    N: int,
    init_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    phase = torch.empty(V, M, N, device=device, dtype=dtype)
    phase.uniform_(0.0, 2.0 * math.pi, generator=generator)
    raw_real = init_scale * torch.cos(phase)
    raw_imag = init_scale * torch.sin(phase)
    return raw_real, raw_imag


def _separable_dft_phase_init(
    V: int,
    M: int,
    N: int,
    init_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    v = torch.arange(V, device=device)
    k_m = (v % M).to(device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)  # [V, 1, 1]
    k_n = ((v // M) % N).to(device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)  # [V, 1, 1]
    m = torch.arange(M, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)  # [1, M, 1]
    n = torch.arange(N, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, N]

    tpi = 2.0 * math.pi
    phase = tpi * (k_m * m / M + k_n * n / N)
    raw_real = init_scale * torch.cos(phase)
    raw_imag = init_scale * torch.sin(phase)
    return raw_real, raw_imag


def _chirp_phase_init(
    V: int,
    M: int,
    N: int,
    init_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    v = torch.arange(V, device=device)
    a = (v % M + 1).to(dtype=dtype).unsqueeze(-1).unsqueeze(-1)  # [V, 1, 1]
    b = ((v // M) % N + 1).to(dtype=dtype).unsqueeze(-1).unsqueeze(-1)  # [V, 1, 1]
    max_dim = max(M, N)
    c = ((v % max_dim) / max_dim).to(dtype=dtype).unsqueeze(-1).unsqueeze(-1)  # [V, 1, 1]

    m = torch.arange(M, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)  # [1, M, 1]
    n = torch.arange(N, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, N]

    m_norm = m / M
    n_norm = n / N
    tpi = 2.0 * math.pi

    phase = math.pi * (a * m_norm.pow(2) + b * n_norm.pow(2)) + tpi * c * (m_norm + n_norm)
    raw_real = init_scale * torch.cos(phase)
    raw_imag = init_scale * torch.sin(phase)
    return raw_real, raw_imag


def _validate_codeword_book(codeword_book: torch.Tensor, config: TransmitterConfig) -> None:
    if not torch.is_tensor(codeword_book):
        raise TypeError(
            f"codeword_book must be a torch.Tensor, got {type(codeword_book).__name__}."
        )
    if not torch.is_complex(codeword_book):
        raise TypeError(
            f"codeword_book must be a complex tensor, got dtype {codeword_book.dtype}."
        )
    V = config.vocab_size
    M = config.M
    N = config.N
    expected = (V, M, N)
    if codeword_book.shape != expected:
        raise ValueError(
            f"codeword_book must have shape {expected}, got {list(codeword_book.shape)}."
        )
