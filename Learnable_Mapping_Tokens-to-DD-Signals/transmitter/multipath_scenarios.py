"""Auditable fixed sparse multipath scenario bank.

Public API:
    SparseMultipathScenarioBank  -- frozen dataclass for R scenarios
    sample_normalized_sparse_multipath_scenario_bank(source_shift_set,
        num_scenarios, num_paths, ...) -> SparseMultipathScenarioBank

Scenarios are fixed, non-learnable, and auditable. Each scenario r has
K path slots with unit active-path-power normalization:
    sum_k active[r,k] * |h[r,k]|^2 == 1

This isolates codeword shaping from overall SNR scaling.  The bank
is a normalized sparse fading direction collection, NOT a complete
channel statistical model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .shaping import SparseShiftSet
from .sparse_multipath import SparseMultipathDDChannel


@dataclass(frozen=True)
class SparseMultipathScenarioBank:
    """Fixed sparse multipath scenario bank.

    Each of the R scenarios has K padded path slots.  Gains are fixed
    (non-learnable).  Active path energy is normalized to 1 per scenario.
    source_shift_indices tracks the origin of each sampled shift for
    auditability.

    Attributes:
        channel: SparseMultipathDDChannel with .path_shifts long [R, K, 2],
            .path_gains complex [R, K], .path_active_mask optional bool [R, K].
            gains.requires_grad must be False.
        scenario_weights: Optional real [R] of scenario sampling weights.
            If None, downstream uses uniform weights.
        source_shift_indices: Optional long [R, K] tracking shift provenance.
        name: Non-empty identifier string.
    """

    channel: SparseMultipathDDChannel
    scenario_weights: torch.Tensor | None = None
    source_shift_indices: torch.Tensor | None = None
    name: str = "sparse_multipath_scenarios"

    def __post_init__(self) -> None:
        ch = self.channel
        if not isinstance(ch, SparseMultipathDDChannel):
            raise TypeError(
                f"channel must be a SparseMultipathDDChannel, "
                f"got {type(ch).__name__}."
            )
        # shape
        if ch.path_shifts.ndim != 3 or ch.path_shifts.shape[2] != 2:
            raise ValueError(
                f"channel.path_shifts must have shape [R, K, 2], "
                f"got {list(ch.path_shifts.shape)}."
            )
        R, K, _ = ch.path_shifts.shape
        if R <= 0:
            raise ValueError(f"R (num scenarios) must be > 0, got R={R}.")
        if K <= 0:
            raise ValueError(f"K (num paths) must be > 0, got K={K}.")
        if ch.path_gains.shape != (R, K):
            raise ValueError(
                f"channel.path_gains must have shape [{R}, {K}], "
                f"got {list(ch.path_gains.shape)}."
            )

        # Gains must be non-learnable.
        if ch.path_gains.requires_grad:
            raise ValueError(
                "channel.path_gains.requires_grad must be False. "
                "Scenario gains are fixed, non-learnable."
            )
        if not torch.isfinite(ch.path_gains).all():
            raise ValueError("channel.path_gains must be finite.")

        # Active mask
        if ch.path_active_mask is not None:
            mask = ch.path_active_mask
            if mask.dtype != torch.bool:
                raise TypeError(
                    "channel.path_active_mask must be bool tensor."
                )
            if mask.shape != (R, K):
                raise ValueError(
                    f"channel.path_active_mask must have shape [{R}, {K}], "
                    f"got {list(mask.shape)}."
                )

        # Per-scenario active-path energy normalization.
        _validate_per_scenario_energy(ch)

        # scenario_weights
        sw = self.scenario_weights
        if sw is not None:
            if not torch.is_tensor(sw):
                raise TypeError(
                    f"scenario_weights must be a torch.Tensor, "
                    f"got {type(sw).__name__}."
                )
            if not sw.dtype.is_floating_point:
                raise TypeError(
                    f"scenario_weights must be real floating, "
                    f"got dtype {sw.dtype}."
                )
            if sw.shape != (R,):
                raise ValueError(
                    f"scenario_weights must have shape [{R}], "
                    f"got {list(sw.shape)}."
                )
            if not torch.isfinite(sw).all():
                raise ValueError("scenario_weights must be finite.")
            if (sw < 0).any():
                raise ValueError("scenario_weights must be non-negative.")
            if sw.sum().item() <= 0:
                raise ValueError("scenario_weights sum must be > 0.")
            if sw.requires_grad:
                raise ValueError(
                    "scenario_weights.requires_grad must be False."
                )

        # source_shift_indices
        ssi = self.source_shift_indices
        if ssi is not None:
            if not torch.is_tensor(ssi):
                raise TypeError(
                    f"source_shift_indices must be a torch.Tensor, "
                    f"got {type(ssi).__name__}."
                )
            if ssi.dtype != torch.long:
                raise TypeError(
                    f"source_shift_indices dtype must be torch.long, "
                    f"got {ssi.dtype}."
                )
            if ssi.shape != (R, K):
                raise ValueError(
                    f"source_shift_indices must have shape [{R}, {K}], "
                    f"got {list(ssi.shape)}."
                )
            if (ssi < 0).any():
                raise ValueError("source_shift_indices must be >= 0.")
            if ssi.device != ch.path_shifts.device:
                raise ValueError(
                    f"source_shift_indices device ({ssi.device}) must "
                    f"match path_shifts device ({ch.path_shifts.device})."
                )

        # name
        if not isinstance(self.name, str):
            raise TypeError(
                f"name must be str, got {type(self.name).__name__}."
            )
        if len(self.name.strip()) == 0:
            raise ValueError("name must be a non-empty string.")

    def normalized_weights(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Normalized scenario weights [R] summing to 1.

        Args:
            device: Optional output device (default: source device or CPU).
            dtype: Floating output dtype. Non-floating dtypes rejected.

        Returns:
            Real tensor [R] of normalized weights.
        """
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError(
                f"dtype must be a floating type, got {dtype}."
            )
        if self.scenario_weights is None:
            R = self.channel.path_shifts.shape[0]
            resolved_dev = (torch.device(device) if device is not None
                            else torch.device("cpu"))
            return torch.full((R,), 1.0 / R,
                              device=resolved_dev, dtype=dtype)
        w = self.scenario_weights.to(
            device=torch.device(device) if device is not None
            else self.scenario_weights.device,
            dtype=dtype,
        )
        return w / w.sum()

    def materialize_channel(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> SparseMultipathDDChannel:
        """Materialize a read-only channel copy on the requested device/dtype.

        Args:
            device: Target device.
            dtype: Complex dtype (torch.complex64 or torch.complex128 only).

        Returns:
            New SparseMultipathDDChannel with gains.requires_grad=False.
            path_shifts is long [R, K, 2], path_gains is complex [R, K],
            and optional path_active_mask is bool [R, K]. The source bank
            is not modified.
        """
        if dtype not in (torch.complex64, torch.complex128):
            raise TypeError(
                f"dtype must be torch.complex64 or torch.complex128, "
                f"got {dtype}."
            )
        dev = torch.device(device)
        ch = self.channel
        shifts = ch.path_shifts.to(device=dev).clone()
        gains = ch.path_gains.to(device=dev, dtype=dtype).clone()
        mask = (ch.path_active_mask.to(device=dev).clone()
                if ch.path_active_mask is not None else None)
        return SparseMultipathDDChannel(
            path_shifts=shifts,
            path_gains=gains,
            path_active_mask=mask,
        )


def sample_normalized_sparse_multipath_scenario_bank(
    source_shift_set: SparseShiftSet,
    num_scenarios: int,
    num_paths: int,
    *,
    generator: torch.Generator | None = None,
    complex_dtype: torch.dtype = torch.complex64,
    name: str = "sampled_sparse_multipath_scenarios",
    unique_shifts_per_scenario: bool = False,
) -> SparseMultipathScenarioBank:
    """Sample a fixed non-learnable sparse multipath scenario bank.

    1. Sample shift source indices from the weighted shift distribution q[s]
       using torch.multinomial. If unique_shifts_per_scenario=True, sample
       distinct DD bins within each scenario without replacement.
    2. Look up sampled shifts from source_shift_set.shifts.
    3. Sample i.i.d. complex Gaussian gains and normalize each scenario:
          h[r] = g_raw[r] / ||g_raw[r]||_2
       so that sum_k |h[r,k]|^2 == 1.

    q is used ONLY during shift index sampling.  scenario_weights is
    returned as None, avoiding double weighting in downstream objectives.

    Args:
        source_shift_set: SparseShiftSet with .shifts long [S, 2].
        num_scenarios: R > 0 (int, not bool).
        num_paths: K > 0 (int, not bool).
        generator: Optional CPU torch.Generator.
        complex_dtype: torch.complex64 or torch.complex128.
        name: Non-empty scenario bank identifier.
        unique_shifts_per_scenario: If True, require source DD bins to be
            unique and sample K distinct bins within each scenario. This is
            appropriate when path slots represent effective on-grid DD taps.

    Returns:
        SparseMultipathScenarioBank (canonical CPU, R scenarios each with
        K paths and real sampling weight per scenario):
            channel.path_shifts    long   [R, K, 2]
            channel.path_gains     complex [R, K]
            channel.path_active_mask  bool [R, K] (all True)
            source_shift_indices   long   [R, K]
            scenario_weights       None (uniform real [R] implied)
    """
    # Validate inputs.
    if not isinstance(source_shift_set, SparseShiftSet):
        raise TypeError(
            f"source_shift_set must be a SparseShiftSet, "
            f"got {type(source_shift_set).__name__}."
        )
    _validate_sampling_args(source_shift_set, num_scenarios, num_paths,
                            complex_dtype, name)
    if not isinstance(unique_shifts_per_scenario, bool):
        raise TypeError(
            "unique_shifts_per_scenario must be bool, "
            f"got {type(unique_shifts_per_scenario).__name__}."
        )
    if generator is not None:
        _validate_cpu_generator(generator)

    R = num_scenarios
    K = num_paths
    rdtype = {torch.complex64: torch.float32,
              torch.complex128: torch.float64}[complex_dtype]

    # --- shift source indices ---
    q_cpu = source_shift_set.normalized_weights(
        device=torch.device("cpu"), dtype=torch.float64,
    )
    # Re-check q for post-construction corruption.
    if not torch.isfinite(q_cpu).all():
        raise ValueError(
            "source_shift_set normalized weights must be finite."
        )
    if (q_cpu < 0).any():
        raise ValueError(
            "source_shift_set normalized weights must be non-negative."
        )
    if q_cpu.sum().item() <= 0:
        raise ValueError(
            "source_shift_set normalized weights sum must be > 0."
        )

    if unique_shifts_per_scenario:
        source_shifts_cpu = source_shift_set.shifts.detach().to(
            device=torch.device("cpu"), dtype=torch.long, copy=True,
        )
        if torch.unique(source_shifts_cpu, dim=0).shape[0] != \
                source_shifts_cpu.shape[0]:
            raise ValueError(
                "unique_shifts_per_scenario=True requires source_shift_set "
                "to contain unique DD bins."
            )
        positive_support = int((q_cpu > 0).sum().item())
        if K > positive_support:
            raise ValueError(
                "unique_shifts_per_scenario=True requires num_paths <= "
                f"positive-weight DD bins, got {K} > {positive_support}."
            )
        source_shift_indices = torch.multinomial(
            q_cpu.expand(R, -1), K, replacement=False, generator=generator,
        )  # long [R, K]
    else:
        flat_indices = torch.multinomial(
            q_cpu, R * K, replacement=True, generator=generator,
        )  # long [R*K]
        source_shift_indices = flat_indices.reshape(R, K)  # [R, K]

    source_shifts_cpu = source_shift_set.shifts.detach().to(
        device=torch.device("cpu"), dtype=torch.long, copy=True,
    )
    path_shifts = source_shifts_cpu[source_shift_indices]  # [R, K, 2]

    # --- sample complex Gaussian gains ---
    real_part = torch.randn(R, K, generator=generator, dtype=rdtype)
    imag_part = torch.randn(R, K, generator=generator, dtype=rdtype)
    g_raw = torch.complex(real_part, imag_part)

    # Per-scenario normalization: h[r] = g[r] / ||g[r]||_2
    g_norm = g_raw.abs().pow(2).sum(dim=1).sqrt()  # [R]
    if bool((g_norm <= 0).any().item()):
        raise ValueError(
            "Sampled raw gain norm <= 0 for at least one scenario. "
            "Retry with a different seed or increase num_paths."
        )
    path_gains = g_raw / g_norm.unsqueeze(1)  # [R, K]

    channel = SparseMultipathDDChannel(
        path_shifts=path_shifts,
        path_gains=path_gains,
        path_active_mask=torch.ones(R, K, dtype=torch.bool),
    )

    return SparseMultipathScenarioBank(
        channel=channel,
        scenario_weights=None,
        source_shift_indices=source_shift_indices,
        name=name,
    )


# -- private helpers -----------------------------------------------------------


def _validate_per_scenario_energy(
    ch: SparseMultipathDDChannel,
) -> None:
    """Verify sum_k active[r,k] * |h[r,k]|^2 == 1 for every scenario."""
    R, K, _ = ch.path_shifts.shape
    energy = ch.path_gains.abs().pow(2)  # [R, K]
    if ch.path_active_mask is not None:
        energy = energy * ch.path_active_mask.to(
            device=energy.device, dtype=energy.real.dtype,
        )
    active_energy = energy.sum(dim=1)  # [R]

    if not torch.isfinite(active_energy).all():
        raise ValueError(
            "Active path energy must be finite for all scenarios."
        )
    if bool((active_energy <= 0).any().item()):
        raise ValueError(
            "Active path energy must be > 0 for every scenario. "
            "Each scenario must have at least one active path."
        )
    # Tolerance depends on dtype.
    if ch.path_gains.dtype == torch.complex64:
        atol, rtol = 1e-5, 1e-5
    else:
        atol, rtol = 1e-10, 1e-10

    if not torch.allclose(active_energy, torch.ones_like(active_energy),
                          atol=atol, rtol=rtol):
        raise ValueError(
            "Active path energy must be 1 per scenario "
            "(unit active-path-power normalization)."
        )


def _validate_sampling_args(
    source_shift_set: SparseShiftSet,
    num_scenarios: int,
    num_paths: int,
    complex_dtype: torch.dtype,
    name: str,
) -> None:
    """Validate inputs to the sampling function."""
    ss = source_shift_set
    if not torch.is_tensor(ss.shifts):
        raise TypeError(
            "source_shift_set.shifts must be a torch.Tensor, "
            f"got {type(ss.shifts).__name__}."
        )
    if ss.shifts.dtype != torch.long:
        raise TypeError(
            f"source_shift_set.shifts dtype must be torch.long, "
            f"got {ss.shifts.dtype}."
        )
    if ss.shifts.ndim != 2 or ss.shifts.shape[1] != 2:
        raise ValueError(
            f"source_shift_set.shifts must have shape [S, 2], "
            f"got {list(ss.shifts.shape)}."
        )
    if ss.shifts.shape[0] <= 0:
        raise ValueError(
            "source_shift_set.shifts S must be > 0."
        )

    if ss.weights is not None:
        w = ss.weights
        if not torch.is_tensor(w):
            raise TypeError(
                f"source_shift_set.weights must be a torch.Tensor, "
                f"got {type(w).__name__}."
            )
        if not w.dtype.is_floating_point:
            raise TypeError(
                f"source_shift_set.weights must be real floating, "
                f"got dtype {w.dtype}."
            )
        if w.shape != (ss.shifts.shape[0],):
            raise ValueError(
                f"source_shift_set.weights must have shape "
                f"[S={ss.shifts.shape[0]}], got {list(w.shape)}."
            )
        if not torch.isfinite(w).all():
            raise ValueError("source_shift_set.weights must be finite.")
        if (w < 0).any():
            raise ValueError(
                "source_shift_set.weights must be non-negative."
            )
        if w.sum().item() <= 0:
            raise ValueError(
                "source_shift_set.weights sum must be > 0."
            )
        if w.requires_grad:
            raise ValueError(
                "source_shift_set.weights.requires_grad must be False. "
                "Shift weights are fixed physical statistics."
            )

    if isinstance(num_scenarios, bool) or not isinstance(num_scenarios, int):
        raise TypeError(
            f"num_scenarios must be int, "
            f"got {type(num_scenarios).__name__}."
        )
    if num_scenarios <= 0:
        raise ValueError(
            f"num_scenarios must be > 0, got {num_scenarios}."
        )
    if isinstance(num_paths, bool) or not isinstance(num_paths, int):
        raise TypeError(
            f"num_paths must be int, "
            f"got {type(num_paths).__name__}."
        )
    if num_paths <= 0:
        raise ValueError(
            f"num_paths must be > 0, got {num_paths}."
        )
    if complex_dtype not in (torch.complex64, torch.complex128):
        raise TypeError(
            f"complex_dtype must be torch.complex64 or torch.complex128, "
            f"got {complex_dtype}."
        )
    if not isinstance(name, str) or len(name.strip()) == 0:
        raise ValueError(
            f"name must be a non-empty str, got {name!r}."
        )


def _validate_cpu_generator(generator: torch.Generator) -> None:
    if not isinstance(generator, torch.Generator):
        raise TypeError(
            f"generator must be a torch.Generator, "
            f"got {type(generator).__name__}."
        )
    if generator.device.type != "cpu":
        raise ValueError(
            "generator must be a CPU torch.Generator, "
            f"got device {generator.device}."
        )
