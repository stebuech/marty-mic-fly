"""Stage 2 — array deconvolution stage.

Internally the stage runs in four phases:

    1. _build_csm           — Welch CSM from time data.
    2. _build_fit_input     — search space (3D Cartesian grid today; future:
                              DOA hemisphere or known-atom set).
    3. _fit_powers          — algorithm fit (CLEAN-SC today; future: NNLS on
                              known atoms etc.) + trace rescale.
    4. _build_masks         — which cells/atoms to subtract; produces an
                              ``active`` boolean over the search space plus a
                              dict of named diagnostic masks for metadata.

The phase split exists so future variants (e.g. DOA grid, NNLS-on-atoms)
can override one phase each without touching the others.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np

from martymicfly.config import ArrayFilterStageConfig, BandConfig
from martymicfly.processing.algorithms import ALGORITHM_REGISTRY
from martymicfly.processing.algorithms.base import (
    SourceMap,
    reconstruct_csm,
    rescale_source_map_to_csm_trace,
)
from martymicfly.processing.beamform_grid import (
    build_diagnostic_grid,
    build_rotor_disc_mask,
    build_target_box_mask,
)
from martymicfly.processing.csm import CsmConfig as RuntimeCsmConfig, build_measurement_csm
from martymicfly.processing.pipeline import register_stage_builder
from martymicfly.processing.steering import steer_to_psd


@dataclass
class FitInput:
    """The search space + reshape hint passed to the algorithm.

    ``positions`` is (G, 3) regardless of whether G represents a Cartesian
    grid, a DOA hemisphere or a discrete known-atom set. ``reshape_hint``
    is an arbitrary tuple used by ``integrate_band_maps`` to fold the
    flat (G,) power vector back into a multi-dim heatmap; for non-grid
    search spaces (atom sets) it is the trivial ``(G,)``. ``aux`` is a
    free-form extension hook (e.g. atom-kind labels for Track B).
    """
    positions: np.ndarray
    reshape_hint: tuple[int, ...]
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaskBundle:
    """Bundle of named masks over the FitInput.

    ``active`` is the boolean (G,) actually subtracted from the source map;
    ``named`` is a dict of additional diagnostic masks to merge into stage
    metadata for downstream inspection.
    """
    active: np.ndarray
    named: dict[str, np.ndarray]


def integrate_band_maps(
    source_map: SourceMap,
    bands: list[BandConfig],
    grid_shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    """Integrate source-map powers per band, reshaped to (nx, ny, nz).

    grid_shape is (nx, ny, nz). Returns dict[band_name, ndarray of that shape].
    """
    nx, ny, nz = grid_shape
    out: dict[str, np.ndarray] = {}
    f = source_map.frequencies
    for band in bands:
        mask = (f >= band.f_min_hz) & (f <= band.f_max_hz)
        if not mask.any():
            out[band.name] = np.zeros((nx, ny, nz), dtype=np.float64)
            continue
        integrated = source_map.powers[mask].sum(axis=0)        # (G,)
        if integrated.size != nx * ny * nz:
            raise ValueError(
                f"powers length {integrated.size} doesn't match grid_shape {grid_shape}"
            )
        out[band.name] = integrated.reshape(nx, ny, nz)
    return out


def integrate_csm_band_maps_from_residual(
    *args, **kwargs,
) -> dict[str, np.ndarray]:
    """Reserved for future: rebuild a 'post' diagnostic map directly from the
    residual CSM via conventional beamforming. Not used in this spec — the
    'post' beam-map is derived from masking the source_map (Task 23)."""
    raise NotImplementedError


class ArrayFilterStage:
    name = "array_filter"

    def __init__(self, cfg: ArrayFilterStageConfig):
        self.cfg = cfg
        if cfg.algorithm not in ALGORITHM_REGISTRY:
            raise ValueError(
                f"unknown algorithm {cfg.algorithm!r}; available: {sorted(ALGORITHM_REGISTRY)}"
            )
        self.algo = ALGORITHM_REGISTRY[cfg.algorithm]()

    # ------------------------------------------------------------------ Phase 1
    def _build_csm(self, ctx) -> tuple[np.ndarray, np.ndarray]:
        rcfg = RuntimeCsmConfig(
            nperseg=self.cfg.csm.nperseg,
            noverlap=self.cfg.csm.noverlap,
            window=self.cfg.csm.window,
            diag_loading_rel=self.cfg.csm.diag_loading_rel,
            f_min_hz=self.cfg.csm.f_min_hz,
            f_max_hz=self.cfg.csm.f_max_hz,
        )
        return build_measurement_csm(ctx.time_data, ctx.sample_rate, rcfg)

    # ------------------------------------------------------------------ Phase 2
    def _build_fit_input(self, ctx) -> FitInput:
        """Cartesian diagnostic grid (default search space)."""
        zmin = self.cfg.diagnostic_grid.z_min_m
        zmax = self.cfg.diagnostic_grid.z_max_m
        if zmin is None:  # both None per validator
            plat = ctx.metadata.get("platform")
            if plat is None:
                raise ValueError(
                    "diagnostic_grid.z_min_m / z_max_m default to platform.rotor_positions[2,0]; "
                    "synth file lacks /platform/. Set them explicitly in the YAML."
                )
            z = float(np.asarray(plat["rotor_positions"])[2, 0])
            zmin = zmax = z
        diag_grid, diag_shape = build_diagnostic_grid(
            self.cfg.diagnostic_grid.extent_xy_m,
            self.cfg.diagnostic_grid.increment_m,
            zmin, zmax,
        )
        return FitInput(positions=diag_grid, reshape_hint=diag_shape)

    # ------------------------------------------------------------------ Phase 3
    def _fit_powers(
        self, csm: np.ndarray, freqs: np.ndarray,
        fit_input: FitInput, ctx,
    ) -> SourceMap:
        source_map = self.algo.fit(
            csm=csm,
            frequencies=freqs,
            time_data=None,
            sample_rate=ctx.sample_rate,
            mic_positions=ctx.mic_positions,
            grid_positions=fit_input.positions,
            params={
                "damp": self.cfg.clean_sc.damp,
                "n_iter": self.cfg.clean_sc.n_iter,
                "r_diag": self.cfg.clean_sc.r_diag,
            },
        )
        # Acoular's `classic`-steer BeamformerCleansc returns powers off by a
        # geometry-dependent factor (array gain). Trace-match per frequency to
        # restore Pa² convention so reconstruct_csm produces a sensible drone CSM.
        source_map, _ = rescale_source_map_to_csm_trace(
            source_map, csm, ctx.mic_positions,
        )
        return source_map

    # ------------------------------------------------------------------ Phase 4
    def _build_masks(self, fit_input: FitInput, ctx) -> MaskBundle:
        plat = ctx.metadata["platform"]
        rotor_disc_mask = build_rotor_disc_mask(
            fit_input.positions,
            np.asarray(plat["rotor_positions"]),
            np.asarray(plat["rotor_radii"]),
            self.cfg.rotor_z_tolerance_m,
        )
        target_box_mask = build_target_box_mask(
            fit_input.positions,
            self.cfg.target_point_m,
            self.cfg.target_box_half_extent_m,
        )
        drone_box_mask = build_target_box_mask(
            fit_input.positions,
            self.cfg.drone_box_center_m,
            self.cfg.drone_box_half_extent_m,
        )
        if self.cfg.mask_mode == "rotor_disc":
            active = rotor_disc_mask
        elif self.cfg.mask_mode == "drone_box":
            active = drone_box_mask
        else:  # "target_box"
            active = ~target_box_mask
        return MaskBundle(
            active=active,
            named={
                "rotor_disc_mask": rotor_disc_mask,
                "target_box_mask": target_box_mask,
                "drone_box_mask": drone_box_mask,
            },
        )

    # ------------------------------------------------------------------ Driver
    def process(self, ctx):
        csm, freqs = self._build_csm(ctx)
        fit_input = self._build_fit_input(ctx)
        source_map = self._fit_powers(csm, freqs, fit_input, ctx)
        masks = self._build_masks(fit_input, ctx)

        drone_csm = reconstruct_csm(
            source_map.subset(masks.active), ctx.mic_positions,
        )
        residual_csm = csm - drone_csm

        beam_maps = integrate_band_maps(
            source_map, self.cfg.bands, fit_input.reshape_hint,
        )
        psd_pre = steer_to_psd(csm, freqs, ctx.mic_positions, self.cfg.target_point_m)
        psd_post = steer_to_psd(residual_csm, freqs, ctx.mic_positions, self.cfg.target_point_m)

        af_meta: dict[str, Any] = {
            "csm_pre": csm,
            "residual_csm": residual_csm,
            "frequencies": freqs,
            "source_map": source_map,
            # Backward-compatible alias for the active subtraction mask:
            "drone_mask": masks.active,
            "mask_mode": self.cfg.mask_mode,
            "beam_maps": beam_maps,
            "target_psd_pre": psd_pre,
            "target_psd_post": psd_post,
            "diagnostic_grid": fit_input.positions,
            "diagnostic_grid_shape": fit_input.reshape_hint,
        }
        af_meta.update(masks.named)
        if fit_input.aux:
            af_meta["fit_input_aux"] = fit_input.aux
        return replace(ctx, metadata={**ctx.metadata, "array_filter": af_meta})


register_stage_builder("array_filter", lambda cfg, **_: ArrayFilterStage(cfg))
