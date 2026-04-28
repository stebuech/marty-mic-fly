"""Stage 2 — array deconvolution stage."""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from martymicfly.config import ArrayFilterStageConfig, BandConfig
from martymicfly.processing.algorithms import ALGORITHM_REGISTRY
from martymicfly.processing.algorithms.base import SourceMap, reconstruct_csm
from martymicfly.processing.beamform_grid import (
    build_diagnostic_grid,
    build_rotor_disc_mask,
)
from martymicfly.processing.csm import CsmConfig as RuntimeCsmConfig, build_measurement_csm
from martymicfly.processing.pipeline import register_stage_builder
from martymicfly.processing.steering import steer_to_psd


def integrate_band_maps(
    source_map: SourceMap,
    bands: list[BandConfig],
    grid_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    nx, ny = grid_shape
    out: dict[str, np.ndarray] = {}
    f = source_map.frequencies
    for band in bands:
        mask = (f >= band.f_min_hz) & (f <= band.f_max_hz)
        if not mask.any():
            out[band.name] = np.zeros((nx, ny), dtype=np.float64)
            continue
        integrated = source_map.powers[mask].sum(axis=0)        # (G,)
        if integrated.size != nx * ny:
            raise ValueError(
                f"powers length {integrated.size} doesn't match grid_shape {grid_shape}"
            )
        out[band.name] = integrated.reshape(nx, ny)
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

    def process(self, ctx):
        # 1. CSM
        rcfg = RuntimeCsmConfig(
            nperseg=self.cfg.csm.nperseg,
            noverlap=self.cfg.csm.noverlap,
            window=self.cfg.csm.window,
            diag_loading_rel=self.cfg.csm.diag_loading_rel,
            f_min_hz=self.cfg.csm.f_min_hz,
            f_max_hz=self.cfg.csm.f_max_hz,
        )
        csm, freqs = build_measurement_csm(ctx.time_data, ctx.sample_rate, rcfg)

        # 2. Diagnostic grid
        z = self.cfg.diagnostic_grid.z_m
        if z is None:
            plat = ctx.metadata.get("platform")
            if plat is None:
                raise ValueError(
                    "diagnostic_grid.z_m=null requires synth file to carry /platform/"
                )
            z = float(np.asarray(plat["rotor_positions"])[2, 0])
        diag_grid, diag_shape = build_diagnostic_grid(
            self.cfg.diagnostic_grid.extent_xy_m,
            self.cfg.diagnostic_grid.increment_m,
            z,
        )

        # 3. CLEAN-SC
        source_map = self.algo.fit(
            csm=csm,
            frequencies=freqs,
            time_data=None,
            sample_rate=ctx.sample_rate,
            mic_positions=ctx.mic_positions,
            grid_positions=diag_grid,
            params={
                "damp": self.cfg.clean_sc.damp,
                "n_iter": self.cfg.clean_sc.n_iter,
            },
        )

        # 4. Rotor-disc mask
        plat = ctx.metadata["platform"]
        drone_mask = build_rotor_disc_mask(
            diag_grid,
            np.asarray(plat["rotor_positions"]),
            np.asarray(plat["rotor_radii"]),
            self.cfg.rotor_z_tolerance_m,
        )

        # 5. Drone CSM and residual
        drone_csm = reconstruct_csm(source_map.subset(drone_mask), ctx.mic_positions)
        residual_csm = csm - drone_csm

        # 6. Beam maps
        beam_maps = integrate_band_maps(source_map, self.cfg.bands, diag_shape)

        # 7. Pseudo-target PSD
        psd_pre = steer_to_psd(csm, freqs, ctx.mic_positions, self.cfg.target_point_m)
        psd_post = steer_to_psd(residual_csm, freqs, ctx.mic_positions, self.cfg.target_point_m)

        new_metadata = {
            **ctx.metadata,
            "array_filter": {
                "csm_pre": csm,
                "residual_csm": residual_csm,
                "frequencies": freqs,
                "source_map": source_map,
                "drone_mask": drone_mask,
                "beam_maps": beam_maps,
                "target_psd_pre": psd_pre,
                "target_psd_post": psd_post,
                "diagnostic_grid": diag_grid,
                "diagnostic_grid_shape": diag_shape,
            },
        }
        return replace(ctx, metadata=new_metadata)


register_stage_builder("array_filter", lambda cfg, **_: ArrayFilterStage(cfg))
