"""CLEAN-SC algorithm wrapper around acoular.BeamformerCleansc.

Verified against acoular 26.01: the modern API uses ImportGrid().pos
(in-memory ndarray, shape (3, G)) and MicGeom(pos_total=...) with shape
(3, M). No tempfile/XML round-trip is needed (older acoular versions
required gpos_file=<xml_path>; that path is not exercised here).
"""
from __future__ import annotations

import numpy as np

from martymicfly.processing.algorithms import register_algorithm
from martymicfly.processing.algorithms.base import Algorithm, SourceMap


def _import_grid_from_array(grid_positions: np.ndarray):
    """Wrap (G, 3) into an Acoular Grid via direct in-memory assignment.

    Acoular >= 26.01 exposes `ImportGrid().pos` as a writable trait of
    shape (3, G); we assign the transposed array directly. If support
    for an older acoular (<= 24.x) is needed later, fall back to writing
    a temporary XML and using `ImportGrid(gpos_file=...)`.
    """
    from acoular import ImportGrid

    pos = np.asarray(grid_positions, dtype=np.float64).T  # (3, G)
    grid = ImportGrid()
    grid.pos = pos
    return grid


@register_algorithm
class CleanScAlgorithm:
    name: str = "clean_sc"
    consumes: str = "csm"

    def fit(self, *, csm, frequencies, mic_positions, grid_positions,
            params, **_) -> SourceMap:
        from acoular import (
            BeamformerCleansc,
            MicGeom,
            PowerSpectraImport,
            SteeringVector,
            config as acoular_config,
        )

        acoular_config.global_caching = "none"
        # MicGeom expects pos_total of shape (3, M); mic_positions arrives as (M, 3).
        mg = MicGeom(pos_total=np.asarray(mic_positions, dtype=np.float64).T.copy())
        grid = _import_grid_from_array(grid_positions)
        steer = SteeringVector(grid=grid, mics=mg, steer_type="classic")

        ps = PowerSpectraImport()
        ps.csm = np.asarray(csm, dtype=np.complex128)
        ps.frequencies = np.asarray(frequencies, dtype=float)

        bf = BeamformerCleansc(
            freq_data=ps, steer=steer, r_diag=False,
            damp=params["damp"], n_iter=params["n_iter"],
            cached=False,
        )
        powers = np.zeros((len(frequencies), len(grid_positions)), dtype=float)
        for i, f in enumerate(frequencies):
            powers[i] = np.asarray(bf.synthetic(float(f), num=0)).ravel()

        return SourceMap(
            positions=np.asarray(grid_positions, dtype=np.float64),
            powers=powers,
            frequencies=np.asarray(frequencies, dtype=float),
            grid_shape=None,
            metadata={"damp": params["damp"], "n_iter": params["n_iter"]},
        )
