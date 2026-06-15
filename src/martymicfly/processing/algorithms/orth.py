"""Orthogonal deconvolution wrapper around acoular.BeamformerOrth.

One of the three deconvolution methods named in AP2-A of the proposal
(orthogonale Entfaltung, CLEAN-SC, CLEAN-T). Orthogonal beamforming maps the
``n`` strongest CSM eigenvalues onto the grid — each eigenmode is placed at
the grid point where its eigen-beamformer response peaks. Unlike CLEAN-SC it
is non-iterative and uses the CSM eigenstructure directly, which tends to be
more stable on small arrays where CLEAN-SC's source-coherence step degrades.

Mirrors the CLEAN-SC wrapper: in-memory ImportGrid().pos and
MicGeom(pos_total=...) on acoular >= 26.01, no XML round-trip.
"""
from __future__ import annotations

import numpy as np

from martymicfly.processing.algorithms import register_algorithm
from martymicfly.processing.algorithms.base import SourceMap


def _import_grid_from_array(grid_positions: np.ndarray):
    """Wrap (G, 3) into an Acoular Grid via direct in-memory assignment."""
    from acoular import ImportGrid

    grid = ImportGrid()
    grid.pos = np.asarray(grid_positions, dtype=np.float64).T  # (3, G)
    return grid


@register_algorithm
class OrthAlgorithm:
    name: str = "orth"
    consumes: str = "csm"

    def fit(self, *, csm, frequencies, mic_positions, grid_positions,
            params, **_) -> SourceMap:
        from acoular import (
            BeamformerOrth,
            MicGeom,
            PowerSpectraImport,
            SteeringVector,
            config as acoular_config,
        )

        acoular_config.global_caching = "none"
        mg = MicGeom(pos_total=np.asarray(mic_positions, dtype=np.float64).T.copy())
        grid = _import_grid_from_array(grid_positions)
        steer = SteeringVector(grid=grid, mics=mg, steer_type="classic")

        ps = PowerSpectraImport()
        ps.csm = np.asarray(csm, dtype=np.complex128)
        ps.frequencies = np.asarray(frequencies, dtype=float)

        bf = BeamformerOrth(
            freq_data=ps, steer=steer,
            n=int(params.get("n", 8)),
            r_diag=bool(params.get("r_diag", True)),
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
            metadata={"n": int(params.get("n", 8))},
        )
