"""Track-B Stage-2-Variante: NNLS auf bekanntem Atom-Set.

Statt eines kartesischen Suchgitters benutzt dieser Stage einen kleinen Satz
diskreter Atome (Rotorpositionen + Zielposition, optional ein Diffuses-Atom).
Der NNLS-Algorithmus fittet Powers direkt; CLEAN-SC-spezifisches Trace-
Rescale entfällt, weil das Vorwärtsmodell bereits in Pa² konsistent ist.

Hinweis zur Diagnostik-Repräsentation:
``FitInput.positions`` enthält pseudo-2D-Koordinaten (alle Atome auf einer
Ebene mit z=0), damit ``eval/array_plots`` die single-z-slice-Pfade benutzt
und nicht auf einer Cartesian-Grid-Annahme abstürzt. Die echten Atom-
Positionen (für reconstruct_csm) liegen separat in ``aux['real_positions']``
und werden direkt an den Algorithmus weitergereicht. ``source_map.positions``
behält die echten Positionen, damit die CSM-Subtraktion physikalisch korrekt
bleibt.
"""
from __future__ import annotations

import numpy as np

from martymicfly.processing.array_filter import (
    ArrayFilterStage,
    FitInput,
    MaskBundle,
)
from martymicfly.processing.algorithms.base import SourceMap


def _platform_rotor_positions(plat) -> np.ndarray:
    rp = np.asarray(plat["rotor_positions"], dtype=np.float64)
    if rp.ndim == 2 and rp.shape[0] == 3:
        rp = rp.T
    return rp


def _inter_rotor_midpoints(rotor_positions: np.ndarray, tol: float = 0.10) -> np.ndarray:
    """Midpoints between nearest-neighbor rotor pairs.

    For a quadcopter (4 rotors at the corners of a square), this returns the
    4 side midpoints — i.e. the points that lie *between* adjacent rotors,
    in the gaps where the broadband downwash radiation is concentrated.

    Algorithm: pairwise distances, identify the minimum (d_min), keep every
    unordered pair (i, j) with |d_ij − d_min| ≤ tol·d_min, return midpoints.
    """
    rp = np.asarray(rotor_positions, dtype=np.float64)
    n = rp.shape[0]
    if n < 2:
        return rp.copy()
    diffs = rp[:, None, :] - rp[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    iu, ju = np.triu_indices(n, k=1)
    pair_d = dists[iu, ju]
    d_min = float(pair_d.min())
    keep = np.abs(pair_d - d_min) <= tol * d_min
    pairs = np.stack([iu[keep], ju[keep]], axis=1)
    return 0.5 * (rp[pairs[:, 0]] + rp[pairs[:, 1]])


class KnownAtomsArrayFilterStage(ArrayFilterStage):
    """Atom-Set-Stage. Erbt Phase 1 (CSM) und den Driver-Loop von ArrayFilterStage."""

    # ------------------------------------------------------------------ Phase 2
    def _build_fit_input(self, ctx) -> FitInput:
        plat = ctx.metadata["platform"]
        rotor_positions = _platform_rotor_positions(plat)        # (R, 3)

        atoms_cfg = self.cfg.atoms
        target_pos = (
            atoms_cfg.target_atom_position_m
            if atoms_cfg is not None and atoms_cfg.target_atom_position_m is not None
            else self.cfg.target_point_m
        )

        drone_mode = atoms_cfg.drone_atoms if atoms_cfg is not None else "rotor_positions"
        if drone_mode == "inter_rotor_midpoints":
            drone_positions = _inter_rotor_midpoints(rotor_positions)
        elif drone_mode == "subsource_positions":
            raise NotImplementedError(
                "drone_atoms='subsource_positions' requires the source artifact "
                "and is not wired through ctx yet"
            )
        else:  # rotor_positions
            drone_positions = rotor_positions

        real_positions: list[np.ndarray] = []
        kinds: list[str] = []
        for i in range(drone_positions.shape[0]):
            real_positions.append(drone_positions[i])
            kinds.append("drone")
        real_positions.append(np.asarray(target_pos, dtype=np.float64))
        kinds.append("target")

        diffuse_idx = None
        if atoms_cfg is not None and atoms_cfg.include_diffuse:
            real_positions.append(
                np.asarray(ctx.mic_positions, dtype=np.float64).mean(axis=0),
            )
            kinds.append("diffuse")
            diffuse_idx = len(kinds) - 1

        real_positions_arr = np.stack(real_positions, axis=0)    # (G, 3)
        kinds_arr = np.array(kinds)
        n_g = real_positions_arr.shape[0]

        # Pseudo-positions for downstream visualization paths that assume a
        # Cartesian (nx, ny, nz) layout. We anchor every atom at z=0 and
        # spread them along x so unique z values collapse to {0} → array_plots
        # falls into its degenerate single-z-slice branch instead of trying
        # to reshape against multiple unique z values.
        pseudo_positions = np.stack([
            np.linspace(-0.5, 0.5, n_g) if n_g > 1 else np.array([0.0]),
            np.zeros(n_g),
            np.zeros(n_g),
        ], axis=1)

        return FitInput(
            positions=pseudo_positions,
            reshape_hint=(n_g, 1, 1),
            aux={
                "atom_kinds": kinds_arr,
                "diffuse_atom_index": diffuse_idx,
                "real_positions": real_positions_arr,
            },
        )

    # ------------------------------------------------------------------ Phase 3
    def _fit_powers(
        self, csm: np.ndarray, freqs: np.ndarray,
        fit_input: FitInput, ctx,
    ) -> SourceMap:
        atoms_cfg = self.cfg.atoms
        params: dict = {
            "ridge": atoms_cfg.ridge if atoms_cfg is not None else 0.0,
            "cond_threshold": atoms_cfg.cond_threshold if atoms_cfg is not None else 1e10,
        }
        diffuse_idx = fit_input.aux.get("diffuse_atom_index")
        if diffuse_idx is not None:
            params["diffuse_atom_index"] = diffuse_idx

        real_positions = fit_input.aux["real_positions"]
        return self.algo.fit(
            csm=csm, frequencies=freqs, time_data=None,
            sample_rate=ctx.sample_rate, mic_positions=ctx.mic_positions,
            grid_positions=real_positions,
            params=params,
        )
        # No trace-rescale: NNLS already fits powers in Pa² against the same
        # forward model used by reconstruct_csm.

    # ------------------------------------------------------------------ Phase 4
    def _build_masks(self, fit_input: FitInput, ctx) -> MaskBundle:
        kinds = fit_input.aux["atom_kinds"]
        drone_atom_mask = kinds == "drone"
        target_atom_mask = kinds == "target"
        diffuse_atom_mask = kinds == "diffuse"
        return MaskBundle(
            active=drone_atom_mask,
            named={
                "drone_atom_mask": drone_atom_mask,
                "target_atom_mask": target_atom_mask,
                "diffuse_atom_mask": diffuse_atom_mask,
                # Keep parent-stage keys present so downstream metadata
                # consumers (eval/array_metrics, run_pipeline plot path)
                # don't KeyError when they expect them. Atoms-stage always
                # subtracts the drone atoms regardless of mask_mode setting.
                "rotor_disc_mask": drone_atom_mask,
                "target_box_mask": target_atom_mask,
                "drone_box_mask": drone_atom_mask,
            },
        )
