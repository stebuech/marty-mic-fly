"""NNLS auf einem bekannten Atom-Set: CSM-Fitting ohne Grid-Suche.

Vorwärtsmodell (identisch zu base.reconstruct_csm):

    C_obs[f] = sum_g P[f, g] * a_g(f) a_g(f)^H
    a_g(f)[m] = (1 / (4π·r_{m,g})) * exp(-j 2π f r_{m,g} / c)

Pro Frequenzbin f wird das überbestimmte komplexe Gleichungssystem

    D[f] · p[f] ≈ vec(C_obs[f])

als reelles NNLS-Problem gelöst (Real- und Imaginärteil von vec(h_g h_g^H)
bzw. vec(C_obs) gestapelt). Liefert eine SourceMap mit Powers ≥ 0.

Optional kann ein einzelnes Atom als "diffus" markiert werden (params
``diffuse_atom_index``); dessen Wörterbuchspalte ist vec(I_M) statt
vec(h h^H), so dass es unkorrelierten Rauschanteil im CSM aufnehmen kann.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from martymicfly.constants import SPEED_OF_SOUND
from martymicfly.processing.algorithms import register_algorithm
from martymicfly.processing.algorithms.base import SourceMap


def _build_steering_matrix(
    mic_positions: np.ndarray, atom_positions: np.ndarray, freq: float, c: float,
) -> np.ndarray:
    diff = mic_positions[:, None, :] - atom_positions[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    r = np.where(r < 1e-9, 1e-9, r)
    return (1.0 / (4.0 * np.pi * r)) * np.exp(-2j * np.pi * freq * r / c)


@register_algorithm
class KnownGeometryLsqAlgorithm:
    name: str = "known_geometry_lsq"
    consumes: str = "csm"

    def fit(self, *, csm, frequencies, mic_positions, grid_positions,
            params, **_) -> SourceMap:
        c = float(params.get("speed_of_sound", SPEED_OF_SOUND))
        ridge = float(params.get("ridge", 0.0))
        cond_threshold = float(params.get("cond_threshold", 1e10))
        diffuse_idx = params.get("diffuse_atom_index", None)
        if diffuse_idx is not None:
            diffuse_idx = int(diffuse_idx)

        atoms = np.asarray(grid_positions, dtype=np.float64)
        mic = np.asarray(mic_positions, dtype=np.float64)
        freqs = np.asarray(frequencies, dtype=float)

        n_g = atoms.shape[0]
        n_m = mic.shape[0]
        n_f = freqs.shape[0]
        identity_vec_complex = np.eye(n_m, dtype=np.complex128).reshape(-1)

        powers = np.zeros((n_f, n_g), dtype=np.float64)
        skipped = 0

        for fi, f in enumerate(freqs):
            H = _build_steering_matrix(mic, atoms, float(f), c)        # (M, G)
            hh = np.einsum("mg,ng->mng", H, H.conj()).reshape(-1, n_g)  # (M*M, G)

            # Replace the diffuse atom's column with identity-column vec(I_M).
            # The identity column models uncorrelated mic noise that can't be
            # explained by any single steering vector.
            if diffuse_idx is not None:
                hh[:, diffuse_idx] = identity_vec_complex

            D_real = np.concatenate([hh.real, hh.imag], axis=0)        # (2*M*M, G)

            y_complex = np.asarray(csm[fi], dtype=np.complex128).reshape(-1)
            y_real = np.concatenate([y_complex.real, y_complex.imag])

            # Ill-conditioned bins (low frequency, near-collinear atoms): skip
            # rather than emit garbage. Cheap rank guard via singular values.
            sv = np.linalg.svd(D_real, compute_uv=False)
            if sv[-1] <= 0.0 or sv[0] / max(sv[-1], 1e-300) > cond_threshold:
                skipped += 1
                continue

            if ridge > 0.0:
                tikh = np.sqrt(ridge) * np.eye(n_g)
                D_aug = np.concatenate([D_real, tikh], axis=0)
                y_aug = np.concatenate([y_real, np.zeros(n_g)])
                p_f, _ = nnls(D_aug, y_aug)
            else:
                p_f, _ = nnls(D_real, y_real)
            powers[fi] = p_f

        return SourceMap(
            positions=atoms,
            powers=powers,
            frequencies=freqs,
            grid_shape=None,
            metadata={
                "algorithm": "known_geometry_lsq",
                "ridge": ridge,
                "cond_threshold": cond_threshold,
                "diffuse_atom_index": diffuse_idx,
                "skipped_bins": int(skipped),
            },
        )
