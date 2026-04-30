"""Tests for the known-geometry NNLS algorithm and stage subclass."""
from __future__ import annotations

import numpy as np
import pytest

from martymicfly.constants import SPEED_OF_SOUND


def _build_atom_csm(positions: np.ndarray, mic_positions: np.ndarray,
                    powers_per_freq: np.ndarray, frequencies: np.ndarray,
                    c: float = SPEED_OF_SOUND) -> np.ndarray:
    """Synthesize C[f] = sum_g p[f, g] * h_g(f) h_g(f)^H using the same
    forward model the algorithm must match (1/(4πr) * exp(-j 2πf r/c))."""
    diff = mic_positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    r = np.where(r < 1e-9, 1e-9, r)
    inv_r = 1.0 / (4.0 * np.pi * r)
    n_f = frequencies.shape[0]
    n_m = mic_positions.shape[0]
    csm = np.zeros((n_f, n_m, n_m), dtype=np.complex128)
    for fi, f in enumerate(frequencies):
        h = inv_r * np.exp(-2j * np.pi * f * r / c)
        weighted = h * powers_per_freq[fi][None, :]
        csm[fi] = weighted @ h.conj().T
    return csm


def _mic_geom_simple() -> np.ndarray:
    return np.array([
        [-0.3, -0.3, 0.0],
        [+0.3, -0.3, 0.0],
        [+0.3, +0.3, 0.0],
        [-0.3, +0.3, 0.0],
        [0.0, 0.0, 0.0],
    ])


def test_algorithm_registered():
    from martymicfly.processing.algorithms import ALGORITHM_REGISTRY
    # Force import of the module so it self-registers.
    import martymicfly.processing.algorithms.known_geometry_lsq  # noqa: F401
    assert "known_geometry_lsq" in ALGORITHM_REGISTRY


def test_single_atom_recovery():
    """Single atom: power must be recovered at correct atom, ≈0 elsewhere."""
    import martymicfly.processing.algorithms.known_geometry_lsq  # noqa: F401
    from martymicfly.processing.algorithms import ALGORITHM_REGISTRY

    mic = _mic_geom_simple()
    atoms = np.array([
        [0.0, 0.0, -1.5],
        [+0.2, 0.0, 0.0],
        [-0.2, 0.0, 0.0],
    ])
    freqs = np.array([800.0, 1500.0])
    P = np.zeros((freqs.size, atoms.shape[0]))
    P[:, 0] = [4.0, 9.0]                      # only first atom has power
    csm = _build_atom_csm(atoms, mic, P, freqs)

    algo = ALGORITHM_REGISTRY["known_geometry_lsq"]()
    sm = algo.fit(
        csm=csm, frequencies=freqs, time_data=None,
        sample_rate=48000.0, mic_positions=mic, grid_positions=atoms,
        params={},
    )
    assert sm.powers.shape == (freqs.size, atoms.shape[0])
    np.testing.assert_allclose(sm.powers[:, 0], P[:, 0], rtol=1e-6, atol=1e-9)
    assert np.all(sm.powers[:, 1:] < 1e-6)


def test_multi_atom_separation():
    """Two incoherent sources: NNLS must split power between them correctly."""
    import martymicfly.processing.algorithms.known_geometry_lsq  # noqa: F401
    from martymicfly.processing.algorithms import ALGORITHM_REGISTRY

    mic = _mic_geom_simple()
    atoms = np.array([
        [+0.25, 0.0, 0.0],     # rotor #1
        [-0.25, 0.0, 0.0],     # rotor #2
        [0.0, 0.0, -1.5],      # target
    ])
    freqs = np.array([1000.0, 2000.0])
    P_truth = np.array([
        [2.0, 1.0, 5.0],
        [3.0, 0.5, 7.0],
    ])
    csm = _build_atom_csm(atoms, mic, P_truth, freqs)

    algo = ALGORITHM_REGISTRY["known_geometry_lsq"]()
    sm = algo.fit(
        csm=csm, frequencies=freqs, time_data=None,
        sample_rate=48000.0, mic_positions=mic, grid_positions=atoms,
        params={},
    )
    np.testing.assert_allclose(sm.powers, P_truth, rtol=1e-4, atol=1e-6)


def test_nnls_powers_nonnegative():
    """Powers from NNLS must always be ≥ 0 even with noisy CSM."""
    import martymicfly.processing.algorithms.known_geometry_lsq  # noqa: F401
    from martymicfly.processing.algorithms import ALGORITHM_REGISTRY

    rng = np.random.default_rng(0)
    mic = _mic_geom_simple()
    atoms = np.array([[0.0, 0.0, -1.5], [+0.2, 0.0, 0.0]])
    freqs = np.array([500.0, 1500.0])
    P = np.array([[2.0, 1.0], [3.0, 0.5]])
    csm = _build_atom_csm(atoms, mic, P, freqs)
    # Add hermitian noise.
    n_m = mic.shape[0]
    for fi in range(freqs.size):
        z = rng.standard_normal((n_m, n_m)) + 1j * rng.standard_normal((n_m, n_m))
        csm[fi] += 0.01 * (z + z.conj().T)

    algo = ALGORITHM_REGISTRY["known_geometry_lsq"]()
    sm = algo.fit(
        csm=csm, frequencies=freqs, time_data=None,
        sample_rate=48000.0, mic_positions=mic, grid_positions=atoms,
        params={},
    )
    assert np.all(sm.powers >= 0.0)


def test_ill_conditioned_low_freq_zeroed():
    """At very low frequency the dictionary collapses; algorithm must guard
    via the cond_threshold and emit zero powers instead of garbage."""
    import martymicfly.processing.algorithms.known_geometry_lsq  # noqa: F401
    from martymicfly.processing.algorithms import ALGORITHM_REGISTRY

    mic = _mic_geom_simple()
    # Two atoms very close together → near-collinear columns.
    atoms = np.array([[0.0, 0.0, -1.5], [0.0001, 0.0, -1.5]])
    freqs = np.array([10.0])  # λ ≫ aperture
    P = np.array([[1.0, 1.0]])
    csm = _build_atom_csm(atoms, mic, P, freqs)

    algo = ALGORITHM_REGISTRY["known_geometry_lsq"]()
    sm = algo.fit(
        csm=csm, frequencies=freqs, time_data=None,
        sample_rate=48000.0, mic_positions=mic, grid_positions=atoms,
        params={"cond_threshold": 1e6},
    )
    # Either forced to zero or finite & non-negative; must not blow up.
    assert np.all(np.isfinite(sm.powers))
    assert np.all(sm.powers >= 0.0)


def test_forward_model_matches_reconstruct_csm():
    """The algorithm's forward model must agree numerically with
    base.reconstruct_csm — otherwise CSM subtraction won't cancel cleanly."""
    import martymicfly.processing.algorithms.known_geometry_lsq  # noqa: F401
    from martymicfly.processing.algorithms import ALGORITHM_REGISTRY
    from martymicfly.processing.algorithms.base import reconstruct_csm

    mic = _mic_geom_simple()
    atoms = np.array([
        [+0.25, 0.0, 0.0],
        [-0.25, 0.0, 0.0],
        [0.0, 0.0, -1.5],
    ])
    freqs = np.array([700.0, 1800.0])
    P = np.array([[1.5, 0.7, 4.0], [2.5, 0.3, 6.5]])
    csm = _build_atom_csm(atoms, mic, P, freqs)

    algo = ALGORITHM_REGISTRY["known_geometry_lsq"]()
    sm = algo.fit(
        csm=csm, frequencies=freqs, time_data=None,
        sample_rate=48000.0, mic_positions=mic, grid_positions=atoms,
        params={},
    )
    csm_rebuilt = reconstruct_csm(sm, mic)
    np.testing.assert_allclose(csm_rebuilt, csm, rtol=1e-5, atol=1e-8)


# --------------------------------------------------------------- Stage tests

def _platform_metadata():
    """Match tiny_synth_mixed structure: 2 rotors at z≈0."""
    import h5py
    with h5py.File("tests/fixtures/tiny_synth_mixed.h5", "r") as f:
        return {
            "rotor_positions": np.asarray(f["platform/rotor_positions"]),
            "rotor_radii": np.asarray(f["platform/rotor_radii"]),
            "blade_counts": np.asarray(f["platform/blade_counts"]),
        }


def test_known_atoms_stage_e2e():
    from martymicfly.config import (
        ArrayFilterStageConfig, AtomSetConfig, BandConfig,
        CleanScConfig, CsmConfig, DiagnosticGridConfig,
    )
    from martymicfly.io.mic_geom import load_mic_geom_xml
    from martymicfly.io.synth_h5 import load_synth_h5
    from martymicfly.processing.array_filter_atoms import KnownAtomsArrayFilterStage
    from martymicfly.processing.pipeline import PipelineContext

    synth = load_synth_h5("tests/fixtures/tiny_synth_mixed.h5")
    mic_pos = load_mic_geom_xml("tests/fixtures/tiny_geom_4mic.xml")
    cfg = ArrayFilterStageConfig(
        kind="array_filter",
        algorithm="known_geometry_lsq",
        csm=CsmConfig(nperseg=256, noverlap=128, f_min_hz=200.0, f_max_hz=4000.0),
        diagnostic_grid=DiagnosticGridConfig(
            extent_xy_m=0.6, increment_m=0.05, z_min_m=0.0, z_max_m=0.0,
        ),
        bands=[BandConfig(name="mid", f_min_hz=500.0, f_max_hz=2000.0)],
        target_point_m=(0.5, 0.0, -0.5),
        rotor_z_tolerance_m=0.05,
        atoms=AtomSetConfig(),
    )
    ctx = PipelineContext(
        time_data=synth["time_data"],
        sample_rate=synth["sample_rate"],
        rpm_per_esc=synth["rpm_per_esc"],
        mic_positions=mic_pos,
        per_motor_bpf=np.zeros((synth["time_data"].shape[0], 2)),
        harm_matrix=np.zeros((synth["time_data"].shape[0], 8)),
        metadata={"platform": synth["platform"]},
    )
    stage = KnownAtomsArrayFilterStage(cfg)
    new_ctx = stage.process(ctx)
    af = new_ctx.metadata["array_filter"]

    # Same metadata schema as ArrayFilterStage:
    for key in ("csm_pre", "residual_csm", "frequencies", "source_map",
                "drone_mask", "target_psd_pre", "target_psd_post",
                "beam_maps", "diagnostic_grid"):
        assert key in af, f"missing metadata key: {key}"

    csm = af["csm_pre"]
    res = af["residual_csm"]
    assert csm.shape == res.shape
    np.testing.assert_allclose(res, res.conj().transpose(0, 2, 1), atol=1e-8)
    assert np.all(np.isfinite(af["target_psd_post"]))
    band_integrated = af["beam_maps"]["mid"]
    assert np.all(band_integrated >= -1e-12)
    # New diagnostic masks specific to atoms-stage:
    assert af["drone_atom_mask"].dtype == bool
    assert af["target_atom_mask"].dtype == bool
    assert af["drone_atom_mask"].sum() >= 1
    assert af["target_atom_mask"].sum() == 1


def test_factory_dispatches_to_known_atoms_stage():
    """Setting algorithm=known_geometry_lsq routes through the factory."""
    from martymicfly.config import (
        ArrayFilterStageConfig, AtomSetConfig, BandConfig,
        CsmConfig, DiagnosticGridConfig,
    )
    from martymicfly.processing.array_filter import _array_filter_factory
    from martymicfly.processing.array_filter_atoms import KnownAtomsArrayFilterStage

    cfg = ArrayFilterStageConfig(
        kind="array_filter",
        algorithm="known_geometry_lsq",
        csm=CsmConfig(),
        diagnostic_grid=DiagnosticGridConfig(z_min_m=0.0, z_max_m=0.0),
        bands=[BandConfig(name="mid", f_min_hz=500.0, f_max_hz=2000.0)],
        atoms=AtomSetConfig(),
    )
    stage = _array_filter_factory(cfg)
    assert isinstance(stage, KnownAtomsArrayFilterStage)
