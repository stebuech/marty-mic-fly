"""Scaling-Ladder Diagnostic.

Diagnoses the ~−27 dB external_recovery bias observed in ext_only smoke runs
by following an analytically-defined white-noise monopole through three
points in the steering chain (mic-PSD, CSM-diagonal, steered PSD).

Run via:  uv run python analysis/scaling_ladder.py [--mic-geom PATH]
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


def propagate_white_noise(
    *,
    n_samples: int,
    sample_rate: float,
    s_q_pa2_per_hz: float,
    source_position: np.ndarray,
    mic_positions: np.ndarray,   # (M, 3)
    rng: np.random.Generator,
    speed_of_sound: float = 343.0,
) -> np.ndarray:
    """Propagate one white-noise monopole to M mics via free-field 1/r-Greens.

    The source signal q(t) is white noise with two-sided PSD = s_q_pa2_per_hz
    (one-sided density as used by `scipy.welch(..., scaling='density')`).
    Each mic receives `p_m(t) = q(t − r_m/c) / r_m` via fractional-sample
    delay (linear interpolation in time domain).

    Returns
    -------
    time_data : (n_samples, M) float64
    """
    src = np.asarray(source_position, dtype=np.float64)
    mics = np.asarray(mic_positions, dtype=np.float64)
    r = np.linalg.norm(mics - src[None, :], axis=1)         # (M,)

    # White noise of length n_samples + max-delay-margin so we can shift safely.
    max_delay_samples = int(np.ceil(r.max() / speed_of_sound * sample_rate)) + 4
    n_pad = n_samples + max_delay_samples

    # Generate q with the right one-sided PSD.
    # For a real Gaussian process with two-sided PSD = s_two_sided,
    # variance = s_two_sided · fs. Using one-sided density convention:
    # s_one_sided = 2 · s_two_sided (for f > 0), so variance = (s_one_sided/2) · fs.
    sigma = float(np.sqrt(s_q_pa2_per_hz * sample_rate / 2.0))
    q = rng.normal(0.0, sigma, size=n_pad)

    # Per-mic fractional delay via linear interpolation.
    n = np.arange(n_samples, dtype=np.float64)
    out = np.zeros((n_samples, mics.shape[0]), dtype=np.float64)
    for m in range(mics.shape[0]):
        delay_samples = r[m] / speed_of_sound * sample_rate
        # Source-time index for output sample n: t_src = n - delay
        # but we generated q starting at -max_delay (offset), so:
        t_src = n + (max_delay_samples - delay_samples)
        i0 = np.floor(t_src).astype(np.int64)
        frac = t_src - i0
        out[:, m] = (1.0 - frac) * q[i0] + frac * q[i0 + 1]
        out[:, m] /= r[m]
    return out


def rung1_mic_psd(
    *,
    time_data: np.ndarray,         # (N, M)
    sample_rate: float,
    s_q_pa2_per_hz: float,
    source_position: np.ndarray,
    mic_positions: np.ndarray,
    f_min_hz: float,
    f_max_hz: float,
    nperseg: int,
    noverlap: int,
    window: str,
) -> dict:
    """Rung 1 — direct Welch PSD per mic, compared to theoretical S_q / r_m^2."""
    f, p = welch(
        time_data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap,
        window=window, scaling="density", axis=0,
    )
    mask = (f >= f_min_hz) & (f <= f_max_hz)
    freqs = f[mask]
    psd = p[mask, :]                                            # (F, M)

    src = np.asarray(source_position, dtype=np.float64)
    mics = np.asarray(mic_positions, dtype=np.float64)
    r = np.linalg.norm(mics - src[None, :], axis=1)             # (M,)
    theoretical = s_q_pa2_per_hz / (r ** 2)                     # (M,)

    measured_per_mic = psd.mean(axis=0)                         # (M,)
    delta_db_per_mic = 10.0 * np.log10(measured_per_mic / theoretical)
    return {
        "frequencies_hz": freqs,
        "psd_per_mic": psd,                                     # (F, M)
        "theoretical_per_mic": theoretical,                     # (M,)
        "delta_db_per_mic": delta_db_per_mic,
        "delta_db_mean": float(delta_db_per_mic.mean()),
    }


def rung2_csm_diag(
    *,
    time_data: np.ndarray,
    sample_rate: float,
    s_q_pa2_per_hz: float,
    source_position: np.ndarray,
    mic_positions: np.ndarray,
    f_min_hz: float,
    f_max_hz: float,
    nperseg: int,
    noverlap: int,
    window: str,
    diag_loading_rel: float,
) -> dict:
    """Rung 2 — production CSM, diagonal vs theoretical S_q / r_m^2."""
    from martymicfly.processing.csm import CsmConfig, build_measurement_csm

    cfg = CsmConfig(
        nperseg=nperseg, noverlap=noverlap, window=window,
        diag_loading_rel=diag_loading_rel,
        f_min_hz=f_min_hz, f_max_hz=f_max_hz,
    )
    csm, freqs = build_measurement_csm(time_data, sample_rate, cfg)
    diag = np.real(np.diagonal(csm, axis1=1, axis2=2))      # (F, M)

    src = np.asarray(source_position, dtype=np.float64)
    mics = np.asarray(mic_positions, dtype=np.float64)
    r = np.linalg.norm(mics - src[None, :], axis=1)
    theoretical = s_q_pa2_per_hz / (r ** 2)

    measured_per_mic = diag.mean(axis=0)
    delta_db_per_mic = 10.0 * np.log10(measured_per_mic / theoretical)
    return {
        "csm": csm,
        "csm_shape": csm.shape,
        "frequencies_hz": freqs,
        "csm_diag_per_mic": diag,
        "theoretical_per_mic": theoretical,
        "delta_db_per_mic": delta_db_per_mic,
        "delta_db_mean": float(delta_db_per_mic.mean()),
    }


def rung3_steered_psd(
    *,
    time_data: np.ndarray,
    sample_rate: float,
    s_q_pa2_per_hz: float,
    source_position: np.ndarray,
    mic_positions: np.ndarray,
    f_min_hz: float,
    f_max_hz: float,
    nperseg: int,
    noverlap: int,
    window: str,
    diag_loading_rel: float,
) -> dict:
    """Rung 3 — production steer_to_psd at the source position vs S_q · <1/r_m>^2.

    The expectation derives from phase-only delay-and-sum with 1/M^2 normalization
    on a unit-amplitude steering vector: for a monopole at the target the
    quadratic form yields S_q · |Σ_m exp(-2j·2π f r_m/c) / r_m|² / M².  When
    the array aperture is small relative to the source distance (all r_m ≈ r̄),
    the sum simplifies to M/r̄ and the expectation approaches S_q · <1/r_m>².
    This approximation holds well for the production 16-mic array at 1.5 m.
    """
    from martymicfly.processing.csm import CsmConfig, build_measurement_csm
    from martymicfly.processing.steering import steer_to_psd

    cfg = CsmConfig(
        nperseg=nperseg, noverlap=noverlap, window=window,
        diag_loading_rel=diag_loading_rel,
        f_min_hz=f_min_hz, f_max_hz=f_max_hz,
    )
    csm, freqs = build_measurement_csm(time_data, sample_rate, cfg)
    psd = steer_to_psd(
        csm=csm,
        frequencies=freqs,
        mic_positions=np.asarray(mic_positions, dtype=np.float64),
        target_point=tuple(np.asarray(source_position, dtype=np.float64).tolist()),
    )                                                            # (F,)

    mics = np.asarray(mic_positions, dtype=np.float64)
    src = np.asarray(source_position, dtype=np.float64)
    r = np.linalg.norm(mics - src[None, :], axis=1)
    geom_factor = float((1.0 / r).mean()) ** 2                   # <1/r>^2
    theoretical = s_q_pa2_per_hz * geom_factor                   # scalar Pa²/Hz

    delta_db = 10.0 * np.log10(psd / theoretical)
    return {
        "frequencies_hz": freqs,
        "steered_psd": psd,
        "theoretical_psd": theoretical,
        "geometric_factor": geom_factor,
        "delta_db_per_freq": delta_db,
        "delta_db_band_mean": float(delta_db.mean()),
    }


def write_report(
    *,
    output_dir,
    rung1: dict,
    rung2: dict,
    rung3: dict,
    s_q_pa2_per_hz: float,
    source_position,
    mic_positions,
    sample_rate: float,
    config_summary: str,
) -> None:
    """Write a Markdown summary plus a metrics.json to output_dir."""
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    d1 = rung1["delta_db_mean"]
    d2 = rung2["delta_db_mean"]
    d3 = rung3["delta_db_band_mean"]

    pass_threshold_db = 0.5
    pass_3_threshold_db = 2.0

    def verdict(delta, thr):
        return "PASS" if abs(delta) < thr else "FAIL"

    diagnosis_lines = []
    if abs(d1) < pass_threshold_db and abs(d2) < pass_threshold_db and abs(d3) >= pass_3_threshold_db:
        diagnosis_lines.append(
            f"Forward + CSM are clean. The ~{d3:+.2f} dB offset enters at the "
            "steering stage. Likely cause: sign-convention mismatch between "
            "propagator (1/r delay) and `steer_to_psd` (h = exp(+j 2πf r/c)), "
            "or steering-norm convention (1/M² vs 1/M)."
        )
    elif abs(d1) < pass_threshold_db and abs(d2) >= pass_threshold_db:
        diagnosis_lines.append(
            f"Forward Welch is clean (Δ₁ = {d1:+.2f} dB) but the CSM stage adds "
            f"Δ₂ − Δ₁ = {d2-d1:+.2f} dB. Suspect: window or density normalization "
            "in `csm.py` (`scipy.signal.csd(..., scaling='density')` vs custom)."
        )
    elif abs(d1) >= pass_threshold_db:
        diagnosis_lines.append(
            f"Δ₁ = {d1:+.2f} dB — forward propagator itself disagrees with theory. "
            "Check the `s_q → variance` conversion (one-sided vs two-sided density) "
            "and the 1/r factor."
        )
    else:
        diagnosis_lines.append(
            "All three rungs within tolerance. The −27 dB bias observed in pipeline "
            "ext_only runs must therefore enter *outside* the CSM-and-steering path "
            "— investigate band-integration (`integrate_band_maps`) or GT-comparison."
        )

    md = [
        "# Scaling-Ladder Diagnostic Report",
        "",
        f"**Config:** {config_summary}",
        f"**Source PSD:** S_q = {10*np.log10(s_q_pa2_per_hz):.2f} dB re 1 Pa²/Hz",
        f"**Source position:** {tuple(np.asarray(source_position).tolist())}",
        f"**Sample rate:** {sample_rate:.0f} Hz",
        f"**Mics:** {len(mic_positions)} channels",
        "",
        "## Rung Deltas (band mean, dB)",
        "",
        "| Rung | Description | Δ (dB) | Threshold | Verdict |",
        "|------|-------------|--------|-----------|---------|",
        f"| 1 | Direct Welch mic-PSD vs S_q/r²        | {d1:+.3f} | ±0.5 | {verdict(d1, pass_threshold_db)} |",
        f"| 2 | CSM-diagonal vs S_q/r²                 | {d2:+.3f} | ±0.5 | {verdict(d2, pass_threshold_db)} |",
        f"| 3 | steer_to_psd at source vs S_q·⟨1/r⟩²   | {d3:+.3f} | ±2.0 | {verdict(d3, pass_3_threshold_db)} |",
        "",
        "## Diagnosis",
        "",
        diagnosis_lines[0],
        "",
        "## Per-mic Δ (Rung 1)",
        "",
        "| mic | r (m) | Δ₁ (dB) |",
        "|-----|-------|---------|",
    ]
    r = np.linalg.norm(np.asarray(mic_positions) - np.asarray(source_position)[None, :], axis=1)
    for m, (rm, d) in enumerate(zip(r, rung1["delta_db_per_mic"])):
        md.append(f"| {m} | {rm:.3f} | {d:+.3f} |")

    (output_dir / "report.md").write_text("\n".join(md) + "\n")

    metrics = {
        "rung1": {
            "delta_db_mean": d1,
            "delta_db_per_mic": rung1["delta_db_per_mic"].tolist(),
        },
        "rung2": {
            "delta_db_mean": d2,
            "delta_db_per_mic": rung2["delta_db_per_mic"].tolist(),
        },
        "rung3": {
            "delta_db_band_mean": d3,
            "geometric_factor": rung3["geometric_factor"],
        },
        "config": {
            "s_q_pa2_per_hz": s_q_pa2_per_hz,
            "sample_rate": sample_rate,
            "n_mics": int(len(mic_positions)),
            "source_position": list(map(float, source_position)),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def write_plots(*, output_dir, rung1: dict, rung2: dict, rung3: dict,
                s_q_pa2_per_hz: float, mic_positions, source_position) -> None:
    """Three HTML plots: per-mic Welch PSD, CSM-diagonal PSD, steered PSD."""
    import plotly.graph_objects as go
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rung 1: per-mic Welch
    f1 = rung1["frequencies_hz"]
    fig = go.Figure()
    r = np.linalg.norm(
        np.asarray(mic_positions) - np.asarray(source_position)[None, :], axis=1
    )
    for m in range(rung1["psd_per_mic"].shape[1]):
        fig.add_trace(go.Scatter(
            x=f1, y=10*np.log10(rung1["psd_per_mic"][:, m]),
            mode="lines", name=f"mic {m} (r={r[m]:.2f})",
            opacity=0.6, line={"width": 1},
        ))
        fig.add_trace(go.Scatter(
            x=[f1[0], f1[-1]],
            y=[10*np.log10(rung1["theoretical_per_mic"][m])]*2,
            mode="lines", name=f"theory mic {m}",
            line={"dash": "dash", "width": 1}, showlegend=False,
        ))
    fig.update_layout(
        title="Rung 1 — Welch PSD per mic vs theoretical S_q/r²",
        xaxis_title="frequency [Hz]", yaxis_title="PSD [dB re 1 Pa²/Hz]",
    )
    fig.write_html(output_dir / "mic_psd_vs_theory.html", include_plotlyjs="cdn")

    # Rung 2: CSM diagonal
    f2 = rung2["frequencies_hz"]
    fig = go.Figure()
    for m in range(rung2["csm_diag_per_mic"].shape[1]):
        fig.add_trace(go.Scatter(
            x=f2, y=10*np.log10(rung2["csm_diag_per_mic"][:, m]),
            mode="lines", name=f"mic {m}", opacity=0.6, line={"width": 1},
        ))
    fig.update_layout(
        title="Rung 2 — CSM diagonal per mic",
        xaxis_title="frequency [Hz]", yaxis_title="PSD [dB re 1 Pa²/Hz]",
    )
    fig.write_html(output_dir / "csm_diag_vs_theory.html", include_plotlyjs="cdn")

    # Rung 3: steered PSD
    f3 = rung3["frequencies_hz"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f3, y=10*np.log10(rung3["steered_psd"]),
        mode="lines", name="steer_to_psd at source",
    ))
    fig.add_trace(go.Scatter(
        x=[f3[0], f3[-1]],
        y=[10*np.log10(rung3["theoretical_psd"])]*2,
        mode="lines", name="theoretical S_q · <1/r>²",
        line={"dash": "dash"},
    ))
    fig.update_layout(
        title="Rung 3 — steered PSD at source vs geometric expectation",
        xaxis_title="frequency [Hz]", yaxis_title="PSD [dB re 1 Pa²/Hz]",
    )
    fig.write_html(output_dir / "steered_psd.html", include_plotlyjs="cdn")


def main() -> None:
    """End-to-end: load mic_geom, propagate, run 3 rungs, write report+plots."""
    import argparse
    from datetime import datetime
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mic-geom", type=Path,
        default=Path("/media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/mic_geom.xml"),
    )
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=float, default=51_200.0)
    parser.add_argument("--source-x", type=float, default=0.0)
    parser.add_argument("--source-y", type=float, default=0.0)
    parser.add_argument("--source-z", type=float, default=-1.5)
    parser.add_argument("--s-q-pa2-per-hz", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026_05_07)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/scaling_ladder")
    )
    args = parser.parse_args()

    from martymicfly.io.mic_geom import load_mic_geom_xml
    mic_positions = load_mic_geom_xml(args.mic_geom)
    source_pos = np.array([args.source_x, args.source_y, args.source_z])
    n_samples = int(args.duration_s * args.sample_rate)
    rng = np.random.default_rng(args.seed)

    print(f"[scaling_ladder] {mic_positions.shape[0]} mics from {args.mic_geom}")
    print(f"[scaling_ladder] propagating {args.duration_s:.1f}s @ {args.sample_rate:.0f} Hz")
    time_data = propagate_white_noise(
        n_samples=n_samples, sample_rate=args.sample_rate,
        s_q_pa2_per_hz=args.s_q_pa2_per_hz,
        source_position=source_pos, mic_positions=mic_positions, rng=rng,
    )

    common = dict(
        time_data=time_data, sample_rate=args.sample_rate,
        s_q_pa2_per_hz=args.s_q_pa2_per_hz,
        source_position=source_pos, mic_positions=mic_positions,
        f_min_hz=200.0, f_max_hz=6000.0,
        nperseg=512, noverlap=256, window="hann",
    )
    print("[scaling_ladder] rung 1 …")
    r1 = rung1_mic_psd(**common)
    print(f"  Δ₁ = {r1['delta_db_mean']:+.3f} dB")
    print("[scaling_ladder] rung 2 …")
    r2 = rung2_csm_diag(**common, diag_loading_rel=0.0)
    print(f"  Δ₂ = {r2['delta_db_mean']:+.3f} dB")
    print("[scaling_ladder] rung 3 …")
    r3 = rung3_steered_psd(**common, diag_loading_rel=0.0)
    print(f"  Δ₃ = {r3['delta_db_band_mean']:+.3f} dB")

    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out = args.output_root / run_id
    write_report(
        output_dir=out, rung1=r1, rung2=r2, rung3=r3,
        s_q_pa2_per_hz=args.s_q_pa2_per_hz,
        source_position=source_pos, mic_positions=mic_positions,
        sample_rate=args.sample_rate,
        config_summary=(
            f"duration={args.duration_s}s, fs={args.sample_rate:.0f}, "
            f"nperseg=512, noverlap=256, window=hann, "
            f"f_band=[200,6000] Hz, source={tuple(source_pos.tolist())}"
        ),
    )
    write_plots(
        output_dir=out, rung1=r1, rung2=r2, rung3=r3,
        s_q_pa2_per_hz=args.s_q_pa2_per_hz,
        mic_positions=mic_positions, source_position=source_pos,
    )
    print(f"[scaling_ladder] wrote report + plots → {out}")


if __name__ == "__main__":
    main()
