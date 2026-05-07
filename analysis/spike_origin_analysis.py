"""Identify where the frequency-domain spikes in steered ext_only spectra
actually come from.

Three candidate sources, ordered by tightness of fit to data:
  (a) array geometry — predicted by T(f) = |Σ exp(-2j·2πf r/c)/r|²/M²
  (b) Welch-bin variance — would manifest as ±0.1 dB random per-bin noise
  (c) CLEAN-SC iteration artifacts — would cluster among DOA methods only,
      since NNLS doesn't use CLEAN-SC

Outputs a correlation matrix across all method outputs + psd_pre + analytical
transfer + a fine-grid analytical curve to visualize geometry smoothness.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import yaml

from martymicfly.io.mic_geom import load_mic_geom_xml
from martymicfly.processing.csm import CsmConfig
from martymicfly.processing.steering import steer_to_psd

import importlib.util
import sys
_SCRIPT = Path(__file__).resolve().parent / "plot_calibration_effect.py"
_spec = importlib.util.spec_from_file_location("plot_calibration_effect", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["plot_calibration_effect"] = _mod
_spec.loader.exec_module(_mod)

_AT = importlib.util.spec_from_file_location(
    "array_transfer_at_target",
    Path(__file__).resolve().parent / "array_transfer_at_target.py",
)
_at_mod = importlib.util.module_from_spec(_AT)
sys.modules["array_transfer_at_target"] = _at_mod
_AT.loader.exec_module(_at_mod)


def db(x): return 10 * np.log10(np.maximum(x, 1e-30))


def correlate(a, b, *, in_db: bool):
    x = db(a) if in_db else np.asarray(a)
    y = db(b) if in_db else np.asarray(b)
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Any ext_only run dir; siblings are scanned for residual_csm.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load((args.run_dir / "config.yaml").read_text())
    stage_cfg = cfg["stages"][0]
    target = tuple(stage_cfg["target_point_m"])
    csm_cfg = CsmConfig(
        nperseg=int(stage_cfg["csm"]["nperseg"]),
        noverlap=int(stage_cfg["csm"]["noverlap"]),
        window=stage_cfg["csm"]["window"],
        diag_loading_rel=float(stage_cfg["csm"]["diag_loading_rel"]),
        f_min_hz=float(stage_cfg["csm"]["f_min_hz"]),
        f_max_hz=float(stage_cfg["csm"]["f_max_hz"]),
    )
    duration_s = float(cfg["segment"]["duration"])
    mic_positions = load_mic_geom_xml(cfg["input"]["mic_geom_xml"])

    psd_pre, freqs = _mod.compute_psd_pre(
        Path(cfg["input"]["audio_h5"]), mic_positions, target, duration_s, csm_cfg,
    )
    gt_psd = _mod.gt_psd_at(
        Path(cfg["input"]["ground_truth_h5"]), freqs, csm_cfg.nperseg, csm_cfg.noverlap,
    )
    s_q = float(gt_psd.mean())

    # All available method outputs
    method_psds: dict[str, np.ndarray] = {"psd_pre": psd_pre}
    for sibling in sorted(args.run_dir.parent.glob("ext_only_*/residual_csm.h5")):
        if "2026-05-07" not in sibling.parent.name:
            continue
        method = sibling.parent.name.replace("ext_only_", "").rsplit("_2026-", 1)[0]
        csm, fr = _mod.load_residual_csm(sibling)
        if not np.allclose(fr, freqs):
            continue
        method_psds[f"psd_post[{method}]"] = steer_to_psd(csm, fr, mic_positions, target)

    # Analytical transfer at coarse (bin) and fine grid
    T_bin = _at_mod.array_transfer_at_target(mic_positions, target, freqs)
    method_psds["analytical T·S_q/(4π)²"] = s_q * T_bin / (4 * np.pi) ** 2

    f_fine = np.linspace(freqs.min(), freqs.max(), 2000)
    T_fine = _at_mod.array_transfer_at_target(mic_positions, target, f_fine)

    # ----- Correlation matrix (in dB) -----
    names = list(method_psds.keys())
    n = len(names)
    C = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            C[i, j] = correlate(method_psds[ni], method_psds[nj], in_db=True)

    # ----- Difference plot: psd_post / psd_pre per method (isolates the algorithm effect) -----
    fig = sp.make_subplots(
        rows=2, cols=1, vertical_spacing=0.10,
        subplot_titles=(
            "All raw spectra (dB) — same vertical axis",
            "Algorithm transfer: psd_post(method) / psd_pre [dB] — isolates per-method effect",
        ),
    )
    palette = {
        "psd_pre": "#000000",
        "psd_post[doa_drone_cone]": "#1f77b4",
        "psd_post[doa_rotor_cone]": "#ff7f0e",
        "psd_post[doa_target_cone]": "#d62728",
        "psd_post[nnls]": "#2ca02c",
        "analytical T·S_q/(4π)²": "#9467bd",
    }

    for name, p in method_psds.items():
        color = palette.get(name, "#888888")
        width = 2.5 if name == "psd_pre" else (1.5 if "analytical" in name else 1.2)
        dash = "dash" if "analytical" in name else None
        fig.add_trace(go.Scatter(
            x=freqs, y=db(p), mode="lines", name=name,
            line={"color": color, "width": width, "dash": dash},
            opacity=0.85, legendgroup="raw", showlegend=True,
        ), row=1, col=1)
    # Fine-grid analytical T overlay (rescaled into the band)
    fig.add_trace(go.Scatter(
        x=f_fine, y=db(s_q * T_fine / (4 * np.pi) ** 2),
        mode="lines", name="analytical T (fine grid)",
        line={"color": "#9467bd", "width": 1, "dash": "dot"},
        opacity=0.8, legendgroup="raw",
    ), row=1, col=1)

    # Algorithm-only effect: ratio per method
    for name, p in method_psds.items():
        if "psd_post" not in name:
            continue
        fig.add_trace(go.Scatter(
            x=freqs, y=db(p) - db(psd_pre), mode="lines",
            name=name + " − psd_pre",
            line={"color": palette.get(name, "#888"), "width": 1.2},
            legendgroup="diff", showlegend=False,
        ), row=2, col=1)
    fig.add_hline(y=0, line={"color": "#000", "width": 1}, row=2, col=1)

    fig.update_xaxes(title_text="frequency [Hz]", row=2, col=1)
    fig.update_yaxes(title_text="PSD [dB re 1 Pa²/Hz]", row=1, col=1)
    fig.update_yaxes(title_text="Δ vs psd_pre [dB]", row=2, col=1)
    fig.update_layout(
        title=f"Spike-Origin-Analyse — target={target}, M={mic_positions.shape[0]} mics",
        height=900, template="plotly_white",
        legend={"yanchor": "top", "y": 0.98, "xanchor": "right", "x": 0.98},
    )

    out = args.out or args.run_dir / "spike_origin_analysis.html"
    fig.write_html(out, include_plotlyjs="cdn")

    # ----- Print correlation matrix -----
    print(f"wrote {out}")
    print()
    print("Correlation matrix (dB-space, Pearson):")
    print(f"  {'':32s} " + " ".join(f"{i:>7d}" for i in range(n)))
    for i, ni in enumerate(names):
        row = " ".join(f"{C[i,j]:>+7.3f}" for j in range(n))
        print(f"  ({i}) {ni:28s} {row}")

    # Summary verdicts
    print()
    print("Geometric ripple — fine-grid analytical T(f):")
    print(f"  T_fine min = {db(T_fine.min()):+.2f} dB,  T_fine max = {db(T_fine.max()):+.2f} dB,")
    print(f"  RMS-Welligkeit über [200,6000] Hz: {np.std(db(T_fine)):.2f} dB")
    print(f"  Anzahl lokaler Maxima im fine grid: "
          f"{int(np.sum((T_fine[1:-1] > T_fine[:-2]) & (T_fine[1:-1] > T_fine[2:])))}")


if __name__ == "__main__":
    main()
