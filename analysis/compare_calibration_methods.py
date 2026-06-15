"""Overlay calibrated psd_post curves from multiple ext_only run dirs.

Builds a single HTML with one calibrated `psd_post · cal` line per method,
plus the ground-truth and the (method-independent) `psd_pre · cal` line.

Usage:
    uv run python analysis/compare_calibration_methods.py \\
        results/pipeline/ext_only_doa_drone_cone_2026-05-07T13-45-57_1dacd20f \\
        results/pipeline/ext_only_doa_rotor_cone_2026-05-07T13-46-16_783203ee \\
        results/pipeline/ext_only_doa_target_cone_2026-05-07T12-07-19_37a7abbf \\
        results/pipeline/ext_only_nnls_2026-05-07T13-46-36_b86628f6 \\
        --out results/ext_only_methods_calibrated.html
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import yaml

# Reuse helpers from the per-run script
import importlib.util
import sys

_SCRIPT = Path(__file__).resolve().parent / "plot_calibration_effect.py"
_spec = importlib.util.spec_from_file_location("plot_calibration_effect", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["plot_calibration_effect"] = _mod
_spec.loader.exec_module(_mod)

from martymicfly.io.mic_geom import load_mic_geom_xml
from martymicfly.processing.csm import CsmConfig
from martymicfly.processing.steering import range_compensation_factor, steer_to_psd


_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def method_name_from_run_dir(run_dir: Path) -> str:
    name = run_dir.name
    m = re.match(r"^ext_only_(.+?)_2026-", name)
    return m.group(1) if m else name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg0 = yaml.safe_load((args.run_dirs[0] / "config.yaml").read_text())
    stage_cfg = cfg0["stages"][0]
    target = tuple(stage_cfg["target_point_m"])
    mic_positions = load_mic_geom_xml(cfg0["input"]["mic_geom_xml"])
    cal = range_compensation_factor(mic_positions, target)
    cal_db = 10 * np.log10(cal)

    csm_cfg = CsmConfig(
        nperseg=int(stage_cfg["csm"]["nperseg"]),
        noverlap=int(stage_cfg["csm"]["noverlap"]),
        window=stage_cfg["csm"]["window"],
        diag_loading_rel=float(stage_cfg["csm"]["diag_loading_rel"]),
        f_min_hz=float(stage_cfg["csm"]["f_min_hz"]),
        f_max_hz=float(stage_cfg["csm"]["f_max_hz"]),
    )
    duration_s = float(cfg0["segment"]["duration"])

    psd_pre_raw, freqs = _mod.compute_psd_pre(
        Path(cfg0["input"]["audio_h5"]), mic_positions, target, duration_s, csm_cfg,
    )
    gt_psd = _mod.gt_psd_at(
        Path(cfg0["input"]["ground_truth_h5"]), freqs, csm_cfg.nperseg, csm_cfg.noverlap,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=10 * np.log10(np.maximum(gt_psd, 1e-30)),
        mode="lines", name="ground_truth PSD",
        line={"color": "#2ca02c", "width": 3, "dash": "dash"},
    ))
    fig.add_trace(go.Scatter(
        x=freqs, y=10 * np.log10(np.maximum(psd_pre_raw * cal, 1e-30)),
        mode="lines", name="psd_pre · cal (ungefiltert, methodenunabhängig)",
        line={"color": "#aaaaaa", "width": 2, "dash": "dot"},
    ))

    summary = []
    for i, run_dir in enumerate(args.run_dirs):
        method = method_name_from_run_dir(run_dir)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        s = cfg["stages"][0]
        if tuple(s["target_point_m"]) != target:
            print(f"warning: {method} has different target_point, skipping")
            continue
        csm_post, freqs_post = _mod.load_residual_csm(run_dir / "residual_csm.h5")
        if not np.allclose(freqs_post, freqs):
            print(f"warning: {method} freq grid disagrees, skipping")
            continue
        psd_post_raw = steer_to_psd(csm_post, freqs_post, mic_positions, target)
        psd_post_cal = psd_post_raw * cal

        d_post_gt = 10 * np.log10(psd_post_cal.mean() / gt_psd.mean())
        d_pre_post = 10 * np.log10((psd_pre_raw * cal).mean() / psd_post_cal.mean())
        summary.append((method, d_pre_post, d_post_gt))

        fig.add_trace(go.Scatter(
            x=freqs, y=10 * np.log10(np.maximum(psd_post_cal, 1e-30)),
            mode="lines",
            name=f"{method} · cal — recovery {d_post_gt:+.2f} dB, subtract {d_pre_post:+.2f} dB",
            line={"color": _PALETTE[i % len(_PALETTE)], "width": 1.8},
        ))

    for b in stage_cfg["bands"]:
        fig.add_vline(x=b["f_min_hz"], line={"color": "#888", "width": 1, "dash": "dot"})
        fig.add_vline(x=b["f_max_hz"], line={"color": "#888", "width": 1, "dash": "dot"})
        fig.add_annotation(x=(b["f_min_hz"] + b["f_max_hz"]) / 2,
                           y=1.02, yref="paper", showarrow=False,
                           text=b["name"], font={"size": 11, "color": "#666"})

    fig.update_layout(
        title=(f"ext_only methods (calibrated, target={target}, "
               f"cal={cal:.1f} = {cal_db:+.2f} dB)"),
        xaxis_title="frequency [Hz]",
        yaxis_title="PSD [dB re 1 Pa²/Hz]",
        legend={"yanchor": "top", "y": 0.98, "xanchor": "right", "x": 0.98},
        template="plotly_white",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"wrote {args.out}")
    print()
    print(f"{'method':24s} {'pre→post (dB)':>14s} {'post→GT (dB)':>14s}")
    for method, dpp, dpg in sorted(summary, key=lambda r: abs(r[2])):
        print(f"{method:24s} {dpp:>+14.3f} {dpg:>+14.3f}")


if __name__ == "__main__":
    main()
