"""Comprehensive Stage-2 methods comparison report.

Runs the three Stage-2 methods (Cartesian + CLEAN-SC, DOA hemisphere +
CLEAN-SC, NNLS on known atoms) on both the external-only methodology
reference and the mixed (drone + external) production data, then emits
a single multi-panel HTML report:

    1. Setup summary
    2. Localization tables per method × scenario
    3. Subtraction aggressiveness (csm_red) per band/mode/method
    4. Methodology floor: ext-only false-positive subtraction
    5. Mixed-data external-source recovery (ext_rec, MAE)
    6. Cross-term damage: ext_rec(mixed) − ext_rec(ext-only)
    7. Pseudo-target PSD overlay (mixed)

Usage:
    python -m martymicfly.cli.methods_comparison_report \\
        [--out path/to/report.html]

Default output: results/methods_comparison_report.html.
The set of configs evaluated is hardcoded (six scenarios). Edit
``_SCENARIO_GROUPS`` in this file to evaluate a different set.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

from martymicfly.cli.compare_modes import (
    _classify_method,
    _first_array_filter,
    _load_segment_and_gt,
    compare_doa_modes,
    compare_mask_modes,
    compare_nnls,
)
from martymicfly.config import AppConfig

log = logging.getLogger("martymicfly.methods_comparison_report")


_SCENARIO_GROUPS: dict[str, dict[str, str]] = {
    "ext_only": {
        "cartesian": "configs/pipeline_external_only_target_box.yaml",
        "doa": "configs/pipeline_external_only_doa_target_cone.yaml",
        "nnls": "configs/pipeline_external_only_nnls.yaml",
    },
    "mixed": {
        "cartesian": "configs/example_pipeline.yaml",
        "doa": "configs/pipeline_mixed_doa_target_cone.yaml",
        "nnls": "configs/pipeline_mixed_nnls.yaml",
    },
}


@dataclass
class _MethodResult:
    method: str             # cartesian | doa | nnls
    scenario: str           # ext_only | mixed
    config_path: str
    report: dict


def _run_one(scenario: str, method: str, cfg_path: str) -> _MethodResult:
    log.info("[%s/%s] %s", scenario, method, cfg_path)
    cfg = AppConfig.model_validate(yaml.safe_load(Path(cfg_path).read_text()))
    stage_cfg = _first_array_filter(cfg)
    classified = _classify_method(stage_cfg)
    if classified != method:
        log.warning(
            "[%s/%s] config classifies as %r, not %r — using config's classification",
            scenario, method, classified, method,
        )
        method = classified

    src, geom, td, fs, gt_block = _load_segment_and_gt(cfg, stage_cfg)
    common = dict(time_data=td, sample_rate=fs, mic_positions=geom,
                  platform=src["platform"], stage_cfg=stage_cfg,
                  gt_block=gt_block)
    if method == "doa":
        report = compare_doa_modes(**common)
    elif method == "nnls":
        report = compare_nnls(**common)
    else:
        report = compare_mask_modes(**common)
    return _MethodResult(method=method, scenario=scenario,
                          config_path=cfg_path, report=report)


def _collect_scalar(
    results: list[_MethodResult],
    metric: str,                # 'csm_trace_reduction_db' | 'target_psd_reduction_db' |
                                # 'ground_truth.external_recovery_db' |
                                # 'ground_truth.spectrum_mae_db'
    scenario: str | None = None,
) -> list[tuple[str, str, str, float]]:
    """Return list of (label, mode, band, value) entries for a chosen metric."""
    rows: list[tuple[str, str, str, float]] = []
    for r in results:
        if scenario is not None and r.scenario != scenario:
            continue
        for mode, m in r.report["per_mode"].items():
            for band_name, b in m["bands"].items():
                if "." in metric:
                    a, k = metric.split(".", 1)
                    sub = b.get(a) or {}
                    val = sub.get(k)
                else:
                    val = b.get(metric)
                if val is None or not np.isfinite(val):
                    continue
                label = f"{r.method}/{mode}"
                rows.append((label, mode, band_name, float(val)))
    return rows


def _bar_per_band_grouped(
    rows: list[tuple[str, str, str, float]],
    title: str,
    y_label: str,
    out_traces: list[go.Bar],
    band_order: list[str],
    color_map: dict[str, str],
) -> None:
    """Group rows by label (= method/mode), draw one bar per band on x.

    Bars share x-axis ['low','mid','high']; one bar group per (method,mode).
    """
    labels_in_order: list[str] = []
    for lab, _, _, _ in rows:
        if lab not in labels_in_order:
            labels_in_order.append(lab)
    for lab in labels_in_order:
        ys = []
        for band in band_order:
            v = next(
                (val for (lab2, _mode, b2, val) in rows
                 if lab2 == lab and b2 == band),
                None,
            )
            ys.append(v if v is not None else float("nan"))
        out_traces.append(go.Bar(
            x=band_order, y=ys, name=lab,
            marker_color=color_map.get(lab.split("/")[0], None),
            hovertemplate=f"{lab}<br>%{{x}} band<br>{y_label}=%{{y:.2f}}<extra></extra>",
        ))


def _localization_html_blocks(results: list[_MethodResult]) -> str:
    """Build an HTML <pre> block per (scenario, method) summarizing
    localization."""
    parts: list[str] = []
    for r in results:
        loc = r.report["localization"]
        parts.append(f"<h4>{r.scenario} / {r.method}</h4>")
        if loc.get("kind") == "nnls_atom_powers":
            lines = [f"NNLS atom-power shares — target=("
                     f"{loc['target_xyz_m'][0]:+.2f},{loc['target_xyz_m'][1]:+.2f},"
                     f"{loc['target_xyz_m'][2]:+.2f}) m"]
            lines.append(f"{'band':<6}{'drone [dB]':>12}{'target [dB]':>13}{'diffuse [dB]':>14}")
            for name, b in loc["bands"].items():
                d = b["drone_db"]; t = b["target_db"]; df = b["diffuse_db"]
                d_s = f"{d:>11.2f}" if np.isfinite(d) else f"{'-inf':>11}"
                t_s = f"{t:>12.2f}" if np.isfinite(t) else f"{'-inf':>12}"
                df_s = f"{df:>13.2f}" if np.isfinite(df) else f"{'n/a':>13}"
                lines.append(f"{name:<6}{d_s} {t_s} {df_s}")
            parts.append(f"<pre>{chr(10).join(lines)}</pre>")
        else:
            tx, ty, tz = loc["target_xyz_m"]
            ztol = loc["z_tolerance_m"]; dtol = loc["drone_z_tolerance_m"]
            lines = [
                f"CLEAN-SC localization — target=({tx:+.2f},{ty:+.2f},{tz:+.2f}) m, "
                f"grid={loc['grid_shape']}",
                f"{'band':<6}{'peak (x,y,z) [m]':<26}{'z 10-90% [m]':<22}"
                f"{'@target±'+str(ztol)+'m':<14}{'@drone±'+str(dtol)+'m':<14}",
            ]
            for name, b in loc["bands"].items():
                px, py, pz = b["peak_xyz_m"]
                z10, z90 = b["z_span_10_90_m"]
                lines.append(
                    f"{name:<6}({px:+.2f},{py:+.2f},{pz:+.2f})       "
                    f"[{z10:+.2f},{z90:+.2f}]      "
                    f"{b['frac_at_target_z']:>9.2%}     {b['frac_at_drone_z']:>9.2%}"
                )
            parts.append(f"<pre>{chr(10).join(lines)}</pre>")
    return "\n".join(parts)


def _ext_rec_floor_table(results: list[_MethodResult]) -> str:
    """ext_rec ext-only vs mixed and the cross-term Δ as an HTML table."""
    by_label_band: dict[tuple[str, str, str], float] = {}
    for r in results:
        for mode, m in r.report["per_mode"].items():
            for band, b in m["bands"].items():
                gt = b.get("ground_truth") or {}
                er = gt.get("external_recovery_db")
                if er is None:
                    continue
                by_label_band[(r.scenario, f"{r.method}/{mode}", band)] = float(er)

    methods_modes: list[str] = []
    for (_, lab, _) in by_label_band.keys():
        if lab not in methods_modes:
            methods_modes.append(lab)
    bands = ["low", "mid", "high"]

    rows = ["<table border='1' cellpadding='4' style='border-collapse: collapse;'>"]
    rows.append("<tr><th>method/mode</th><th>band</th>"
                "<th>ext_rec ext-only [dB]</th><th>ext_rec mixed [dB]</th>"
                "<th>Δ [dB] (cross-term)</th></tr>")
    for lab in methods_modes:
        for band in bands:
            ext = by_label_band.get(("ext_only", lab, band))
            mix = by_label_band.get(("mixed", lab, band))
            if ext is None and mix is None:
                continue
            delta = (mix - ext) if (ext is not None and mix is not None) else None
            ext_s = f"{ext:+.2f}" if ext is not None else ""
            mix_s = f"{mix:+.2f}" if mix is not None else ""
            delta_s = f"{delta:+.2f}" if delta is not None else ""
            rows.append(
                f"<tr><td>{lab}</td><td>{band}</td>"
                f"<td>{ext_s}</td><td>{mix_s}</td><td>{delta_s}</td></tr>"
            )
    rows.append("</table>")
    return "\n".join(rows)


def _build_report_html(
    out_path: Path,
    results: list[_MethodResult],
) -> None:
    band_order = ["low", "mid", "high"]
    color_map = {"cartesian": "#d62728", "doa": "#1f77b4", "nnls": "#2ca02c"}

    # --- Section 3 + 4 + 5: Bar plots ---
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            "csm_red [dB] — ext-only (false-positive subtraction)",
            "csm_red [dB] — mixed (drone + external)",
            "ext_rec [dB] — ext-only (methodology floor)",
            "ext_rec [dB] — mixed (filter performance)",
            "spectrum_mae [dB] — ext-only",
            "spectrum_mae [dB] — mixed",
            "drone_share [dB] — ext-only",
            "drone_share [dB] — mixed",
        ],
        vertical_spacing=0.10, horizontal_spacing=0.08,
    )
    for i, scenario in enumerate(("ext_only", "mixed")):
        col = i + 1
        for row, metric, ylabel in [
            (1, "csm_trace_reduction_db", "csm_red"),
            (2, "ground_truth.external_recovery_db", "ext_rec"),
            (3, "ground_truth.spectrum_mae_db", "mae"),
            (4, "drone_power_share_db", "drone_share"),
        ]:
            rows_ = _collect_scalar(results, metric, scenario=scenario)
            traces: list[go.Bar] = []
            _bar_per_band_grouped(rows_, "", ylabel, traces, band_order, color_map)
            for t in traces:
                # only show legend in first subplot to avoid duplicate entries
                t.showlegend = (row == 1 and col == 1)
                fig.add_trace(t, row=row, col=col)

    fig.update_layout(
        title="Stage-2 method comparison — Cartesian vs DOA vs NNLS",
        barmode="group",
        height=1200, width=1500,
        legend=dict(orientation="v", x=1.02, y=1.0),
    )

    # --- Section 7: target PSD overlay (mixed) ---
    psd_fig = go.Figure()
    psd_fig.add_annotation(
        text=("Pseudo-target PSDs — mixed input, all methods/modes overlaid. "
              "Lower psd_post matches GT ⇔ better filter."),
        xref="paper", yref="paper", x=0.0, y=1.08, showarrow=False,
    )
    plotted_pre = False
    plotted_gt = False
    for r in results:
        if r.scenario != "mixed":
            continue
        freqs = r.report.get("frequencies")
        psd_pre = r.report.get("psd_pre")
        gt_psd = r.report.get("gt_psd")
        if freqs is None or psd_pre is None:
            continue
        if not plotted_pre:
            psd_fig.add_trace(go.Scatter(
                x=freqs, y=10 * np.log10(np.maximum(psd_pre, 1e-30)),
                mode="lines", name="pre (mixed)", line=dict(color="black", width=2),
            ))
            plotted_pre = True
        if gt_psd is not None and not plotted_gt:
            psd_fig.add_trace(go.Scatter(
                x=freqs, y=10 * np.log10(np.maximum(gt_psd, 1e-30)),
                mode="lines", name="ground truth (external)",
                line=dict(color="gray", dash="dot", width=2),
            ))
            plotted_gt = True
        for mode, m in r.report["per_mode"].items():
            psd_post = m.get("psd_post")
            if psd_post is None:
                continue
            psd_fig.add_trace(go.Scatter(
                x=freqs, y=10 * np.log10(np.maximum(psd_post, 1e-30)),
                mode="lines", name=f"post — {r.method}/{mode}",
                line=dict(color=color_map.get(r.method, None), width=1),
                opacity=0.85,
            ))
    psd_fig.update_xaxes(title_text="Frequency [Hz]")
    psd_fig.update_yaxes(title_text="PSD [dB rel. unit²/Hz]")
    psd_fig.update_layout(
        title="Pseudo-target PSD — mixed input, all methods",
        height=600, width=1500,
        legend=dict(orientation="v", x=1.02, y=1.0),
    )

    # --- Compose the full HTML page ---
    intro = []
    for scenario in ("ext_only", "mixed"):
        intro.append(f"<li><b>{scenario}</b>:")
        for r in results:
            if r.scenario == scenario:
                intro.append(f"<br>&nbsp;&nbsp;{r.method}: <code>{r.config_path}</code>")
        intro.append("</li>")
    intro_html = "<ul>" + "".join(intro) + "</ul>"

    bars_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    psd_html = psd_fig.to_html(full_html=False, include_plotlyjs=False)
    loc_html = _localization_html_blocks(results)
    floor_html = _ext_rec_floor_table(results)

    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stage-2 methods comparison</title>
<style>
  body {{ font-family: sans-serif; max-width: 1600px; margin: 1em auto; padding: 0 1em; }}
  h1, h2, h3, h4 {{ color: #222; }}
  pre {{ background: #f5f5f5; padding: 0.5em; overflow-x: auto; font-size: 0.85em; }}
  code {{ background: #f5f5f5; padding: 0 0.2em; }}
  table {{ font-size: 0.9em; margin: 0.5em 0; }}
  th {{ background: #eee; }}
</style></head><body>
<h1>Stage-2 methods comparison — Cartesian vs DOA vs NNLS</h1>

<h2>1. Setup</h2>
<p>Six runs: three methods (Cartesian + CLEAN-SC, DOA hemisphere + CLEAN-SC,
known-geometry NNLS) × two scenarios (external-only methodology reference,
mixed drone + external).</p>
{intro_html}

<h2>2. Localization summary</h2>
<p>Where each method places the source-map energy. CLEAN-SC variants list
peak xyz, z-span and the energy fractions near the true target z and the
drone z. NNLS lists per-band power shares per atom kind.</p>
{loc_html}

<h2>3. Per-band metrics — bar plots</h2>
<p>Left column: ext-only (any subtraction is a false positive). Right column:
mixed (the production case).</p>
{bars_html}

<h2>4. External-source recovery floor + cross-term damage</h2>
<p>ext_rec on ext-only is the methodology floor (≈ −26 dB phase-only steering
bias). ext_rec on mixed shows what the filter actually does. Δ = mixed − ext-only
is the additional damage caused by the drone-source cross term: large
positive Δ means the filter leaves drone leakage in the residual; large
negative Δ means it subtracts external-source energy by mistake.</p>
{floor_html}

<h2>5. Pseudo-target PSD overlay (mixed)</h2>
<p>The closer post-filter PSD lies to the GT line, the better the filter.
The CLEAN-SC steering bias (~−26 dB) shifts everything below GT — only the
relative shape and offset between methods is informative.</p>
{psd_html}

<hr>
<p><i>Generated by martymicfly.cli.methods_comparison_report</i></p>
</body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    log.info("wrote %s (%.1f kB)", out_path, len(body) / 1024)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="methods-comparison-report")
    p.add_argument("--out", default="results/methods_comparison_report.html")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    results: list[_MethodResult] = []
    for scenario, methods in _SCENARIO_GROUPS.items():
        for method, cfg_path in methods.items():
            results.append(_run_one(scenario, method, cfg_path))

    _build_report_html(Path(args.out), results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
