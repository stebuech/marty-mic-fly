"""Study runner — expandiert Studie-YAML in Liste von Run-Specs und führt sie aus.

Schema einer Studie:
  phase: baseline | phase1 | phase2
  configs: list of base-config yaml paths
  scenarios: list of S0..S3
  axes: dict[dotted_key, list_of_values]
  output_root: results-dir

Pro Achse wird ein einzelner-Knopf-Sweep generiert (alle anderen Achsen bleiben
auf base-config-Defaults). Plus 1 Baseline-Run pro (config, scenario) ohne
Override."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from analysis.separation_study.run_cache import config_hash, is_run_complete

log = logging.getLogger("separation_study.runner")


def _slug(value: Any) -> str:
    return str(value).replace("/", "_").replace(".", "p").replace(" ", "_")


def expand_run_plan(study: dict) -> list[dict]:
    """Pro (config, scenario) ein Baseline-Run + pro (axis, value) ein Override-Run."""
    plan: list[dict] = []
    output_root = Path(study["output_root"])
    for cfg_path in study["configs"]:
        cfg_name = Path(cfg_path).stem
        for scenario in study["scenarios"]:
            plan.append({
                "config": cfg_path,
                "scenario": scenario,
                "overrides": {},
                "run_id": f"{cfg_name}__{scenario}__baseline",
                "output_dir": str(output_root /
                                  f"{cfg_name}__{scenario}__baseline"),
            })
            for axis, values in study.get("axes", {}).items():
                for v in values:
                    overrides = {axis: v}
                    h = config_hash(overrides)
                    rid = f"{cfg_name}__{scenario}__{_slug(axis)}_{_slug(v)}_{h}"
                    plan.append({
                        "config": cfg_path,
                        "scenario": scenario,
                        "overrides": overrides,
                        "run_id": rid,
                        "output_dir": str(output_root / rid),
                    })
    return plan


def execute_plan(plan: list[dict], scenario_paths: dict, *,
                 force: bool = False) -> list[dict]:
    """Führt jeden Run-Spec aus, überspringt bereits komplette Runs."""
    from analysis.separation_study.pipeline_wrapper import (
        augment_metrics, run_pipeline_with_overrides,
    )
    from analysis.separation_study.synth_scenarios import ensure_scenario
    from martymicfly.io.mic_geom import load_mic_geom_xml

    results: list[dict] = []
    for spec in plan:
        run_dir = Path(spec["output_dir"])
        if not force and is_run_complete(run_dir):
            log.info("skip %s — already complete", spec["run_id"])
            results.append({**spec, "status": "skipped"})
            continue

        scenario = spec["scenario"]
        sp = scenario_paths[scenario]

        # Synth-on-demand for non-S0 scenarios. Generates both the mixed
        # (drone + ext) and ext_only files in one go so that D_ref stays
        # consistent (same seed → same ext-signal → drone = mixed − ext).
        if "audio_h5" in sp and "ground_truth_h5" in sp:
            from analysis.separation_study.synth_scenarios import (
                ensure_scenario_pair,
            )
            ensure_scenario_pair(
                scenario=scenario,
                base_drone_artifact=sp["drone_artifact"],
                mic_geom_xml=sp["mic_geom_xml"],
                out_mixed_h5=Path(sp["audio_h5"]),
                out_mixed_gt_h5=Path(sp["ground_truth_h5"]),
                out_ext_only_h5=Path(sp["ext_only_audio_h5"]),
                out_ext_only_gt_h5=Path(sp["ext_only_gt_h5"]),
            )

        cfg_path = Path(spec["config"])
        cfg_yaml = yaml.safe_load(cfg_path.read_text())
        af_stage_cfg = next(s for s in cfg_yaml["stages"]
                            if s.get("kind") == "array_filter")

        overrides = dict(spec["overrides"])
        # scenario_paths can declare config_overrides that apply to ALL runs
        # of the scenario (e.g., target_point_m for off-axis source).
        # Sweep axes in `spec["overrides"]` take precedence.
        for k, v in sp.get("config_overrides", {}).items():
            overrides.setdefault(k, v)
        # Only override input paths when scenario_paths supplies them — otherwise
        # the config keeps its own audio_h5/ground_truth_h5 (per-config-correct
        # for S0 where ext_only-configs and mixed-configs use different files).
        if "audio_h5" in sp:
            overrides.setdefault("input.audio_h5", sp["audio_h5"])
        if "ground_truth_h5" in sp:
            overrides.setdefault("input.ground_truth_h5", sp["ground_truth_h5"])
        # Disk-saving: study runs don't need per-channel notch plots (~14 MB
        # per mixed run × 72+ runs = excessive). Study has its own plots.
        overrides.setdefault("plots.enabled", False)

        # Auto-fix: nperseg override must not break existing noverlap (scipy
        # requires noverlap < nperseg). When the override drops nperseg below
        # the config's noverlap, halve noverlap so the constraint holds.
        nperseg_key = "array_filter.csm.nperseg"
        noverlap_key = "array_filter.csm.noverlap"
        if nperseg_key in overrides and noverlap_key not in overrides:
            new_nperseg = int(overrides[nperseg_key])
            current_noverlap = int(af_stage_cfg["csm"]["noverlap"])
            if current_noverlap >= new_nperseg:
                overrides[noverlap_key] = new_nperseg // 2

        log.info("run %s overrides=%s", spec["run_id"], overrides)
        actual_run_dir = run_pipeline_with_overrides(
            base_config_path=cfg_path,
            overrides=overrides,
            output_dir=run_dir,
        )

        # Metric augmentation needs config bands + mic_positions + target.
        # Use the OVERRIDE-aware view of csm params so the augmentation is
        # apples-to-apples with what the pipeline actually computed.
        af_stage = af_stage_cfg
        bands = af_stage["bands"]
        target_point = tuple(overrides.get(
            "array_filter.target_point_m", af_stage["target_point_m"]))
        nperseg = int(overrides.get(nperseg_key, af_stage["csm"]["nperseg"]))
        noverlap = int(overrides.get(noverlap_key, af_stage["csm"]["noverlap"]))
        window = af_stage["csm"]["window"]
        diag = float(overrides.get("array_filter.csm.diag_loading_rel",
                                   af_stage["csm"].get("diag_loading_rel", 0.0)))
        f_min = float(af_stage["csm"].get("f_min_hz", 0.0))
        f_max = float(af_stage["csm"].get("f_max_hz", 1.0e9))
        seg_duration = float(cfg_yaml.get("segment", {}).get("duration", 10.0))
        seg_mode = str(cfg_yaml.get("segment", {}).get("mode", "middle"))
        mic_positions = load_mic_geom_xml(sp["mic_geom_xml"])

        ext_gt = Path(sp["ext_only_gt_h5"])
        # Drone-only D_ref is built by subtracting ext-only mic-array audio
        # from mixed mic-array audio at the array level, then steering.
        ext_only_audio = Path(sp["ext_only_audio_h5"])
        # Mixed audio path: prefer mixed_audio_h5 (S0), fall back to audio_h5
        # (S1-S3 where the synth-on-demand uses audio_h5 = mixed file).
        mixed_audio = Path(sp.get("mixed_audio_h5") or sp["audio_h5"])
        augment_metrics(
            run_dir=actual_run_dir, ext_gt_h5=ext_gt,
            ext_only_audio_h5=ext_only_audio, mixed_audio_h5=mixed_audio,
            bands=bands, mic_positions=mic_positions, target_point=target_point,
            welch_nperseg=nperseg, welch_noverlap=noverlap, window=window,
            welch_floor_db=-50.0,
            csm_diag_loading=diag, f_min_hz=f_min, f_max_hz=f_max,
            segment_duration_s=seg_duration, segment_mode=seg_mode,
        )

        (actual_run_dir / "run_meta.json").write_text(json.dumps({
            "run_id": spec["run_id"],
            "config": spec["config"],
            "scenario": spec["scenario"],
            "overrides": spec["overrides"],
            "method": Path(spec["config"]).stem,
        }, indent=2))

        # Disk-saving cleanup: filtered.h5 is the post-notch audio (~350 MB
        # for mixed runs at 10s × 16 mics × 51200 Hz), not needed for the
        # study (we keep residual_csm.h5, metrics*, run_meta).
        for big in ("filtered.h5",):
            p = actual_run_dir / big
            if p.exists():
                p.unlink()
        # Also drop the plots/ subdir (per-channel notch plots) if it slipped
        # past the plots.enabled override.
        plots_dir = actual_run_dir / "plots"
        if plots_dir.exists():
            import shutil as _shutil
            _shutil.rmtree(plots_dir)

        results.append({**spec, "status": "ok",
                        "actual_run_dir": str(actual_run_dir)})
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--scenario-paths", type=Path, required=True,
                   help="YAML mapping scenario → file paths")
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    study = yaml.safe_load(args.study.read_text())
    scenario_paths = yaml.safe_load(args.scenario_paths.read_text())
    plan = expand_run_plan(study)
    log.info("expanded %d runs", len(plan))
    results = execute_plan(plan, scenario_paths, force=args.force)
    summary = {"total": len(results),
               "ok": sum(1 for r in results if r["status"] == "ok"),
               "skipped": sum(1 for r in results if r["status"] == "skipped")}
    log.info("done: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
