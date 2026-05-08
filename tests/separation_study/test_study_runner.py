import yaml
from pathlib import Path


def test_expand_run_plan_baseline(tmp_path):
    """baseline: 2 configs × 1 scenario = 2 Runs (keine axes)."""
    from analysis.separation_study.study_runner import expand_run_plan
    study = {
        "phase": "baseline",
        "configs": [
            "configs/pipeline_external_only_doa_target_cone.yaml",
            "configs/pipeline_mixed_doa_target_cone.yaml",
        ],
        "scenarios": ["S0"],
        "axes": {},
        "output_root": str(tmp_path / "out"),
    }
    plan = expand_run_plan(study)
    assert len(plan) == 2
    for r in plan:
        assert r["scenario"] == "S0"
        assert r["overrides"] == {}


def test_expand_run_plan_phase1_axes(tmp_path):
    """1 Config × 2 Achsen × 2 Punkte + 1 Baseline × 1 Szenario = 5 Runs."""
    from analysis.separation_study.study_runner import expand_run_plan
    study = {
        "phase": "phase1",
        "configs": ["configs/pipeline_mixed_doa_target_cone.yaml"],
        "scenarios": ["S0"],
        "axes": {
            "array_filter.csm.nperseg": [256, 1024],
            "array_filter.clean_sc.damp": [0.3, 0.9],
        },
        "output_root": str(tmp_path / "out"),
    }
    plan = expand_run_plan(study)
    assert len(plan) == 5
    baselines = [r for r in plan if not r["overrides"]]
    assert len(baselines) == 1


def test_expand_includes_unique_run_ids(tmp_path):
    from analysis.separation_study.study_runner import expand_run_plan
    plan = expand_run_plan({
        "phase": "baseline",
        "configs": ["configs/a.yaml", "configs/b.yaml"],
        "scenarios": ["S0", "S1"],
        "axes": {},
        "output_root": str(tmp_path / "out"),
    })
    ids = [r["run_id"] for r in plan]
    assert len(ids) == len(set(ids))
