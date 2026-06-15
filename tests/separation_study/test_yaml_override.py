def test_dotted_key_simple():
    from analysis.separation_study.yaml_override import apply_overrides
    base = {"a": {"b": {"c": 1}}}
    out = apply_overrides(base, {"a.b.c": 42})
    assert out == {"a": {"b": {"c": 42}}}


def test_dotted_key_does_not_mutate_input():
    from analysis.separation_study.yaml_override import apply_overrides
    base = {"a": {"b": 1}}
    out = apply_overrides(base, {"a.b": 2})
    assert base["a"]["b"] == 1


def test_stage_index_path():
    """stages[0].csm.nperseg patcht den ersten Stage."""
    from analysis.separation_study.yaml_override import apply_overrides
    base = {"stages": [{"kind": "array_filter", "csm": {"nperseg": 512}}]}
    out = apply_overrides(base, {"stages[0].csm.nperseg": 1024})
    assert out["stages"][0]["csm"]["nperseg"] == 1024


def test_missing_path_raises():
    from analysis.separation_study.yaml_override import apply_overrides
    import pytest
    with pytest.raises(KeyError):
        apply_overrides({"a": 1}, {"b.c": 5})


def test_array_filter_stage_finds_first():
    """Convenience: 'array_filter.csm.nperseg' findet den ersten array_filter
    stage automatisch (Notch-Stage in mixed-configs überspringen)."""
    from analysis.separation_study.yaml_override import apply_overrides
    base = {"stages": [
        {"kind": "notch", "pole_radius": 0.999},
        {"kind": "array_filter", "csm": {"nperseg": 512}},
    ]}
    out = apply_overrides(base, {"array_filter.csm.nperseg": 256})
    assert out["stages"][1]["csm"]["nperseg"] == 256
