import numpy as np
import pytest


def test_external_source_spec_validates_required_fields():
    from martymicfly.synth.external_source import ExternalSourceSpec
    spec = ExternalSourceSpec(kind="noise", duration_s=0.5, seed=0)
    assert spec.kind == "noise"
    assert spec.position_m == (0.0, 0.0, -1.5)


def test_external_source_spec_rejects_sine_without_freq():
    from martymicfly.synth.external_source import ExternalSourceSpec
    with pytest.raises(ValueError, match="sine_freq_hz"):
        ExternalSourceSpec(kind="sine", duration_s=0.5)


def test_external_source_generate_noise_deterministic():
    from martymicfly.synth.external_source import (
        ExternalSourceSpec,
        generate_external_signal,
    )
    spec = ExternalSourceSpec(kind="noise", duration_s=0.5, seed=42)
    fs = 16000.0
    s1 = generate_external_signal(spec, fs, n_samples=int(0.5 * fs))
    s2 = generate_external_signal(spec, fs, n_samples=int(0.5 * fs))
    np.testing.assert_array_equal(s1, s2)
    assert s1.shape == (8000,)


def test_external_source_generate_sine_correct_frequency():
    from martymicfly.synth.external_source import (
        ExternalSourceSpec,
        generate_external_signal,
    )
    spec = ExternalSourceSpec(kind="sine", duration_s=1.0, sine_freq_hz=500.0)
    fs = 16000.0
    sig = generate_external_signal(spec, fs, n_samples=fs.__int__())
    spec_fft = np.abs(np.fft.rfft(sig))
    f = np.fft.rfftfreq(len(sig), 1 / fs)
    peak_f = f[np.argmax(spec_fft)]
    assert abs(peak_f - 500.0) < 2.0
