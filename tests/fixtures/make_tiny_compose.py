"""Build the tiny composed mix + ground-truth files used by Stage-2 tests."""
from pathlib import Path

from martymicfly.synth.compose_external import compose_external
from martymicfly.synth.external_source import ExternalSourceSpec


def main():
    here = Path(__file__).parent
    art = here / "tiny_drone_artifact.h5"
    geom = here / "tiny_geom_4mic.xml"
    out_synth = here / "tiny_synth_mixed.h5"
    out_gt = here / "tiny_gt.h5"

    spec = ExternalSourceSpec(
        kind="noise",
        position_m=(0.5, 0.0, -0.5),
        amplitude_db=-6.0,
        seed=42,
    )
    compose_external(str(art), str(geom), spec, str(out_synth), str(out_gt))
    print(f"wrote {out_synth} and {out_gt}")


if __name__ == "__main__":
    main()
