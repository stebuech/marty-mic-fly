"""Backward-compatible shim: forwards to martymicfly.cli.run_pipeline."""
from martymicfly.cli.run_pipeline import main


if __name__ == "__main__":
    _rc = main()
    if _rc:
        raise SystemExit(_rc)
