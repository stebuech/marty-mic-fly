# Importing stage modules here ensures their builders register with the
# stage registry in pipeline.py at package import time. Without this, callers
# that only `from martymicfly.processing.pipeline import build_stage` would
# see an empty registry.
from . import notch  # noqa: F401  (side-effect: registers "notch" builder)
