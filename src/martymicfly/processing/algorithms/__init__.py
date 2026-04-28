"""Algorithm registry for ArrayFilterStage. Concrete algorithms register
themselves on import."""
from martymicfly.processing.algorithms.base import (
    Algorithm,
    SourceMap,
    reconstruct_csm,
)

ALGORITHM_REGISTRY: dict[str, type[Algorithm]] = {}


def register_algorithm(cls: type[Algorithm]) -> type[Algorithm]:
    ALGORITHM_REGISTRY[cls.name] = cls
    return cls
