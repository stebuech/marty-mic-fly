"""Pydantic models for martymicfly notch-pipeline configuration."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio_h5: str
    mic_geom_xml: str


class SegmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["middle", "head", "tail", "explicit"]
    duration: float | None = None
    start: float | None = None
    end: float | None = None

    @model_validator(mode="after")
    def _check_mode_fields(self) -> "SegmentConfig":
        if self.mode == "explicit":
            if self.start is None or self.end is None:
                raise ValueError("segment.mode=explicit requires start and end")
            if self.end <= self.start:
                raise ValueError("segment.end must be > segment.start")
        else:
            if self.duration is None or self.duration <= 0:
                raise ValueError(
                    f"segment.mode={self.mode} requires positive duration"
                )
        return self


class ChannelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selection: Literal["all", "list"]
    list: Optional[List[int]] = None

    @model_validator(mode="after")
    def _check_list(self) -> "ChannelsConfig":
        if self.selection == "list" and not self.list:
            raise ValueError("channels.selection=list requires non-empty list")
        return self


class RotorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_blades: int = Field(ge=1)
    n_harmonics: int = Field(ge=1)


class PoleRadiusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["scalar", "linear"]
    value: float | None = None
    k_cover: float | None = None
    margin_hz: float | None = None
    delta_bpf_hz: float | None = None
    r_min: float = 0.90
    r_max: float = 0.9995

    @model_validator(mode="after")
    def _check_mode_fields(self) -> "PoleRadiusConfig":
        if self.mode == "scalar":
            if self.value is None or not (0.0 < self.value < 1.0):
                raise ValueError("pole_radius.scalar requires value in (0, 1)")
        else:  # linear
            if self.k_cover is None or self.margin_hz is None:
                raise ValueError(
                    "pole_radius.linear requires k_cover and margin_hz"
                )
            if not (0.0 < self.r_min < self.r_max < 1.0):
                raise ValueError("require 0 < r_min < r_max < 1")
        return self


class NotchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pole_radius: PoleRadiusConfig
    multichannel: bool = False
    block_size: int = Field(default=4096, ge=64)


class NotchStageConfig(NotchConfig):
    """NotchConfig fields + a `kind` discriminator for the stages list."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["notch"]


class ArrayFilterStageConfig(BaseModel):
    # NOTE: placeholder until Task 19. extra="allow" is intentional during the
    # transition so Task 4 can shape-check stages-list YAMLs that already carry
    # an array_filter block. Task 19 MUST switch to extra="forbid" once the
    # full body lands — otherwise typos in YAML will be silently ignored.
    model_config = ConfigDict(extra="allow")
    kind: Literal["array_filter"]


StageConfig = Annotated[
    Union[NotchStageConfig, ArrayFilterStageConfig],
    Field(discriminator="kind"),
]


class MetricsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    welch_nperseg: int = Field(ge=64)
    welch_noverlap: int = Field(ge=0)
    bandwidth_factor: float = Field(default=1.0, gt=0.0)
    broadband_low_hz: float | None = None


class PlotsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    fmax_hz: float | None = None
    spectrogram_window: int = Field(ge=64)
    spectrogram_overlap: int = Field(ge=0)
    channel_subset: list[int] | None = None


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dir: str
    filtered_h5: str = "filtered.h5"
    metrics_json: str = "metrics.json"
    metrics_csv: str = "metrics.csv"
    plots_subdir: str = "plots"
    copy_config: bool = True


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: InputConfig
    segment: SegmentConfig
    channels: ChannelsConfig
    rotor: RotorConfig
    stages: list[StageConfig]
    metrics: MetricsConfig
    plots: PlotsConfig
    output: OutputConfig

    def canonical_json(self) -> str:
        return self.model_dump_json(exclude_none=False)

    def config_hash(self) -> str:
        payload = json.loads(self.canonical_json())
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]
