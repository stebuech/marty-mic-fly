"""Stage 2 — DOA hemisphere variant (Track A).

Replaces the Cartesian volumetric search grid with a focal-radius hemisphere
sampled in (azimuth, elevation). The CLEAN-SC algorithm is unchanged; only
phases 2 (search space) and 4 (mask construction) are overridden.

Aligned with the proposal text on AP 2:
    "Durch Anwendung eines Arrayverfahrens werden diese nun durch räumliche
    Filterung nach der Einfallsrichtung (direction of arrival, DOA) getrennt."

Activated by setting ``cfg.doa_grid`` on ArrayFilterStageConfig; the factory
in array_filter.py dispatches to this subclass.
"""
from __future__ import annotations

import numpy as np

from martymicfly.processing.array_filter import (
    ArrayFilterStage,
    FitInput,
    MaskBundle,
)
from martymicfly.processing.beamform_grid import (
    build_doa_cone_mask,
    build_doa_grid,
    build_rotor_doa_cones_mask,
)


_CONE_MASK_MODES = ("rotor_cone", "target_cone", "drone_cone")


class DoaArrayFilterStage(ArrayFilterStage):
    """ArrayFilterStage variant that samples a DOA hemisphere instead of a
    Cartesian volume. Inherits CSM build, fit and reconstruction unchanged
    from the parent.
    """

    def _build_fit_input(self, ctx) -> FitInput:
        if self.cfg.doa_grid is None:
            raise ValueError(
                "DoaArrayFilterStage requires cfg.doa_grid to be set"
            )
        if self.cfg.mask_mode not in _CONE_MASK_MODES:
            raise ValueError(
                f"DoaArrayFilterStage requires mask_mode in {_CONE_MASK_MODES!r}; "
                f"got {self.cfg.mask_mode!r}"
            )
        gcfg = self.cfg.doa_grid
        positions, (n_az, n_el) = build_doa_grid(
            focal_radius_m=gcfg.focal_radius_m,
            azimuth_step_deg=gcfg.azimuth_step_deg,
            elevation_step_deg=gcfg.elevation_step_deg,
            hemisphere=gcfg.hemisphere,
        )
        # reshape_hint padded with a singleton 3rd dim so integrate_band_maps
        # (which unpacks (nx, ny, nz)) keeps working — band-integrated maps
        # land as (n_az, n_el, 1).
        return FitInput(
            positions=positions,
            reshape_hint=(n_az, n_el, 1),
            aux={"grid_kind": "doa", "n_az": n_az, "n_el": n_el,
                 "focal_radius_m": gcfg.focal_radius_m,
                 "hemisphere": gcfg.hemisphere},
        )

    def _build_masks(self, fit_input: FitInput, ctx) -> MaskBundle:
        gcfg = self.cfg.doa_grid
        plat = ctx.metadata["platform"]
        rotor_positions = np.asarray(plat["rotor_positions"])

        rotor_cone_mask = build_rotor_doa_cones_mask(
            fit_input.positions, rotor_positions, gcfg.rotor_cone_half_angle_deg,
        )

        # The target "direction" is the unit vector to target_point_m. If
        # target_point_m happens to be the origin (degenerate config), fall
        # back to straight down (matches the AP2-A measurement geometry).
        target_dir = np.asarray(self.cfg.target_point_m, dtype=np.float64)
        if float(np.linalg.norm(target_dir)) < 1e-9:
            target_dir = np.array([0.0, 0.0, -1.0])
        target_cone_mask = build_doa_cone_mask(
            fit_input.positions, target_dir, gcfg.target_cone_half_angle_deg,
        )

        # drone_cone is a wider union of rotor cones — analogous to drone_box
        # being a wider catch than rotor_disc in the Cartesian variant.
        drone_cone_mask = build_rotor_doa_cones_mask(
            fit_input.positions, rotor_positions, gcfg.drone_cone_half_angle_deg,
        )

        if self.cfg.mask_mode == "rotor_cone":
            active = rotor_cone_mask
        elif self.cfg.mask_mode == "drone_cone":
            active = drone_cone_mask
        else:  # "target_cone"
            active = ~target_cone_mask

        return MaskBundle(
            active=active,
            named={
                "rotor_cone_mask": rotor_cone_mask,
                "target_cone_mask": target_cone_mask,
                "drone_cone_mask": drone_cone_mask,
                # Aliases so the Cartesian-oriented downstream metadata
                # consumers (run_pipeline plotting, compute_array_metrics)
                # see something sensible. Semantically the analog mappings:
                "rotor_disc_mask": rotor_cone_mask,
                "target_box_mask": target_cone_mask,
                "drone_box_mask": drone_cone_mask,
            },
        )
