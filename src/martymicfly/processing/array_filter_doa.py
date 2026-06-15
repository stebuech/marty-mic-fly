"""Stage 2 — DOA hemisphere variant (Track A).

Replaces the Cartesian volumetric search grid with a focal-radius hemisphere
sampled in (azimuth, elevation). The CLEAN-SC algorithm is unchanged; only
phases 2 (search space) and 4 (mask construction) are overridden.

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
    build_doa_disk_mask,
    inter_rotor_midpoints_3xR,
    build_doa_grid,
    build_rotor_doa_cones_mask,
)


_CONE_MASK_MODES = ("rotor_cone", "target_cone", "drone_disk")


class DoaArrayFilterStage(ArrayFilterStage):
    """ArrayFilterStage variant that samples a DOA hemisphere. Inherits CSM build, fit and reconstruction unchanged
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
        # rotor_cone centers on inter-rotor midpoints — the broadband downwash
        # radiation has its spatial centroid between the rotors, not on them.
        cone_axes = inter_rotor_midpoints_3xR(rotor_positions)

        rotor_cone_mask = build_rotor_doa_cones_mask(
            fit_input.positions, cone_axes, gcfg.rotor_cone_half_angle_deg,
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

        # drone_disk is the equatorial belt around the rotor plane (z=0).
        # Built as the complement of two cones along ±z with large opening
        # angles, so all near-axial DOAs are excluded and only DOAs roughly
        # in the rotor plane survive.
        drone_disk_mask = build_doa_disk_mask(
            fit_input.positions, gcfg.drone_disk_half_width_deg,
        )

        if self.cfg.mask_mode == "rotor_cone":
            active = rotor_cone_mask
        elif self.cfg.mask_mode == "drone_disk":
            active = drone_disk_mask
        else:  # "target_cone"
            active = ~target_cone_mask

        return MaskBundle(
            active=active,
            named={
                "rotor_cone_mask": rotor_cone_mask,
                "target_cone_mask": target_cone_mask,
                "drone_disk_mask": drone_disk_mask,
                # Aliases so the Cartesian-oriented downstream metadata
                # consumers (run_pipeline plotting, compute_array_metrics)
                # see something sensible. Semantically the analog mappings:
                "rotor_disc_mask": rotor_cone_mask,
                "target_box_mask": target_cone_mask,
                "drone_box_mask": drone_disk_mask,
            },
        )
