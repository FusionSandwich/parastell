import argparse
from dataclasses import replace
from pathlib import Path
from abc import ABC

import numpy as np
from scipy.interpolate import (
    RegularGridInterpolator,
    CloughTocher2DInterpolator,
)


import cadquery as cq
import pydagmc
from pymoab import core, types
import gmsh

from . import log
from .cubit_utils import (
    create_new_cubit_instance,
    import_step_cubit,
    export_mesh_cubit,
    orient_spline_surfaces,
    merge_surfaces,
    mesh_volume_auto_factor,
    mesh_surface_coarse_trimesh,
)
from .utils import (
    ToroidalMesh,
    normalize,
    expand_list,
    read_yaml_config,
    create_vol_mesh_from_surf_mesh,
    m2cm,
)
from .ports import PortGeometryResult, parse_ports
from .port_aperture import (
    ApertureBoundary,
    build_aperture_model,
    line_triangle_intersections,
)
from .native_port_geometry import build_native_port_surface_complex
from .pystell import read_vmec


def create_moab_tris_from_verts(corners, mbc, reverse=False):
    """Create 2 moab triangle elements from a list of 4 pymoab verts.

    Arguments:
        corners (4x3 numpy array): list of 4 (x,y,z) points. Connecting the
            points in the order given should result in a polygon
        mbc (pymoab core): pymoab core instance to create elements with.

    Returns:
        list of two pymoab MBTRI elements
    """
    if reverse:
        tri_1 = mbc.create_element(
            types.MBTRI, [corners[0], corners[1], corners[2]]
        )
        tri_2 = mbc.create_element(
            types.MBTRI, [corners[0], corners[2], corners[3]]
        )
    else:
        tri_1 = mbc.create_element(
            types.MBTRI, [corners[2], corners[1], corners[0]]
        )
        tri_2 = mbc.create_element(
            types.MBTRI, [corners[3], corners[2], corners[0]]
        )

    return [tri_1, tri_2]


class ReferenceSurface(ABC):
    """An object representing the innermost surface from which subsequent
    layers are built.
    """

    def __init__(self):
        self.poloidal_perturbation = 1e-4

    def angles_to_xyz(self, toroidal_angle, poloidal_angles, s, scale):
        """Method to go from a location defined by two angles and some
        constant to x, y, z coordinates.

        Arguments:
            toroidal_angles (float): toroidal angle at which to evaluate
                Cartesian coordinates [rad].
            poloidal_angles (iterable of float): poloidal angles at which to
                evaluate Cartesian coordinates [rad].
            s (float): Generic parameter which may affect the evaluation of
                the cartesian coordinate at a given angle pair.
            scale (float): a scaling factor between input and output data.

        Returns:
            coords (Nx3 numpy.array): array of Cartesian coordinates at each
                angle pair specified.
        """
        pass

    def calculate_tangents(self, toroidal_angle, poloidal_angles, s, scale):
        """Compute the tangents of a set of points, defined by a set of
        poloidal angles, at a given toroidal angle.

        Arguments:
            toroidal_angles (float): toroidal angle at which to evaluate
                tangents [rad].
            poloidal_angles (iterable of float): poloidal angles at which to
                evaluate tangents [rad].
            s (float): Generic parameter which may affect the evaluation of
                the cartesian coordinate at a given angle pair.
            scale (float): a scaling factor between input and output data.

        Returns:
            (Nx3 numpy.array): array of poloidal tangents at each angle pair
                specified.
        """
        backward_pt_loci = self.angles_to_xyz(
            toroidal_angle,
            poloidal_angles - self.poloidal_perturbation,
            s,
            scale,
        )
        forward_pt_loci = self.angles_to_xyz(
            toroidal_angle,
            poloidal_angles + self.poloidal_perturbation,
            s,
            scale,
        )

        return normalize(forward_pt_loci - backward_pt_loci)


class VMECSurface(ReferenceSurface):
    """An object that uses VMEC data to represent the innermost surface
    of an in vessel build.

    Arguments:
        vmec_obj (object): plasma equilibrium VMEC object as defined by the
            PyStell-UW VMEC reader. Must have a method
            'vmec2xyz(s, theta, phi)' that returns an (x,y,z) coordinate for
            any closed flux surface label, s, poloidal angle, theta, and
            toroidal angle, phi.
    """

    def __init__(self, vmec_obj):
        super().__init__()

        self.vmec_obj = vmec_obj

    def angles_to_xyz(self, toroidal_angle, poloidal_angles, s, scale):
        """Evaluate the Cartesian coordinates for a set of toroidal and
        poloidal angles and flux surface label.

        Arguments:
            toroidal_angles (float): toroidal angle at which to evaluate
                Cartesian coordinates [rad].
            poloidal_angles (iterable of float): poloidal angles at which to
                evaluate Cartesian coordinates [rad].
            s (float): the normalized closed flux surface label defining the
                point of reference for offset.
            scale (float): a scaling factor between input and output data.

        Returns:
            coords (Nx3 numpy.array): array of Cartesian coordinates at each
                poloidal angle specified.
        """
        coords = []

        for poloidal_angle in poloidal_angles:
            x, y, z = self.vmec_obj.vmec2xyz(s, poloidal_angle, toroidal_angle)
            coords.append([x, y, z])

        return np.array(coords) * scale


class RibBasedSurface(ReferenceSurface):
    """An object that uses closed loops of R, Z points (ribs) on planes of
    constant toroidal angle to approximate the first wall surface of an in-
    vessel build. This class must be used with split_chamber = False.

    Arguments:
        rib_data (numpy array): NxMx2 array of of R, Z points. The first
            dimension corresponds to the plane of constant toroidal angle on
            which the closed loop of points lies. The second dimension is the
            location on the closed loop at which the point lies, and the third
            dimension is the R, Z values of that point. ParaStell expects the
            following from this data set:
            - The data spans exactly one field period
            - The coordinates of each toroidal slice (rib) precess counter-
              clockwise (facing in the positive toroidal direction)
            - The coordinates obey helical (stellarator) symmetry, i.e.,
                - The (R,Z) coordinates of the first and final ribs are exactly
                  equal
                - The (R,Z) coordinates of the first, toroidal midplane, and
                  final ribs are symmetric about the axial midplane
                - The (R,Z) coordinates each half-period are a helical
                  reflection of the other half-period
        toroidal_angles (iterable of float): List of toroidal angles
            corresponding to the first dimension of rib_data. Measured in
            degrees.
        poloidal_angles (iterable of float): List of poloidal angles
            corresponding to the second dimension of rib_data. Measured in
            degrees. Should start at 0 degrees and end at 360 degrees.

    Optional attributes:
        poloidal_perturbation (float): perturbation to apply to poloidal angles
            for computing profile tangents via central difference (defaults to
            1e-4).
    """

    def __init__(self, rib_data, toroidal_angles, poloidal_angles, **kwargs):
        super().__init__()

        self.rib_data = rib_data
        self.toroidal_angles = toroidal_angles
        self.poloidal_angles = poloidal_angles

        for name in kwargs.keys() & ("poloidal_perturbation", "stuff"):
            self.__setattr__(name, kwargs[name])

        self.build_analytic_surface()

    def _extract_rib_data(self, ribs, toroidal_angles, poloidal_angles):
        """Internal function, not intended for use externally. Updates
        member variables that track R, Z values corresponding to
        angle pairs for use when building the interpolators.

        Arguments:
            ribs (np array): NxMx2 array of of R, Z points. The first
                dimension corresponds to the plane of constant toroidal angle
                on which the closed loop of points lies. The second dimension
                is the location on the closed loop at which the point lies, and
                the third dimension is the R, Z values of that point.
            toroidal_angles (iterable of float): List of toroidal angles
                corresponding to the first dimension of rib_data. Measured in
                degrees.
            poloidal_angles (iterable of float): List of poloidal angles
                corresponding to the second dimension of rib_data. Measured in
                degrees.
        """
        for phi, rib in zip(toroidal_angles, ribs):
            for theta, rib_locus in zip(poloidal_angles, rib):
                self.r_data.append(rib_locus[0])
                self.z_data.append(rib_locus[1])
                self.grid_points.append([phi, theta])

    def build_analytic_surface(self):
        """Build interpolators for R, Z coordinates using provided
        rib_data, toroidal_angles, and poloidal_angles. Adds copies of the data
        shifted by one period ahead of and behind provided data in the toroidal
        and poloidal directions to preserve periodicity.
        """
        self.r_data = []
        self.z_data = []
        self.grid_points = []

        # Toroidal Periodicity Before Period
        toroidal_shift = -max(self.toroidal_angles)
        shifted_toroidal_angles = self.toroidal_angles[:-1] + toroidal_shift
        rib_subset = self.rib_data[:-1]
        self._extract_rib_data(
            rib_subset, shifted_toroidal_angles, self.poloidal_angles
        )

        # Poloidal Periodicity Before Period
        poloidal_shift = -360.0
        shifted_poloidal_angles = self.poloidal_angles[:-1] + poloidal_shift
        rib_subset = self.rib_data[:, :-1]
        self._extract_rib_data(
            rib_subset, self.toroidal_angles, shifted_poloidal_angles
        )

        # Provided data
        self._extract_rib_data(
            self.rib_data,
            self.toroidal_angles,
            self.poloidal_angles,
        )

        # Toroidal Periodicity After Period
        toroidal_shift = max(self.toroidal_angles)
        shifted_toroidal_angles = self.toroidal_angles[1:] + toroidal_shift
        rib_subset = self.rib_data[1:]
        self._extract_rib_data(
            rib_subset, shifted_toroidal_angles, self.poloidal_angles
        )

        # Poloidal Periodicity After Period
        poloidal_shift = 360.0
        shifted_poloidal_angles = self.poloidal_angles[1:] + poloidal_shift
        rib_subset = self.rib_data[:, 1:]
        self._extract_rib_data(
            rib_subset, self.toroidal_angles, shifted_poloidal_angles
        )

        self.r_interp = CloughTocher2DInterpolator(
            self.grid_points, self.r_data
        )
        self.z_interp = CloughTocher2DInterpolator(
            self.grid_points, self.z_data
        )

    def angles_to_xyz(self, toroidal_angle, poloidal_angles, s, scale):
        """Return the cartesian coordinates from the interpolators for a
        toroidal angle and a set of poloidal angles. Takes s as a argument for
        compatibility, but does nothing with it.

        Arguments:
            toroidal_angles (float): toroidal angle at which to evaluate
                Cartesian coordinates [rad].
            poloidal_angles (iterable of float): poloidal angles at which to
                evaluate Cartesian coordinates [rad].
            s (float): Not used.
            scale (float): a scaling factor between input and output data.

        Returns:
            coords (Nx3 numpy.array): array of Cartesian coordinates at each
                angle pair specified.
        """
        coords = []
        toroidal_angle = np.rad2deg(toroidal_angle)
        poloidal_angles = np.rad2deg(poloidal_angles)

        for poloidal_angle in poloidal_angles:
            r = self.r_interp(toroidal_angle, poloidal_angle)
            z = self.z_interp(toroidal_angle, poloidal_angle)
            x = r * np.cos(np.deg2rad(toroidal_angle))
            y = r * np.sin(np.deg2rad(toroidal_angle))
            coord = np.array([x, y, z])
            coords.append(coord)

        return np.array(coords) * scale


class InVesselBuild(object):
    """Parametrically models fusion stellarator in-vessel components using
    plasma equilibrium VMEC data and a user-defined radial build.

    Arguments:
        ref_surf (object): ReferenceSurface object. Must have a method
            'angles_to_xyz(toroidal_angles, poloidal_angles, s, scale)' that
            returns an Nx3 numpy array of cartesian coordinates for any closed
            flux surface label, s, poloidal angle (theta), and toroidal angle
            (phi).
        radial_build (object): RadialBuild class object with all attributes
            defined.
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.

    Optional attributes:
        repeat (int): number of times to repeat build segment for full model
            (defaults to 0).
        num_ribs (int): total number of ribs over which to loft for each build
            segment (defaults to 61). Ribs are set at toroidal angles
            interpolated between those specified in 'toroidal_angles' if this
            value is greater than the number of entries in 'toroidal_angles'.
        num_rib_pts (int): total number of points defining each rib spline
            (defaults to 67). Points are set at poloidal angles interpolated
            between those specified in 'poloidal_angles' if this value is
            greater than the number of entries in 'poloidal_angles'.
        scale (float): a scaling factor between input and output data
            (defaults to m2cm = 100).
        use_pydagmc (bool): If True, generate components with pydagmc, rather
            than CadQuery (defaults to False).
    """

    def __init__(self, ref_surf, radial_build, logger=None, **kwargs):
        self.logger = logger
        self.ref_surf = ref_surf
        self.radial_build = radial_build
        self.ports = parse_ports(
            kwargs.get("ports", None),
            self.radial_build.user_layer_names,
        )
        self._ported_layer_names = set()
        self.port_void_components = {}
        self.port_liner_components = {}
        self.port_outer_envelopes = {}
        self.port_aperture_models = {}
        self.port_specs = {port.name: port for port in self.ports}
        self.port_geometry_diagnostics = {}
        # Private aliases are retained for downstream Prompt-2 callers.
        self._port_fill_components = self.port_void_components
        self._port_fill_specs = self.port_specs
        self._endpoint_reference_solids = {}
        self._anchor_reference_surfaces = {}

        self.repeat = 0
        self.num_ribs = 61
        self.num_rib_pts = 61
        self.scale = m2cm
        self.use_pydagmc = False

        if "scale" not in kwargs.keys():
            w = Warning(
                "No factor specified to scale InVesselBuild input data. "
                "Assuming a scaling factor of 100.0, which is consistent with "
                "input being in units of [m] and desired output in units of "
                "[cm]."
            )
            self._logger.warning(w.args[0])

        for name in kwargs.keys() & (
            "repeat",
            "num_ribs",
            "num_rib_pts",
            "scale",
            "use_pydagmc",
        ):
            self.__setattr__(name, kwargs[name])

        self.Surfaces = {}
        self.Components = {}

    @property
    def ref_surf(self):
        return self._ref_surf

    @ref_surf.setter
    def ref_surf(self, ref_surf):
        self._ref_surf = ref_surf

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger_object):
        self._logger = log.check_init(logger_object)

    @property
    def repeat(self):
        return self._repeat

    @repeat.setter
    def repeat(self, num):
        self._repeat = num
        if (self._repeat + 1) * self.radial_build.toroidal_angles[-1] > 360.0:
            e = AssertionError(
                "Total toroidal extent requested with repeated geometry "
                'exceeds 360 degrees. Please examine the "repeat" parameter '
                'and the "toroidal_angles" parameter of "radial_build".'
            )
            self._logger.error(e.args[0])
            raise e

    @property
    def use_pydagmc(self):
        return self._use_pydagmc

    @use_pydagmc.setter
    def use_pydagmc(self, value):
        self._use_pydagmc = value
        if self._use_pydagmc:
            self.mbc = core.Core()
            self.dag_model = pydagmc.Model(self.mbc)

    def _interpolate_offset_matrix(self, offset_mat):
        """Interpolates total offset for expanded angle lists using cubic spline
        interpolation.
        (Internal function not intended to be called externally)

        Returns:
            interpolated_offset_mat (np.ndarray(double)): expanded matrix
                including interpolated offset values at additional rows and
                columns [cm].
        """
        interpolator = RegularGridInterpolator(
            (
                self.radial_build.toroidal_angles,
                self.radial_build.poloidal_angles,
            ),
            offset_mat,
            method="linear" if self.use_pydagmc else "pchip",
        )

        interpolated_offset_mat = np.array(
            [
                [
                    interpolator([np.rad2deg(phi), np.rad2deg(theta)])[0]
                    for theta in self._poloidal_angles_exp
                ]
                for phi in self._toroidal_angles_exp
            ]
        )

        return interpolated_offset_mat

    def populate_surfaces(self):
        """Populates Surface class objects representing the outer surface of
        each component specified in the radial build.
        """
        self._logger.info(
            "Populating surface objects for in-vessel components..."
        )

        self._toroidal_angles_exp = np.deg2rad(
            expand_list(self.radial_build.toroidal_angles, self.num_ribs)
        )
        self._poloidal_angles_exp = np.deg2rad(
            expand_list(self.radial_build.poloidal_angles, self.num_rib_pts)
        )

        offset_mat = np.zeros(
            (
                len(self.radial_build.toroidal_angles),
                len(self.radial_build.poloidal_angles),
            )
        )

        for name, layer_data in self.radial_build.radial_build.items():
            if name == "plasma":
                s = 1.0
            else:
                s = self.radial_build.wall_s

            offset_mat += np.array(layer_data["thickness_matrix"])
            interpolated_offset_mat = self._interpolate_offset_matrix(
                offset_mat
            )

            self.Surfaces[name] = Surface(
                self._ref_surf,
                s,
                self._poloidal_angles_exp,
                self._toroidal_angles_exp,
                interpolated_offset_mat,
                self.scale,
            )

        [surface.populate_ribs() for surface in self.Surfaces.values()]

    def calculate_loci(self):
        """Calls calculate_loci method in Surface class for each component
        specified in the radial build.
        """
        self._logger.info("Computing point cloud for in-vessel components...")

        [surface.calculate_loci() for surface in self.Surfaces.values()]
        self._resolve_surface_port_placements()

    def _anchor_reference_surface(self, reference, layer=None):
        """Resolve an anchor reference to a point-cloud-backed surface."""
        if reference in {"layer_inner", "layer_outer"}:
            surface_names = list(self.Surfaces)
            outer_index = surface_names.index(layer)
            selected_index = (
                outer_index if reference == "layer_outer" else outer_index - 1
            )
            if selected_index < 0:
                raise ValueError(f"Layer {layer!r} has no inner surface")
            return self.Surfaces[surface_names[selected_index]]

        if reference in self._anchor_reference_surfaces:
            return self._anchor_reference_surfaces[reference]
        s = 1.0 if reference == "plasma_surface" else self.radial_build.wall_s
        offsets = np.zeros(
            (len(self._toroidal_angles_exp), len(self._poloidal_angles_exp))
        )
        surface = Surface(
            self._ref_surf,
            s,
            self._poloidal_angles_exp,
            self._toroidal_angles_exp,
            offsets,
            self.scale,
        )
        surface.populate_ribs()
        surface.calculate_loci()
        self._anchor_reference_surfaces[reference] = surface
        return surface

    def _resolve_surface_port_placements(self):
        """Resolve angular anchors and local frames after rib loci exist."""
        resolved_ports = []
        for port in self.ports:
            placement = port.placement
            if placement.mode != "surface" or placement.is_resolved:
                resolved_ports.append(port)
                continue
            anchor_spec = placement.surface_anchor
            axis_spec = placement.surface_axis
            phi = np.deg2rad(anchor_spec.toroidal_angle)
            theta = np.deg2rad(anchor_spec.poloidal_angle)
            surface = self._anchor_reference_surface(
                anchor_spec.reference, anchor_spec.layer
            )
            phi_min, phi_max = float(surface.phi_list[0]), float(
                surface.phi_list[-1]
            )
            seam_tolerance = max(1e-9, (phi_max - phi_min) * 1e-7)
            if phi_max - phi_min < 2.0 * np.pi - 1e-8 and (
                abs(phi - phi_min) <= seam_tolerance
                or abs(phi - phi_max) <= seam_tolerance
            ):
                raise ValueError(
                    f"Port {port.name!r} surface anchor is ambiguous at a sector seam"
                )
            anchor, poloidal, toroidal, outward = surface.local_surface_frame(
                phi, theta
            )
            poloidal_tilt = np.deg2rad(axis_spec.poloidal_tilt)
            toroidal_tilt = np.deg2rad(axis_spec.toroidal_tilt)
            tilted = (
                np.cos(poloidal_tilt) * outward
                + np.sin(poloidal_tilt) * poloidal
            )
            axis = (
                np.cos(toroidal_tilt) * tilted
                + np.sin(toroidal_tilt) * toroidal
            )
            axis /= np.linalg.norm(axis)
            if np.dot(axis, outward) <= 0.0:
                raise ValueError(
                    f"Port {port.name!r} surface axis does not point outward"
                )
            local_reference = poloidal - np.dot(poloidal, axis) * axis
            local_reference /= np.linalg.norm(local_reference)
            local_normal = np.cross(axis, local_reference)
            local_normal /= np.linalg.norm(local_normal)
            roll = np.deg2rad(placement.roll)
            local_reference = (
                np.cos(roll) * local_reference + np.sin(roll) * local_normal
            )
            local_reference /= np.linalg.norm(local_reference)
            resolved_placement = placement.resolve_surface_frame(
                anchor, axis, local_reference
            )
            resolved_ports.append(replace(port, placement=resolved_placement))
        self.ports = tuple(resolved_ports)
        self.port_specs = {port.name: port for port in self.ports}

    def generate_components(self):
        if self.use_pydagmc:
            self.generate_components_pydagmc()
        else:
            self.generate_components_cadquery()

    def generate_components_cadquery(self):
        """Constructs a CAD solid for each component specified in the radial
        build by cutting the interior surface solid from the outer surface
        solid for a given component.
        """
        self._logger.info(
            "Constructing CadQuery objects for in-vessel components..."
        )

        interior_surface = None

        segment_angles = np.linspace(
            self.radial_build.toroidal_angles[-1],
            self._repeat * self.radial_build.toroidal_angles[-1],
            num=self._repeat,
        )

        for name, surface in self.Surfaces.items():
            outer_surface = surface.generate_surface()

            if interior_surface is not None:
                segment = outer_surface.cut(interior_surface)
            else:
                segment = outer_surface

            component = segment

            for angle in segment_angles:
                rot_segment = segment.rotate((0, 0, 0), (0, 0, 1), angle)
                component = component.fuse(rot_segment)

            self.Components[name] = component
            interior_surface = outer_surface

        if self.ports:
            self._apply_ports_to_components()

    def _generate_endpoint_reference_solid(self, reference):
        """Build the enclosed ``s=1`` or ``wall_s`` reference volume lazily."""
        if reference in self._endpoint_reference_solids:
            return self._endpoint_reference_solids[reference]
        if not hasattr(self, "_toroidal_angles_exp"):
            raise ValueError(
                f"{reference} endpoint geometry is unavailable before surfaces "
                "are populated."
            )
        s = 1.0 if reference == "plasma_surface" else self.radial_build.wall_s
        offsets = np.zeros(
            (len(self._toroidal_angles_exp), len(self._poloidal_angles_exp))
        )
        surface = Surface(
            self._ref_surf,
            s,
            self._poloidal_angles_exp,
            self._toroidal_angles_exp,
            offsets,
            self.scale,
        )
        surface.populate_ribs()
        surface.calculate_loci()
        segment = surface.generate_surface()
        solid = segment
        segment_angles = np.linspace(
            self.radial_build.toroidal_angles[-1],
            self._repeat * self.radial_build.toroidal_angles[-1],
            num=self._repeat,
        )
        for angle in segment_angles:
            solid = solid.fuse(segment.rotate((0, 0, 0), (0, 0, 1), angle))
        solid = self._as_solid(solid)
        self._require_valid_solid(solid, f"Internal {reference} volume")
        self._endpoint_reference_solids[reference] = solid
        return solid

    def _shape_solids(self, shape):
        if shape is None:
            return []
        try:
            if shape.ShapeType() == "Solid":
                return [cq.Shape.cast(shape.wrapped)]
        except Exception:
            pass
        if hasattr(shape, "Solids"):
            try:
                solids = shape.Solids()
                if isinstance(solids, (list, tuple)):
                    return list(solids)
                return list(solids.vals())
            except Exception:
                pass
        if hasattr(shape, "solids"):
            try:
                return list(shape.solids().vals())
            except Exception:
                pass
        if hasattr(shape, "val"):
            try:
                return [shape.val()]
            except Exception:
                pass
        return []

    def _shape_volume(self, shape):
        solids = self._shape_solids(shape)
        if solids:
            return float(sum(abs(solid.Volume()) for solid in solids))
        try:
            return float(abs(shape.Volume()))
        except Exception:
            return 0.0

    @staticmethod
    def _bool_tolerance(reference_volume: float) -> float:
        """Scale-aware volume tolerance for OpenCascade boolean checks.

        Lofted spline sectors accumulate more Boolean integration error than
        analytic primitives, so closure uses 0.1 parts per million while
        retaining a small absolute floor for model-scale solids.
        """
        return max(1e-7, 1e-7 * max(1.0, abs(reference_volume)))

    def _as_solid(self, shape):
        solids = self._shape_solids(shape)
        if len(solids) == 0:
            return None
        if len(solids) == 1:
            return solids[0]
        e = NotImplementedError(
            "Port intersections that are disconnected into multiple "
            "components are not supported."
        )
        self._logger.error(e.args[0])
        raise e

    def _fuse_shapes(self, shapes):
        fused = None
        for shape in shapes:
            if shape is None:
                continue
            fused = shape if fused is None else fused.fuse(shape)
        return fused

    def _require_valid_solid(self, solid, description):
        if solid is None or not solid.isValid():
            e = ValueError(f"{description} is not a valid OpenCascade solid.")
            self._logger.error(e.args[0])
            raise e

    def _repair_and_require_valid_solid(self, solid, description):
        if solid is not None and solid.isValid():
            return solid

        repaired = solid
        for operation in ("clean", "fix", "clean_fix"):
            try:
                if operation == "clean":
                    candidate = solid.clean()
                elif operation == "fix":
                    candidate = solid.fix()
                else:
                    candidate = solid.clean().fix()
            except Exception:
                continue
            if candidate is not None and candidate.isValid():
                repaired = candidate
                break

        self._require_valid_solid(repaired, description)
        return repaired

    def _assert_volume_closure(
        self,
        original_volume,
        remaining_volume,
        removed_volume,
        description,
    ):
        error = abs(original_volume - remaining_volume - removed_volume)
        tolerance = self._bool_tolerance(original_volume)
        if error > tolerance:
            e = ValueError(
                f"{description} violates volume closure: error {error} "
                f"exceeds tolerance {tolerance}."
            )
            self._logger.error(e.args[0])
            raise e

    def _build_port_prism(
        self, port, start, end, radial_expansion=0.0, axial_expansion=0.0
    ):
        """Build a finite prism in the port's local frame."""
        start = float(start) - float(axial_expansion)
        end = float(end) + float(axial_expansion)
        if end <= start:
            raise ValueError(f"Port {port.name!r} has a nonpositive extent.")
        axis = np.asarray(port.placement.local_axis, dtype=float)
        reference = np.asarray(port.placement.local_reference, dtype=float)
        origin = np.asarray(port.placement.anchor, dtype=float) + axis * start
        plane = cq.Plane(
            origin=tuple(origin), xDir=tuple(reference), normal=tuple(axis)
        )
        workplane = cq.Workplane(plane)
        if port.cross_section.shape == "circle":
            radius = port.cross_section.radius + radial_expansion
            prism = workplane.circle(radius).extrude(end - start)
        else:
            width = port.cross_section.width + 2.0 * radial_expansion
            height = port.cross_section.height + 2.0 * radial_expansion
            prism = workplane.rect(width, height).extrude(end - start)
        solid = self._as_solid(prism.val())
        self._require_valid_solid(solid, f"Prism for port {port.name!r}")
        return solid

    def build_port_clearance_envelope(self, port):
        """Return the configured conservative magnet-clearance envelope."""
        result = self.port_geometry_diagnostics.get(port.name)
        if result is None:
            raise ValueError(f"Port {port.name!r} has not been generated.")
        clearance = port.collision.minimum_magnet_clearance
        radial = (
            port.liner.thickness if port.liner.enabled else 0.0
        ) + clearance
        return self._build_port_prism(
            port,
            result.resolved_start,
            result.resolved_end + result.outer_extension,
            radial_expansion=radial,
            axial_expansion=clearance,
        )

    def _line_interval(self, port, solid, description):
        """Resolve one connected centerline interval inside a solid."""
        half = port.placement.max_search_length / 2.0
        anchor = np.asarray(port.placement.anchor, dtype=float)
        axis = np.asarray(port.placement.local_axis, dtype=float)
        line = cq.Edge.makeLine(
            tuple(anchor - axis * half), tuple(anchor + axis * half)
        )
        intersection = solid.intersect(line)
        edges = [edge for edge in intersection.Edges() if edge.Length() > 1e-8]
        if not edges:
            raise ValueError(
                f"Port {port.name!r} centerline does not intersect {description}."
            )
        if len(edges) != 1:
            raise NotImplementedError(
                f"Port {port.name!r} centerline intersects {description} in "
                "multiple disconnected or far-side locations."
            )
        coordinates = [
            float(np.dot(np.asarray(vertex.toTuple()) - anchor, axis))
            for vertex in edges[0].Vertices()
        ]
        if len(coordinates) < 2:
            raise ValueError(
                f"Port {port.name!r} has a degenerate centerline intersection "
                f"with {description}."
            )
        return min(coordinates), max(coordinates)

    def _resolve_port_endpoint(self, port, endpoint, source_components):
        if port.placement.mode == "surface":
            triangles, expected_point = self._endpoint_surface_data(
                port, endpoint
            )
            coordinate = self._point_cloud_surface_coordinate(
                port, triangles, expected_point, endpoint.reference
            )
            return coordinate + endpoint.axial_offset
        if endpoint.reference == "layer":
            solid = source_components[endpoint.layer]
            low, high = self._line_interval(
                port, solid, f"layer {endpoint.layer!r}"
            )
            coordinate = low + endpoint.fraction * (high - low)
        else:
            solid = self._generate_endpoint_reference_solid(endpoint.reference)
            _, coordinate = self._line_interval(
                port, solid, endpoint.reference
            )
        return coordinate + endpoint.axial_offset

    def _point_cloud_surface_coordinate(
        self, port, triangles, expected_point, description
    ):
        anchor = np.asarray(port.placement.anchor, dtype=float)
        axis = np.asarray(port.placement.local_axis, dtype=float)
        expected = float(np.dot(np.asarray(expected_point) - anchor, axis))
        candidates = line_triangle_intersections(anchor, axis, triangles)
        half = port.placement.max_search_length / 2.0
        candidates = candidates[np.abs(candidates) <= half]
        nearby = candidates[np.abs(candidates - expected) <= 10.0]
        if len(nearby) == 0:
            raise ValueError(
                f"Port {port.name!r} centerline has no point-cloud intersection "
                f"with {description}."
            )
        coordinate = float(nearby[np.argmin(np.abs(nearby - expected))])
        equally_near = nearby[np.abs(nearby - coordinate) <= 1e-5]
        if len(equally_near) > 1:
            raise ValueError(
                f"Port {port.name!r} has ambiguous point-cloud intersections "
                f"with {description}."
            )
        return coordinate

    def _layer_boundary_surfaces(self, layer_name):
        surface_names = list(self.Surfaces)
        outer_index = surface_names.index(layer_name)
        if outer_index == 0:
            raise ValueError(f"Layer {layer_name!r} has no inner boundary")
        return (
            self.Surfaces[surface_names[outer_index - 1]],
            self.Surfaces[layer_name],
        )

    def _endpoint_surface_data(self, port, endpoint):
        anchor_spec = port.placement.surface_anchor
        phi = np.deg2rad(anchor_spec.toroidal_angle)
        theta = np.deg2rad(anchor_spec.poloidal_angle)
        if endpoint.reference in {"plasma_surface", "wall_surface"}:
            surface = self._anchor_reference_surface(endpoint.reference)
            return self._port_surface_triangles(
                port, surface
            ), surface.evaluate(phi, theta)
        inner, outer = self._layer_boundary_surfaces(endpoint.layer)
        triangles = self._interpolate_surface_triangles(
            port, inner, outer, endpoint.fraction
        )
        expected_point = (1.0 - endpoint.fraction) * inner.evaluate(
            phi, theta
        ) + endpoint.fraction * outer.evaluate(phi, theta)
        return triangles, expected_point

    def _port_layer_interval(self, port, layer_name, source_component):
        if port.placement.mode != "surface":
            return self._line_interval(
                port, source_component, f"layer {layer_name!r}"
            )
        anchor_spec = port.placement.surface_anchor
        phi = np.deg2rad(anchor_spec.toroidal_angle)
        theta = np.deg2rad(anchor_spec.poloidal_angle)
        inner, outer = self._layer_boundary_surfaces(layer_name)
        low = self._point_cloud_surface_coordinate(
            port,
            self._port_surface_triangles(port, inner),
            inner.evaluate(phi, theta),
            f"inner boundary of layer {layer_name!r}",
        )
        high = self._point_cloud_surface_coordinate(
            port,
            self._port_surface_triangles(port, outer),
            outer.evaluate(phi, theta),
            f"outer boundary of layer {layer_name!r}",
        )
        return min(low, high), max(low, high)

    @staticmethod
    def _port_aperture_half_width(port):
        liner = port.liner.thickness if port.liner.enabled else 0.0
        if port.cross_section.shape == "circle":
            return port.cross_section.radius + liner
        return (
            np.hypot(port.cross_section.width, port.cross_section.height) / 2.0
            + np.sqrt(2.0) * liner
        )

    def _port_surface_triangles(self, port, surface):
        """Refine the continuous build surface locally around one aperture."""
        anchor_spec = port.placement.surface_anchor
        return surface.triangulated_local_patch(
            np.deg2rad(anchor_spec.toroidal_angle),
            np.deg2rad(anchor_spec.poloidal_angle),
            self._port_aperture_half_width(port),
            0.05,
        )

    def _interpolate_surface_triangles(
        self, port, inner_surface, outer_surface, fraction
    ):
        anchor_spec = port.placement.surface_anchor
        parameter_grid = inner_surface.local_patch_parameter_grid(
            np.deg2rad(anchor_spec.toroidal_angle),
            np.deg2rad(anchor_spec.poloidal_angle),
            self._port_aperture_half_width(port),
            0.05,
        )
        inner = inner_surface.triangulate_parameter_grid(*parameter_grid)
        outer = outer_surface.triangulate_parameter_grid(*parameter_grid)
        if inner.shape != outer.shape:
            raise ValueError(
                "Radial boundary point clouds have different topology"
            )
        return (1.0 - fraction) * inner + fraction * outer

    def _endpoint_boundary_triangles(self, port, endpoint):
        if endpoint.reference in {"plasma_surface", "wall_surface"}:
            return self._port_surface_triangles(
                port, self._anchor_reference_surface(endpoint.reference)
            )
        inner, outer = self._layer_boundary_surfaces(endpoint.layer)
        return self._interpolate_surface_triangles(
            port, inner, outer, endpoint.fraction
        )

    def _aperture_boundaries(
        self,
        port,
        source_components,
        target_layers,
        resolved_start,
        resolved_end,
    ):
        entries = [
            ApertureBoundary(
                f"start:{port.extent.start.reference}",
                self._endpoint_boundary_triangles(port, port.extent.start),
                resolved_start,
            )
        ]
        for layer_name in target_layers:
            low, high = self._port_layer_interval(
                port, layer_name, source_components[layer_name]
            )
            inner_surface, outer_surface = self._layer_boundary_surfaces(
                layer_name
            )
            if resolved_start < low < resolved_end:
                entries.append(
                    ApertureBoundary(
                        f"{layer_name}:inner",
                        self._port_surface_triangles(port, inner_surface),
                        low,
                        (layer_name,),
                    )
                )
            if resolved_start < high < resolved_end:
                entries.append(
                    ApertureBoundary(
                        f"{layer_name}:outer",
                        self._port_surface_triangles(port, outer_surface),
                        high,
                        (layer_name,),
                    )
                )
        entries.append(
            ApertureBoundary(
                f"end:{port.extent.end.reference}",
                self._endpoint_boundary_triangles(port, port.extent.end),
                resolved_end,
            )
        )
        entries.sort(key=lambda item: item.expected_w)
        unique = []
        for entry in entries:
            if (
                unique
                and abs(entry.expected_w - unique[-1].expected_w) <= 1e-6
            ):
                previous = unique[-1]
                unique[-1] = ApertureBoundary(
                    f"{previous.name}|{entry.name}",
                    previous.triangles,
                    (previous.expected_w + entry.expected_w) / 2.0,
                    tuple(dict.fromkeys((*previous.layers, *entry.layers))),
                )
            else:
                unique.append(entry)
        if len(unique) < 2:
            raise ValueError(
                f"Port {port.name!r} has fewer than two boundaries"
            )
        return tuple(unique)

    def _trim_inner_endpoint(self, port, solid):
        endpoint = port.extent.start
        if endpoint.reference not in {"plasma_surface", "wall_surface"}:
            return solid
        if endpoint.axial_offset != 0.0:
            return solid
        reference = self._generate_endpoint_reference_solid(endpoint.reference)
        trimmed = self._as_solid(solid.cut(reference))
        if trimmed is None:
            raise ValueError(
                f"Port {port.name!r} was eliminated while conformally trimming "
                f"to {endpoint.reference}."
            )
        return self._repair_and_require_valid_solid(
            trimmed, f"Conformally trimmed port {port.name!r}"
        )

    @staticmethod
    def _intersection_location(shape):
        center = shape.CenterOfBoundBox()
        return (float(center.x), float(center.y), float(center.z))

    def _apply_ports_to_components(self):
        user_layers = [
            name
            for name in self.radial_build.user_layer_names
            if name in self.Components
        ]
        source_components = {
            name: self._repair_and_require_valid_solid(
                self._as_solid(self.Components[name]),
                f"Source component {name!r}",
            )
            for name in user_layers
        }
        baseline_volumes = {
            name: self._shape_volume(solid)
            for name, solid in source_components.items()
        }
        baseline_adjacent_overlaps = {
            (inner, outer): self._shape_volume(
                source_components[inner].intersect(source_components[outer])
            )
            for inner, outer in zip(user_layers, user_layers[1:])
        }
        all_target_layers = set()

        for port in self.ports:
            void_name = f"{port.name}__void"
            liner_name = f"{port.name}__liner"
            if port.name in self.Components or void_name in self.Components:
                raise ValueError(
                    f"Port name {port.name!r} conflicts with an in-vessel component."
                )
            if port.repetition.mode == "per_period":
                raise NotImplementedError(
                    "per_period port repetition is not implemented yet."
                )

            resolved_start = self._resolve_port_endpoint(
                port, port.extent.start, source_components
            )
            resolved_end = self._resolve_port_endpoint(
                port, port.extent.end, source_components
            )
            tolerance = self._bool_tolerance(
                max(baseline_volumes.values(), default=1.0)
            )
            if resolved_start >= resolved_end - tolerance:
                raise ValueError(
                    f"Port {port.name!r} resolves start coordinate "
                    f"{resolved_start} at or beyond end coordinate {resolved_end}; "
                    "the supplied axis may need to be reversed."
                )
            final_end = resolved_end + port.extent.outer_extension
            liner_thickness = (
                port.liner.thickness if port.liner.enabled else 0.0
            )
            transverse_pad = (
                port.cross_section.radius
                if port.cross_section.shape == "circle"
                else max(port.cross_section.width, port.cross_section.height)
                / 2.0
            ) + liner_thickness
            build_start = resolved_start
            if (
                port.extent.start.reference
                in {"plasma_surface", "wall_surface"}
                and port.extent.start.axial_offset == 0.0
            ):
                build_start -= transverse_pad + 1.0
            aperture_model = None
            if port.placement.mode == "surface":
                preliminary_layers = []
                for layer_name in user_layers:
                    low, high = self._port_layer_interval(
                        port, layer_name, source_components[layer_name]
                    )
                    if (
                        high > resolved_start + tolerance
                        and low < resolved_end - tolerance
                    ):
                        preliminary_layers.append(layer_name)
                if not preliminary_layers:
                    raise ValueError(
                        f"Port {port.name!r} centerline does not cross a user layer"
                    )
                boundaries = self._aperture_boundaries(
                    port,
                    source_components,
                    preliminary_layers,
                    resolved_start,
                    resolved_end,
                )
                aperture_model = build_aperture_model(port, boundaries)
                inner_aperture = self._trim_inner_endpoint(
                    port, aperture_model.inner_solid
                )
                outer_envelope = self._trim_inner_endpoint(
                    port, aperture_model.outer_solid
                )
                liner = (
                    self._trim_inner_endpoint(port, aperture_model.liner_solid)
                    if aperture_model.liner_solid is not None
                    else None
                )
                aperture_model = replace(
                    aperture_model,
                    inner_solid=inner_aperture,
                    outer_solid=outer_envelope,
                    liner_solid=liner,
                )
                self.port_aperture_models[port.name] = aperture_model
            else:
                inner_aperture = self._build_port_prism(
                    port, build_start, final_end
                )
                outer_envelope = self._build_port_prism(
                    port,
                    build_start,
                    final_end,
                    radial_expansion=liner_thickness,
                )
                inner_aperture = self._trim_inner_endpoint(
                    port, inner_aperture
                )
                outer_envelope = self._trim_inner_endpoint(
                    port, outer_envelope
                )

                liner = None
                if port.liner.enabled:
                    liner = self._as_solid(outer_envelope.cut(inner_aperture))
                    liner = self._repair_and_require_valid_solid(
                        liner, f"Liner for port {port.name!r}"
                    )

            for existing_name, existing in self.port_outer_envelopes.items():
                overlap_shape = outer_envelope.intersect(existing)
                overlap_volume = self._shape_volume(overlap_shape)
                if overlap_volume > self._bool_tolerance(
                    outer_envelope.Volume()
                ):
                    location = self._intersection_location(overlap_shape)
                    raise ValueError(
                        f"Port {port.name!r} outer envelope overlaps port "
                        f"{existing_name!r}: volume {overlap_volume}, "
                        f"approximate location {location}."
                    )

            if port.placement.mode == "surface":
                target_layers = tuple(preliminary_layers)
            else:
                layer_intersections = []
                for index, layer_name in enumerate(user_layers):
                    intersection = outer_envelope.intersect(
                        source_components[layer_name]
                    )
                    volume = self._shape_volume(intersection)
                    if volume > self._bool_tolerance(
                        baseline_volumes[layer_name]
                    ):
                        low, _ = self._port_layer_interval(
                            port, layer_name, source_components[layer_name]
                        )
                        layer_intersections.append((low, index, layer_name))
                layer_intersections.sort()
                target_layers = tuple(item[2] for item in layer_intersections)
            if not target_layers:
                box = outer_envelope.BoundingBox()
                raise ValueError(
                    f"Port {port.name!r} finite outer envelope does not intersect "
                    "any user layer; envelope "
                    f"volume={outer_envelope.Volume()}, "
                    f"bbox=({box.xmin}, {box.ymin}, {box.zmin}) to "
                    f"({box.xmax}, {box.ymax}, {box.zmax})."
                )
            if (
                port.expected_layers is not None
                and target_layers != port.expected_layers
            ):
                raise ValueError(
                    f"Port {port.name!r} expected layers {port.expected_layers} "
                    f"but geometrically intersects {target_layers}."
                )
            if port.resolution is not None and set(target_layers) != set(
                port.resolution.layers
            ):
                raise ValueError(
                    f"Deprecated layer_span for port {port.name!r} resolves "
                    f"{port.resolution.layers} but finite geometry intersects "
                    f"{target_layers}."
                )

            staged_components = {}
            original_volume = 0.0
            remaining_volume = 0.0
            total_cut_volume = 0.0
            blanket_union = None
            for layer_name in target_layers:
                component = self._repair_and_require_valid_solid(
                    self._as_solid(self.Components[layer_name]),
                    f"Current component {layer_name!r}",
                )
                before = self._shape_volume(component)
                if aperture_model is not None:
                    layer_low, layer_high = self._port_layer_interval(
                        port, layer_name, source_components[layer_name]
                    )
                    cutters = [
                        segment
                        for segment_low, segment_high, segment in aperture_model.boolean_segments
                        if segment_high > layer_low
                        and segment_low < layer_high
                    ]
                    remaining_shape = component
                    for cutter in cutters:
                        cut_error = None
                        for fuzzy_tolerance in (1e-7, 1e-6, 1e-5, 1e-4):
                            try:
                                candidate = remaining_shape.cut(
                                    cutter, tol=fuzzy_tolerance
                                )
                            except (ValueError, RuntimeError) as error:
                                cut_error = error
                                continue
                            if (
                                candidate is not None
                                and not candidate.isNull()
                            ):
                                remaining_shape = candidate
                                break
                        else:
                            raise ValueError(
                                f"Point-cloud aperture cut failed for port "
                                f"{port.name!r}, layer {layer_name!r}"
                            ) from cut_error
                    remaining = self._as_solid(remaining_shape)
                else:
                    remaining = self._as_solid(
                        component.cut(outer_envelope, tol=1e-6)
                    )
                if remaining is None:
                    raise ValueError(
                        f"Port {port.name!r} completely removes layer {layer_name!r}."
                    )
                remaining = self._repair_and_require_valid_solid(
                    remaining,
                    f"Remaining component for port {port.name!r}, layer {layer_name!r}",
                )
                after = self._shape_volume(remaining)
                if port.placement.mode == "surface":
                    removed_volume = before - after
                else:
                    removed = self._as_solid(
                        component.intersect(outer_envelope)
                    )
                    if removed is None:
                        raise ValueError(
                            f"Port {port.name!r} does not remove positive volume "
                            f"from layer {layer_name!r}."
                        )
                    removed_volume = self._shape_volume(removed)
                if removed_volume <= self._bool_tolerance(before):
                    raise ValueError(
                        f"Port {port.name!r} does not remove positive volume "
                        f"from layer {layer_name!r}."
                    )
                self._assert_volume_closure(
                    before,
                    after,
                    removed_volume,
                    f"Port {port.name!r}, layer {layer_name!r}",
                )
                for assembly_shape, kind in (
                    (inner_aperture, "void"),
                    (liner, "liner"),
                ):
                    if assembly_shape is None:
                        continue
                    overlap = self._shape_volume(
                        remaining.intersect(assembly_shape)
                    )
                    if overlap > self._bool_tolerance(before):
                        raise ValueError(
                            f"Port {port.name!r} {kind} overlaps remaining "
                            f"material in layer {layer_name!r}: volume "
                            f"{overlap}, tolerance {self._bool_tolerance(before)}."
                        )
                staged_components[layer_name] = remaining
                original_volume += before
                remaining_volume += after
                total_cut_volume += removed_volume
                source = source_components[layer_name]
                blanket_union = (
                    source
                    if blanket_union is None
                    else blanket_union.fuse(source)
                )

            void_inside = self._shape_volume(
                inner_aperture.intersect(blanket_union)
            )
            liner_inside = (
                self._shape_volume(liner.intersect(blanket_union))
                if liner is not None
                else 0.0
            )
            void_outside = inner_aperture.Volume() - void_inside
            liner_outside = (
                (liner.Volume() - liner_inside) if liner is not None else 0.0
            )
            closure_error = abs(
                original_volume - remaining_volume - void_inside - liner_inside
            )
            if closure_error > self._bool_tolerance(original_volume):
                raise ValueError(
                    f"Port {port.name!r} assembly violates blanket volume "
                    f"closure: error {closure_error}."
                )
            partition_tolerance = self._bool_tolerance(total_cut_volume)
            if aperture_model is not None:
                partition_tolerance = max(
                    partition_tolerance, 2e-4 * total_cut_volume
                )
            if (
                abs(total_cut_volume - void_inside - liner_inside)
                > partition_tolerance
            ):
                raise ValueError(
                    f"Port {port.name!r} void and liner do not partition the "
                    f"removed blanket volume: cut={total_cut_volume}, "
                    f"void={void_inside}, liner={liner_inside}, tolerance="
                    f"{partition_tolerance}."
                )
            if liner is not None:
                void_liner_overlap = self._shape_volume(
                    inner_aperture.intersect(liner)
                )
                if void_liner_overlap > self._bool_tolerance(liner.Volume()):
                    raise ValueError(
                        f"Port {port.name!r} liner overlaps its clear void."
                    )

            plasma_overlap = 0.0
            if (
                port.extent.start.reference == "plasma_surface"
                and liner is not None
            ):
                plasma = self._generate_endpoint_reference_solid(
                    "plasma_surface"
                )
                plasma_overlap = self._shape_volume(liner.intersect(plasma))
                if plasma_overlap > self._bool_tolerance(plasma.Volume()):
                    raise ValueError(
                        f"Port {port.name!r} liner penetrates the plasma volume."
                    )

            self.Components.update(staged_components)
            self.Components[void_name] = inner_aperture
            if liner is not None:
                self.Components[liner_name] = liner
                self.port_liner_components[port.name] = liner
            self.port_void_components[port.name] = inner_aperture
            self.port_outer_envelopes[port.name] = outer_envelope
            if aperture_model is not None:
                self.port_aperture_models[port.name] = aperture_model
            self._ported_layer_names.update(target_layers)
            all_target_layers.update(target_layers)
            self.port_geometry_diagnostics[port.name] = PortGeometryResult(
                name=port.name,
                resolved_start=resolved_start,
                resolved_end=resolved_end,
                outer_extension=port.extent.outer_extension,
                ordered_intersected_layers=target_layers,
                original_blanket_volume=original_volume,
                remaining_blanket_volume=remaining_volume,
                void_volume_inside_blanket=void_inside,
                liner_volume_inside_blanket=liner_inside,
                void_volume_outside_blanket=max(0.0, void_outside),
                liner_volume_outside_blanket=max(0.0, liner_outside),
                total_cut_volume=total_cut_volume,
                closure_error=closure_error,
                maximum_liner_overlap_with_plasma=plasma_overlap,
            )

        for layer_name in user_layers:
            final_volume = self._shape_volume(self.Components[layer_name])
            original_volume = baseline_volumes[layer_name]
            tolerance = self._bool_tolerance(original_volume)
            if layer_name in all_target_layers:
                if final_volume >= original_volume - tolerance:
                    raise ValueError(
                        f"Ported layer {layer_name!r} did not lose positive volume."
                    )
            elif abs(final_volume - original_volume) > tolerance:
                raise ValueError(
                    f"Unselected layer {layer_name!r} changed volume by "
                    f"{abs(final_volume - original_volume)}."
                )

        for pair, baseline_overlap in baseline_adjacent_overlaps.items():
            inner, outer = pair
            final_overlap = self._shape_volume(
                self.Components[inner].intersect(self.Components[outer])
            )
            tolerance = self._bool_tolerance(
                max(baseline_volumes[inner], baseline_volumes[outer])
            )
            if final_overlap > baseline_overlap + tolerance:
                raise ValueError(
                    f"Port booleans introduced overlap between adjacent layers "
                    f"{inner!r} and {outer!r}."
                )

        for name in all_target_layers:
            self._require_valid_solid(
                self._as_solid(self.Components[name]),
                f"Final component {name!r}",
            )
        for name, solid in self.port_void_components.items():
            self._require_valid_solid(solid, f"Final port void {name!r}")
        for name, solid in self.port_liner_components.items():
            self._require_valid_solid(solid, f"Final port liner {name!r}")

    def _connect_ribs_with_tris_moab(self, rib1, rib2, reverse=False):
        """Creat MBTRI elements add add them to a surface between two ribs.
        (Internal function not intended to be called externally)

        Arguments:
            rib1 (Rib object): First of two ribs to be connected.
            rib2 (Rib object): Second of two ribs to be connected.
            reverse (bool): Optional. Whether to reverse the connectivity of
                the MBTRIs being generated (defaults to False).

        Returns:
            mb_tris (list of Entity Handle): List of the entity handles of the
                MBTRIs connecting the two ribs.
        """
        mb_tris = []
        for rib_loci_index, _ in enumerate(rib1.rib_loci[0:-1]):
            corner1 = rib1.mb_verts[rib_loci_index]
            corner2 = rib1.mb_verts[rib_loci_index + 1]
            corner3 = rib2.mb_verts[rib_loci_index + 1]
            corner4 = rib2.mb_verts[rib_loci_index]
            corners = [corner1, corner2, corner3, corner4]
            mb_tris += create_moab_tris_from_verts(
                corners, self.mbc, reverse=reverse
            )
        return mb_tris

    def _generate_pymoab_verts(self):
        """Generate MBVERTEX entities from rib loci in all surfaces
        (Internal function not intended to be called externally)
        """
        [
            surface._generate_pymoab_verts(self.mbc)
            for surface in self.Surfaces.values()
        ]

    def _generate_curved_surfaces_pydagmc(self, continuous_360=False):
        """Generate the faceted representation of each curved surface and
        add it to the PyDAGMC model, remembering the surface ids. The sense
        of the triangles should point outward (increasing radial direction),
        with the exception of the first surface, which should point inward
        since the implicit complement is being used for the plasma chamber.
        (Internal function not intended to be called externally)

        Arguments:
            continuous_360 (bool): flag indicating whether 360-degree,
                continuous geometries should be generated.
        """
        self.curved_surface_ids = []
        surfaces = list(self.Surfaces.values())
        first_surface = surfaces[0]
        for surface in surfaces:
            mb_tris = []

            if continuous_360:
                ribs = surface.Ribs[:-1] + [surface.Ribs[0]]
            else:
                ribs = surface.Ribs

            for rib, next_rib in zip(ribs[0:-1], ribs[1:]):
                mb_tris += self._connect_ribs_with_tris_moab(
                    rib,
                    next_rib,
                    reverse=(surface == first_surface),
                )
            dagmc_surface = self.dag_model.create_surface()
            self.dag_model.mb.add_entities(dagmc_surface.handle, mb_tris)
            self.curved_surface_ids.append(dagmc_surface.id)

    def _generate_end_cap_surfaces_pydagmc(self):
        """Generate the faceted representation of the planar end cap surfaces
        and add them to the PyDAGMC model, remembering the surface ids.
        The sense of the triangles should point toward the implicit complement.
        (Internal function not intended to be called externally)
        """
        self.end_cap_surface_ids = []
        for surface, next_surface in zip(
            list(self.Surfaces.values())[0:-1],
            list(self.Surfaces.values())[1:],
        ):
            end_cap_pair = []
            for index in (0, -1):
                mb_tris = self._connect_ribs_with_tris_moab(
                    surface.Ribs[index],
                    next_surface.Ribs[index],
                    reverse=(index == -1),
                )
                end_cap = self.dag_model.create_surface()
                self.mbc.add_entities(end_cap.handle, mb_tris)
                end_cap_pair.append(end_cap.id)

            self.end_cap_surface_ids.append(end_cap_pair)

    def _generate_volumes_pydagmc(self, continuous_360=False):
        """Use the curved surface and end cap surface IDs to build the
        the volumes by applying the correct surface sense to each surface.
        The convention here is to point the surface sense toward the implicit
        complement, or if the surface is between two volumes then the surface
        sense should point in the increasing radial direction.
        (Internal function not intended to be called externally)

        Arguments:
            continuous_360 (bool): flag indicating whether 360-degree,
                continuous geometries should be generated.
        """

        [self.dag_model.create_volume() for _ in list(self.Surfaces)[:-1]]

        # First surface goes to the implicit complement (plasma chamber)
        first_surface = self.dag_model.surfaces_by_id[
            self.curved_surface_ids[0]
        ]
        first_surface.senses = [
            self.dag_model.volumes_by_id[first_surface.id],
            None,
        ]

        for surface_id in self.curved_surface_ids[1:-1]:
            self.dag_model.surfaces_by_id[surface_id].senses = [
                self.dag_model.volumes_by_id[surface_id - 1],
                self.dag_model.volumes_by_id[surface_id],
            ]

        # if it the last surface it goes to the implicit complement
        last_surface = self.dag_model.surfaces_by_id[
            self.curved_surface_ids[-1]
        ]
        last_surface.senses = [
            self.dag_model.volumes_by_id[last_surface.id - 1],
            None,
        ]

        # all end caps go to the implicit complement.
        if not continuous_360:
            for vol_id, end_cap_ids in enumerate(
                self.end_cap_surface_ids, start=1
            ):
                for end_cap_id in end_cap_ids:
                    self.dag_model.surfaces_by_id[end_cap_id].senses = [
                        self.dag_model.volumes_by_id[vol_id],
                        None,
                    ]

    def _tag_volumes_with_materials_pydagmc(self):
        """Tag each volume with the appropriate material name
        (Internal function not intended to be called externally)
        """
        for vol, (layer_name, layer_data) in zip(
            self.dag_model.volumes,
            list(self.radial_build.radial_build.items())[1:],
        ):
            mat = layer_data.get("mat_tag", layer_name)
            group = pydagmc.Group.create(self.dag_model, name="mat:" + mat)
            group.add_set(vol)
            layer_data["vol_id"] = vol.id

    def generate_components_pydagmc(
        self,
        *,
        include_graveyard=True,
        aperture_chord_tolerance=0.05,
        vertex_merge_tolerance=1.0e-9,
    ):
        """Use PyDAGMC to build a DAGMC model of the invessel components.

        Native port models can omit their graveyard while independent physical
        models are assembled.  The default preserves the standalone behavior.
        """
        self._logger.info(
            "Generating DAGMC model of in-vessel components with PyDAGMC..."
        )

        if self.ports:
            self.native_port_complex = build_native_port_surface_complex(
                self,
                include_graveyard=include_graveyard,
                aperture_chord_tolerance=aperture_chord_tolerance,
                vertex_merge_tolerance=vertex_merge_tolerance,
            )
            self.dag_model = self.native_port_complex.to_pydagmc()
            self.mbc = self.dag_model.mb
            return

        if np.isclose(
            self.radial_build.toroidal_angles[-1]
            - self.radial_build.toroidal_angles[0],
            360.0,
        ):
            continuous_360 = True
        else:
            continuous_360 = False

        self._generate_pymoab_verts()
        self._generate_curved_surfaces_pydagmc(continuous_360=continuous_360)
        if not continuous_360:
            self._generate_end_cap_surfaces_pydagmc()
        self._generate_volumes_pydagmc(continuous_360=continuous_360)
        self._tag_volumes_with_materials_pydagmc()

    def get_loci(self):
        """Returns the set of point-loci defining the outer surfaces of the
        components specified in the radial build.
        """
        return np.array(
            [surface.get_loci() for surface in self.Surfaces.values()]
        )

    def merge_surfaces(self):
        """Merges ParaStell in-vessel component surfaces in Coreform Cubit
        based on surface IDs rather than imprinting and merging all. Assumes
        that the radial_build dictionary is ordered radially outward. Note that
        overlaps between magnet volumes and in-vessel components will not be
        merged in this workflow.
        """
        # Tracks the surface id of the outer surface of the previous layer
        prev_outer_surface_id = None

        for data in self.radial_build.radial_build.values():
            inner_surface_id, outer_surface_id = orient_spline_surfaces(
                data["vol_id"]
            )

            # Conditionally skip merging (first iteration only)
            if prev_outer_surface_id is None:
                prev_outer_surface_id = outer_surface_id
            else:
                merge_surfaces(inner_surface_id, prev_outer_surface_id)
                prev_outer_surface_id = outer_surface_id

    def import_step_cubit(self):
        """Imports STEP files from in-vessel build into Coreform Cubit."""
        for name, data in self.radial_build.radial_build.items():
            import_path = Path(self.export_dir) / Path(name)
            vol_id = import_step_cubit(import_path)
            data["vol_id"] = vol_id

    def export_step(self, export_dir=""):
        """Export CAD solids as STEP files via CadQuery.

        Arguments:
            export_dir (str): directory to which to export the STEP output files
                (defaults to empty string).
        """
        self._logger.info("Exporting STEP files for in-vessel components...")

        self.export_dir = export_dir

        for name, component in self.Components.items():
            export_path = Path(self.export_dir) / Path(name).with_suffix(
                ".step"
            )
            cq.exporters.export(component, str(export_path))

    def export_native_port_artifacts(self, output_dir, **kwargs):
        """Export native point-cloud DAGMC and conformal volume-mesh files."""
        from .native_port_artifacts import export_native_port_artifacts

        return export_native_port_artifacts(self, output_dir, **kwargs)

    def extract_solids_and_mat_tags(self):
        """Get a list of all cadquery solids, and a corresponding list of
        the respective material tags.

        Returns:
            solids (list): list of in-vessel component CadQuery solid objects.
            mat_tags (list): list of in-vessel component material tags.
        """
        solids = []
        mat_tags = []

        for name, solid in self.Components.items():
            solids.append(solid)
            if name in self.radial_build.radial_build:
                mat_tags.append(
                    self.radial_build.radial_build[name]["mat_tag"]
                )
            elif name.endswith("__void"):
                port_spec = self.port_specs[name[: -len("__void")]]
                mat_tags.append(port_spec.fill.mat_tag)
            elif name.endswith("__liner"):
                port_spec = self.port_specs[name[: -len("__liner")]]
                mat_tags.append(port_spec.liner.mat_tag)
            else:
                raise ValueError(f"No material tag is defined for {name!r}.")

        return solids, mat_tags

    def mesh_components_moab(self, components):
        """Creates a tetrahedral mesh of in-vessel component volumes via MOAB.
        This mesh is created using the point cloud of the specified components
        and as such, each component's mesh will be one tetrahedron thick.

        Arguments:
            components (array of str): array containing the names of the
                in-vessel components to be meshed.
        """
        self._logger.info(
            "Generating tetrahedral mesh of in-vessel component(s) via MOAB..."
        )

        def remove_inner_component(component):
            """Upon identification of a requested component whose meshing is
            not supported by the MOAB workflow, removes that component from the
            input list and raises a warning.

            Arguments:
                component (str): component to be removed.
            """
            w = Warning(
                f"Meshing of {component} volume not supported for MOAB "
                f"workflow; {component} volume will be removed from list of "
                "components to be meshed."
            )
            self._logger.warning(w.args[0])
            components.remove(component)

        if "plasma" in components:
            remove_inner_component("plasma")
        elif "chamber" in components:
            remove_inner_component("chamber")

        if self.ports:
            affected = self._ported_layer_names.intersection(components)
            if affected:
                e = NotImplementedError(
                    "MOAB mesh workflow is not supported for ported components: "
                    f"{sorted(affected)}"
                )
                self._logger.error(e.args[0])
                raise e
            port_component_names = {
                f"{name}__void" for name in self.port_void_components
            }.union(f"{name}__liner" for name in self.port_liner_components)
            affected_ports = port_component_names.intersection(components)
            if affected_ports:
                e = NotImplementedError(
                    "MOAB mesh workflow is not supported for port fills: "
                    f"{sorted(affected_ports)}"
                )
                self._logger.error(e.args[0])
                raise e

        surface_keys = list(self.Surfaces.keys())

        # Check if components list is ordered correctly
        sorted_components = sorted(
            components, key=lambda component: surface_keys.index(component)
        )
        if components != sorted_components:
            w = Warning(
                "List of components to be meshed is not properly ordered. "
                "Reordering input list."
            )
            self._logger.warning(w.args[0])
            components = sorted_components

        # Initialize the list of Surface class objects to be included in the
        # mesh, to be used to define mesh vertices on those surfaces later
        surfaces = []
        # Initialize the list booleans identifying whether the regions between
        # mesh surfaces should be meshed or not
        gap_map = []

        # Identify surfaces and gaps in mesh
        for component in components:
            # Extract inner and outer surfaces of current component
            outer_surface = self.Surfaces[component]
            # Inner surface of current component is outer surface of the
            # previous component. Since surfaces are created in order of
            # components and named after the component for which they are the
            # outer surface, it can be found by the ordered list of surface
            # keys
            inner_surf_idx = surface_keys.index(component) - 1
            inner_component = surface_keys[inner_surf_idx]
            inner_surface = self.Surfaces[inner_component]

            # Handle first component
            if len(surfaces) == 0:
                surfaces.append(inner_surface)
            # If the inner component is not the previous component specified to
            # be meshed, identify a gap and add the inner surface
            elif surfaces[-1] != inner_surface:
                surfaces.append(inner_surface)
                # Don't mesh the gap between this surface and its predecessor
                gap_map.append(True)

            surfaces.append(outer_surface)
            gap_map.append(False)

        self.moab_mesh = InVesselComponentMesh(surfaces, gap_map, self._logger)
        self.moab_mesh.create_vertices()
        self.moab_mesh.create_mesh()

    def export_mesh_moab(self, filename, export_dir=""):
        """Exports a tetrahedral mesh of in-vessel component volumes in H5M
        format via MOAB.

        Arguments:
            filename (str): name of H5M output file.
            export_dir (str): directory to which to export the h5m output file
                (defaults to empty string).
        """
        self.moab_mesh.export_mesh(filename, export_dir=export_dir)

    def mesh_components_gmsh(
        self, components, min_mesh_size=5.0, max_mesh_size=20.0, algorithm=1
    ):
        """Creates a tetrahedral mesh of in-vessel component volumes via Gmsh.

        Arguments:
            components (array of str): array containing the names of the
                in-vessel components to be meshed.
            min_mesh_size (float): minimum size of mesh elements (defaults to
                5.0).
            max_mesh_size (float): maximum size of mesh elements (defaults to
                20.0).
            algorithm (int): integer identifying the meshing algorithm to use
                for the surface boundary (defaults to 1). Options are as
                follows, refer to Gmsh documentation for explanations of each.
                1: MeshAdapt, 2: automatic, 3: initial mesh only, 4: N/A,
                5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay
                for Quads, 9: Packing of Parallelograms, 11: Quasi-structured
                Quad.
        """
        self._logger.info(
            "Generating tetrahedral mesh of in-vessel component(s) via Gmsh..."
        )

        gmsh.initialize()

        gmsh.option.setNumber(
            "General.NumThreads", 0
        )  # Use all available cores

        if self._use_pydagmc:
            self._gmsh_from_pydagmc(
                components, min_mesh_size, max_mesh_size, algorithm
            )
        else:
            self._gmsh_from_cadquery(
                components, min_mesh_size, max_mesh_size, algorithm
            )

    def _gmsh_from_pydagmc(
        self, components, min_mesh_size, max_mesh_size, algorithm
    ):
        """Adds PyDAGMC geometry to Gmsh instance.
        (Internal function not intended to be called externally)

        Arguments:
            components (array of str): array containing the names of the
                in-vessel components to be meshed.
            min_mesh_size (float): minimum size of mesh elements.
            max_mesh_size (float): maximum size of mesh elements.
            algorithm (int): integer identifying the meshing algorithm to use
                for the surface boundary.
        """
        mesh_files = []

        # Extract each component from PyDAGMC model and remesh it in Gmsh
        for component in components:
            volume_id = self.radial_build.radial_build[component]["vol_id"]

            vtk_path = str(Path(f"volume_{volume_id}_tmp").with_suffix(".vtk"))
            self.dag_model.volumes_by_id[volume_id].to_vtk(vtk_path)

            mesh_files.append(
                create_vol_mesh_from_surf_mesh(
                    min_mesh_size, max_mesh_size, algorithm, vtk_path
                )
            )

        # Combine all component meshes into one
        for mesh_file in mesh_files:
            gmsh.merge(mesh_file)
            Path(mesh_file).unlink()

    def _gmsh_from_cadquery(
        self, components, min_mesh_size, max_mesh_size, algorithm
    ):
        """Adds CadQuery geometry to Gmsh instance.
        (Internal function not intended to be called externally)

        Arguments:
            components (array of str): array containing the names of the
                in-vessel components to be meshed.
            min_mesh_size (float): minimum size of mesh elements.
            max_mesh_size (float): maximum size of mesh elements.
            algorithm (int): integer identifying the meshing algorithm to use
                for the surface boundary.
        """
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_mesh_size)
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)

        for component in components:
            gmsh.model.occ.importShapesNativePointer(
                self.Components[component].wrapped._address()
            )

        gmsh.model.occ.synchronize()

        gmsh.model.mesh.generate(dim=3)

    def export_mesh_gmsh(self, filename, export_dir=""):
        """Exports a tetrahedral mesh of in-vessel component volumes in H5M
        format via Gmsh and MOAB.

        Arguments:
            filename (str): name of H5M output file.
            export_dir (str): directory to which to export the h5m output file
                (defaults to empty string).
        """
        self._logger.info("Exporting mesh H5M file...")

        vtk_path = Path(export_dir) / Path(filename).with_suffix(".vtk")
        moab_path = vtk_path.with_suffix(".h5m")

        gmsh.write(str(vtk_path))

        gmsh.clear()
        gmsh.finalize()

        self.mesh_mbc = core.Core()
        self.mesh_mbc.load_file(str(vtk_path))
        self.mesh_mbc.write_file(str(moab_path))

        Path(vtk_path).unlink()

    def mesh_components_cubit(
        self,
        components,
        mesh_size=5,
        anisotropic_ratio=100.0,
        deviation_angle=5.0,
        import_dir="",
    ):
        """Creates a tetrahedral mesh of in-vessel component volumes via
        Coreform Cubit.

        Arguments:
            components (array of str): array containing the names of the
                in-vessel components to be meshed.
            mesh_size (float): controls the size of the mesh. Takes values
                between 1.0 (finer) and 10.0 (coarser) (defaults to 5.0).
            anisotropic_ratio (float): controls edge length ratio of elements
                (defaults to 100.0).
            deviation_angle (float): controls deviation angle of facet from
                surface (i.e., lesser deviation angle results in more elements
                in areas with higher curvature) (defaults to 5.0).
            import_dir (str): directory containing the STEP file of
                the in-vessel component (defaults to empty string).
        """
        self._logger.info(
            "Generating tetrahedral mesh of in-vessel component(s) via Coreform"
            " Cubit..."
        )

        create_new_cubit_instance()

        volume_ids = []

        for component in components:
            import_path = Path(import_dir) / Path(component)
            volume_id = import_step_cubit(import_path)
            volume_ids.append(volume_id)

        mesh_surface_coarse_trimesh(
            anisotropic_ratio=anisotropic_ratio,
            deviation_angle=deviation_angle,
        )
        mesh_volume_auto_factor(volume_ids, mesh_size=mesh_size)

    def export_mesh_cubit(self, filename, export_dir=""):
        """Exports a tetrahedral mesh of in-vessel component volumes in H5M
        format via Coreform Cubit and MOAB.

        Arguments:
            filename (str): name of H5M output file.
            export_dir (str): directory to which to export the h5m output file
                (defaults to empty string).
        """
        self._logger.info("Exporting mesh H5M file...")

        export_mesh_cubit(
            filename=filename,
            export_dir=export_dir,
            delete_upon_export=True,
        )


class Surface(object):
    """An object representing a surface formed by lofting across a set of
    "ribs" located at different toroidal planes and offset from a reference
    surface.

    Arguments:
        ref_surf (object): ReferenceSurface object. Must have a method
            'angles_to_xyz(toroidal_angles, poloidal_angles, s, scale)' that
            returns an Nx3 numpy array of cartesian coordinates for any closed
            flux surface label, s, poloidal angle (theta), and toroidal angle
            (phi).
        s (float): the normalized closed flux surface label defining the point
            of reference for offset.
        theta_list (np.array(double)): the set of poloidal angles specified for
            each rib [rad].
        phi_list (np.array(double)): the set of toroidal angles defining the
            plane in which each rib is located [rad].
        offset_mat (np.array(double)): the set of offsets from the surface
            defined by s for each toroidal angle, poloidal angle pair on the
            surface [cm].
        scale (float): a scaling factor between input and output data.
    """

    def __init__(self, ref_surf, s, theta_list, phi_list, offset_mat, scale):
        self.ref_surf = ref_surf
        self.s = s
        self.theta_list = theta_list
        self.phi_list = phi_list
        self.offset_mat = offset_mat
        self.scale = scale

        self.surface = None
        self._offset_interpolator = None

    def _canonical_angles(self, toroidal_angle, poloidal_angle):
        phi = float(toroidal_angle)
        theta = float(poloidal_angle)
        phi_min, phi_max = float(self.phi_list[0]), float(self.phi_list[-1])
        if phi < phi_min - 1e-12 or phi > phi_max + 1e-12:
            raise ValueError(
                f"Toroidal angle {np.rad2deg(phi)} degrees lies outside the "
                f"surface sector [{np.rad2deg(phi_min)}, {np.rad2deg(phi_max)}]."
            )
        theta_min, theta_max = (
            float(self.theta_list[0]),
            float(self.theta_list[-1]),
        )
        period = theta_max - theta_min
        if period <= 0.0:
            raise ValueError("Surface poloidal coordinates are not increasing")
        theta = (theta - theta_min) % period + theta_min
        return min(max(phi, phi_min), phi_max), theta

    def _offset_at(self, toroidal_angle, poloidal_angle):
        if self._offset_interpolator is None:
            self._offset_interpolator = RegularGridInterpolator(
                (self.phi_list, self.theta_list),
                self.offset_mat,
                method="linear",
                bounds_error=True,
            )
        return float(
            self._offset_interpolator(
                np.array([[toroidal_angle, poloidal_angle]], dtype=float)
            )[0]
        )

    def evaluate(self, toroidal_angle, poloidal_angle):
        """Evaluate the same continuous offset surface used by the rib cloud."""
        phi, theta = self._canonical_angles(toroidal_angle, poloidal_angle)
        theta_values = np.array([theta], dtype=float)
        point = np.asarray(
            self.ref_surf.angles_to_xyz(phi, theta_values, self.s, self.scale),
            dtype=float,
        ).reshape(-1, 3)[0]
        offset = self._offset_at(phi, theta)
        if offset != 0.0:
            poloidal = np.asarray(
                self.ref_surf.calculate_tangents(
                    phi, theta_values, self.s, self.scale
                ),
                dtype=float,
            ).reshape(-1, 3)[0]
            toroidal_plane_normal = np.array(
                [-np.sin(phi), np.cos(phi), 0.0], dtype=float
            )
            outward = np.cross(toroidal_plane_normal, poloidal)
            outward /= np.linalg.norm(outward)
            point = point + offset * outward
        return point

    def local_surface_frame(self, toroidal_angle, poloidal_angle):
        """Return point, poloidal/toroidal tangents, and selected outward normal."""
        phi, theta = self._canonical_angles(toroidal_angle, poloidal_angle)
        phi_span = float(self.phi_list[-1] - self.phi_list[0])
        delta = min(1e-5, max(phi_span * 1e-4, 1e-7))
        if phi - delta < self.phi_list[0]:
            toroidal = self.evaluate(phi + delta, theta) - self.evaluate(
                phi, theta
            )
        elif phi + delta > self.phi_list[-1]:
            toroidal = self.evaluate(phi, theta) - self.evaluate(
                phi - delta, theta
            )
        else:
            toroidal = self.evaluate(phi + delta, theta) - self.evaluate(
                phi - delta, theta
            )
        poloidal = self.evaluate(phi, theta + delta) - self.evaluate(
            phi, theta - delta
        )
        toroidal /= np.linalg.norm(toroidal)
        poloidal = poloidal - np.dot(poloidal, toroidal) * toroidal
        poloidal /= np.linalg.norm(poloidal)
        outward = np.cross(toroidal, poloidal)
        outward /= np.linalg.norm(outward)

        reference_poloidal = np.asarray(
            self.ref_surf.calculate_tangents(
                phi, np.array([theta]), self.s, self.scale
            ),
            dtype=float,
        ).reshape(-1, 3)[0]
        expected_outward = np.cross(
            np.array([-np.sin(phi), np.cos(phi), 0.0]), reference_poloidal
        )
        if np.dot(outward, expected_outward) < 0.0:
            poloidal = -poloidal
            outward = -outward
        return self.evaluate(phi, theta), poloidal, toroidal, outward

    def triangulated_point_cloud(self):
        """Triangulate the refined rib loci without creating a CAD surface."""
        loci = self.get_loci()
        triangles = []
        for rib_index in range(loci.shape[0] - 1):
            for point_index in range(loci.shape[1] - 1):
                a = loci[rib_index, point_index]
                b = loci[rib_index + 1, point_index]
                c = loci[rib_index + 1, point_index + 1]
                d = loci[rib_index, point_index + 1]
                triangles.append((a, b, c))
                triangles.append((a, c, d))
        return np.asarray(triangles, dtype=float)

    def triangulated_local_patch(
        self,
        toroidal_angle,
        poloidal_angle,
        physical_half_width,
        geometric_tolerance,
    ):
        """Create a tolerance-driven local refinement of the continuous surface."""
        parameter_grid = self.local_patch_parameter_grid(
            toroidal_angle,
            poloidal_angle,
            physical_half_width,
            geometric_tolerance,
        )
        return self.triangulate_parameter_grid(*parameter_grid)

    def local_patch_parameter_grid(
        self,
        toroidal_angle,
        poloidal_angle,
        physical_half_width,
        geometric_tolerance,
    ):
        """Return an angular grid shared by corresponding radial surfaces."""
        phi, theta = self._canonical_angles(toroidal_angle, poloidal_angle)
        delta = 1e-5
        phi_speed = np.linalg.norm(
            self.evaluate(phi + delta, theta)
            - self.evaluate(phi - delta, theta)
        ) / (2.0 * delta)
        theta_speed = np.linalg.norm(
            self.evaluate(phi, theta + delta)
            - self.evaluate(phi, theta - delta)
        ) / (2.0 * delta)
        patch_radius = max(float(physical_half_width) * 1.75, 1.0)
        phi_half_span = patch_radius / phi_speed
        theta_half_span = patch_radius / theta_speed
        phi_min = max(float(self.phi_list[0]), phi - phi_half_span)
        phi_max = min(float(self.phi_list[-1]), phi + phi_half_span)
        target_edge = max(
            float(geometric_tolerance) * 4.0, patch_radius / 12.0
        )
        phi_count = max(
            9, int(np.ceil((phi_max - phi_min) * phi_speed / target_edge)) + 1
        )
        theta_count = max(
            17,
            int(np.ceil(2.0 * theta_half_span * theta_speed / target_edge))
            + 1,
        )
        phi_values = np.linspace(phi_min, phi_max, phi_count)
        theta_values = np.linspace(
            theta - theta_half_span, theta + theta_half_span, theta_count
        )
        return phi_values, theta_values

    def triangulate_parameter_grid(self, phi_values, theta_values):
        """Evaluate and triangulate this surface on a supplied angular grid."""
        loci = np.asarray(
            [
                [
                    self.evaluate(phi_value, theta_value)
                    for theta_value in theta_values
                ]
                for phi_value in phi_values
            ],
            dtype=float,
        )
        triangles = []
        for phi_index in range(len(phi_values) - 1):
            for theta_index in range(len(theta_values) - 1):
                a = loci[phi_index, theta_index]
                b = loci[phi_index + 1, theta_index]
                c = loci[phi_index + 1, theta_index + 1]
                d = loci[phi_index, theta_index + 1]
                triangles.append((a, b, c))
                triangles.append((a, c, d))
        return np.asarray(triangles, dtype=float)

    def populate_ribs(self):
        """Populates Rib class objects for each toroidal angle specified in
        the surface.
        """
        self.Ribs = [
            Rib(
                self.ref_surf,
                self.s,
                self.theta_list,
                phi,
                self.offset_mat[i, :],
                self.scale,
            )
            for i, phi in enumerate(self.phi_list)
        ]

    def calculate_loci(self):
        """Calls calculate_loci method in Rib class for each rib in the surface."""
        [rib.calculate_loci() for rib in self.Ribs]

    def _generate_pymoab_verts(self, mbc):
        """Generate MBTVERTEX entities from rib loci in all ribs.
        (Internal function not intended to be called externally)

        Arguments:
            mbc (PyMOAB Core): PyMOAB Core instance to add the MBVERTEX
                entities to.
        """
        [rib._generate_pymoab_verts(mbc) for rib in self.Ribs]

    def generate_surface(self):
        """Constructs a surface by lofting across a set of rib splines."""
        if not self.surface:
            self.surface = cq.Solid.makeLoft(
                [rib.generate_rib() for rib in self.Ribs]
            )

        return self.surface

    def get_loci(self):
        """Returns the set of point-loci defining the ribs in the surface."""
        return np.array([rib.rib_loci for rib in self.Ribs])


class Rib(object):
    """An object representing a curve formed by interpolating a spline through
    a set of points located in the same toroidal plane but differing poloidal
    angles and offset from a reference curve.

    Arguments:
        ref_surf (object): ReferenceSurface object. Must have a method
            'angles_to_xyz(toroidal_angles, poloidal_angles, s, scale)' that
            returns an Nx3 numpy array of cartesian coordinates for any closed
            flux surface label, s, poloidal angle (theta), and toroidal angle
            (phi).
        s (float): the normalized closed flux surface label defining the point
            of reference for offset.
        phi (float): the toroidal angle defining the plane in which the rib is
            located [rad].
        theta_list (np.array(double)): the set of poloidal angles specified for
            the rib [rad].
        offset_list (np.array(double)): the set of offsets from the curve
            defined by s for each toroidal angle, poloidal angle pair in the rib
            [cm].
        scale (float): a scaling factor between input and output data.
    """

    def __init__(self, ref_surf, s, theta_list, phi, offset_list, scale):
        self.ref_surf = ref_surf
        self.s = s
        self.theta_list = theta_list
        self.phi = phi
        self.offset_list = offset_list
        self.scale = scale

    def _normals(self):
        """Approximate the normal to the curve at each poloidal angle by first
        approximating the tangent to the curve and then taking the
        cross-product of that tangent with a vector defined as normal to the
        plane at this toroidal angle.
        (Internal function not intended to be called externally)

        Arguments:
            r_loci (np.array(double)): Cartesian point-loci of reference
                surface rib [cm].
        """
        tangents = self.ref_surf.calculate_tangents(
            self.phi, self.theta_list, self.s, self.scale
        )

        plane_norm = np.array([-np.sin(self.phi), np.cos(self.phi), 0])

        normals = np.cross(plane_norm, tangents)

        return normalize(normals)

    def calculate_loci(self):
        """Generates Cartesian point-loci for stellarator rib. Sets the last
        element to the value of the first to ensure the loop is closed exactly.
        """
        self.rib_loci = self.ref_surf.angles_to_xyz(
            self.phi, self.theta_list, self.s, self.scale
        )

        if not np.all(self.offset_list == 0):
            self.rib_loci += self.offset_list[:, np.newaxis] * self._normals()

        self.rib_loci[-1] = self.rib_loci[0]

    def _generate_pymoab_verts(self, mbc):
        """Converts point-loci to MBVERTEX and adds them to a PyMOAB
        Core instance. The first and last rib loci are identical. To avoid
        having separate MBVERTEX entities which are coincident, the last
        element in rib_loci is not made into an MBVERTEX, and the entity
        handle corresponding to the first rib locus is appended to the array
        of MBVERTEX, closing the loop.
        (Internal function not intended to be called externally)

        Arguments:
            mbc (PyMOAB Core): PyMOAB Core instance to add the MBVERTEX
                entities to.
        """
        self.mb_verts = mbc.create_vertices(
            self.rib_loci[0:-1].flatten()
        ).to_array()
        self.mb_verts = np.append(self.mb_verts, self.mb_verts[0])

    def generate_rib(self):
        """Constructs component rib by constructing a spline connecting all
        specified Cartesian point-loci.
        """
        rib_loci = [cq.Vector(tuple(r)) for r in self.rib_loci]
        spline = cq.Edge.makeSpline(rib_loci).close()
        rib_spline = cq.Wire.assembleEdges([spline]).close()

        return rib_spline


class InVesselComponentMesh(ToroidalMesh):
    """Generates a tetrahedral mesh of in-vessel component volumes via MOAB.
    This mesh is created using the point cloud of each component's Surface
    class objects and as such, each component's mesh will be one tetrahedron
    thick. Inherits from ToroidalMesh.

    Arguments:
        surfaces (list of object): the Surface class objects of the components
            in the mesh, ordered radially outward.
        gap_map (list of bool): an ordered map indicating gaps in the mesh. As
            such, should be one entry shorter than "surfaces" argument.
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.
    """

    def __init__(self, surfaces, gap_map, logger=None):
        super().__init__(logger=logger)

        self.surfaces = surfaces
        self.gap_map = gap_map

        self.volumes = []

        self._add_tags_to_core()

    @property
    def surfaces(self):
        return self._surfaces

    @surfaces.setter
    def surfaces(self, list):
        self._surfaces = list
        # Extract dimensions of surface point cloud
        self._num_ribs = len(list[0].phi_list)
        self._num_rib_pts = len(list[0].theta_list)

    @property
    def gap_map(self):
        return self._gap_map

    @gap_map.setter
    def gap_map(self, list):
        if len(list) != len(self._surfaces) - 1:
            e = AssertionError(
                "'gap_map' indicates gap regions in the mesh between the "
                "'surfaces' argument and as such, should be one entry shorter "
                "than 'surfaces'."
            )
            self._logger.error(e.args[0])
            raise e

        self._gap_map = list

    def _add_tags_to_core(self):
        """Creates PyMOAB core instance with source strength tag.
        (Internal function not intended to be called externally)
        """
        tag_type = types.MB_TYPE_DOUBLE
        tag_size = 1
        storage_type = types.MB_TAG_DENSE

        vol_tag_name = "Volume"
        self.volume_tag = self.mbc.tag_get_handle(
            vol_tag_name,
            tag_size,
            tag_type,
            storage_type,
            create_if_missing=True,
        )

    def create_vertices(self):
        """Creates mesh vertices and adds them to PyMOAB core."""
        self.coords = []
        for surface in self.surfaces:
            for rib in surface.Ribs:
                self.coords.extend(rib.rib_loci)
        self.coords = np.array(self.coords)
        self.add_vertices(self.coords)

    def _compute_and_tag_tet_volume(self, vert_ids, tet):
        """Computes tetrahedron volume, and sets the corresponding value of
        the respective tag for that tetrahedron.
        (Internal function not intended to be called externally)

        Arguments:
            vert_ids (list of int): tetrahedron vertex indices.
            tet (object): pymoab.EntityHandle of tetrahedron.

        Returns:
            tet_vol (float): volume of tetrahedron.
        """
        tet_vol = self._compute_tet_volume(vert_ids)
        # Tag tetrahedra with data
        self.mbc.tag_set_data(self.volume_tag, tet, [tet_vol])

        return tet_vol

    def create_mesh(self):
        """Creates volumetric mesh in real space."""
        for surface_idx, _ in enumerate(self.surfaces[:-1]):
            if self.gap_map[surface_idx]:
                continue  # Skip iteration if a gap is indicated
            for toroidal_idx in range(self._num_ribs - 1):
                for poloidal_idx in range(self._num_rib_pts - 1):
                    tets, vertex_id_list = self._create_tets_from_hex(
                        surface_idx, poloidal_idx, toroidal_idx
                    )
                    self.volumes.extend(
                        [
                            self._compute_and_tag_tet_volume(vert_ids, tet)
                            for vert_ids, tet in zip(vertex_id_list, tets)
                        ]
                    )

    def _get_vertex_id(self, vertex_idx):
        """Computes vertex index in row-major order as stored by MOAB from
        three-dimensional n x 3 matrix indices.
        (Internal function not intended to be called externally)

        Arguments:
            vertex_idx (list): vertex's 3-D grid indices in order
                [surface index, poloidal angle index, toroidal angle index]

        Returns:
            id (int): vertex index in row-major order as stored by MOAB
        """
        surface_idx, poloidal_idx, toroidal_idx = vertex_idx

        verts_per_surface = self._num_ribs * self._num_rib_pts
        surface_offset = surface_idx * verts_per_surface

        toroidal_offset = toroidal_idx * self._num_rib_pts

        poloidal_offset = poloidal_idx
        # Wrap around if poloidal angle is 2*pi
        if poloidal_idx == self._num_rib_pts - 1:
            poloidal_offset = 0

        id = surface_offset + toroidal_offset + poloidal_offset

        return id


class RadialBuild(object):
    """Parametrically defines ParaStell in-vessel component geometries.
    In-vessel component thicknesses are defined on a grid of toroidal and
    poloidal angles, and the first wall profile is defined by a closed flux
    surface extrapolation.

    Arguments:
        toroidal_angles (array of float): toroidal angles at which radial build
            is specified. This list should always begin at 0.0 and it is
            advised not to extend beyond one stellarator period. To build a
            geometry that extends beyond one period, make use of the 'repeat'
            parameter [deg].
        poloidal_angles (array of float): poloidal angles at which radial build
            is specified. This array should always span 360 degrees [deg].
        wall_s (float): closed flux surface label extrapolation at wall.
        radial_build (dict): dictionary representing the three-dimensional
            radial build of in-vessel components, including
            {
                'component': {
                    'thickness_matrix': 2-D matrix defining component
                        thickness at (toroidal angle, poloidal angle)
                        locations. Rows represent toroidal angles, columns
                        represent poloidal angles, and each must be in the same
                        order provided in toroidal_angles and poloidal_angles
                        [cm](ndarray(float)).
                    'mat_tag': DAGMC material tag for component in DAGMC
                        neutronics model (str, defaults to None). If None is
                        supplied, the 'component' key will be used.
                }
            }.
        split_chamber (bool): if wall_s > 1.0, separate interior vacuum
            chamber into plasma and scrape-off layer components (defaults to
            False). If an item with a 'sol' key is present in the radial_build
            dictionary, settting this to False will not combine the resultant
            'chamber' with 'sol'. To include a custom scrape-off layer
            definition for 'chamber', add an item with a 'chamber' key and
            desired 'thickness_matrix' value to the radial_build dictionary.
        logger (object): logger object (defaults to None). If no
            logger is supplied, a default logger will be instantiated.

    Optional attributes:
        plasma_mat_tag (str): DAGMC material tag to use for plasma if
            split_chamber is True (defaults to 'Vacuum').
        sol_mat_tag (str): DAGMC material tag to use for scrape-off layer if
            split_chamber is True (defaults to 'Vacuum').
        chamber_mat_tag (str): DAGMC material tag to use for interior vacuum
            chamber if split_chamber is False (defaults to 'Vacuum).
    """

    def __init__(
        self,
        toroidal_angles,
        poloidal_angles,
        wall_s,
        radial_build,
        split_chamber=False,
        logger=None,
        **kwargs,
    ):
        self.logger = logger
        self.toroidal_angles = toroidal_angles
        self.poloidal_angles = poloidal_angles
        self.wall_s = wall_s
        self.radial_build = radial_build
        self._user_layer_names = tuple(self.radial_build.keys())
        self.split_chamber = split_chamber

        for name in kwargs.keys() & (
            "plasma_mat_tag",
            "sol_mat_tag",
            "chamber_mat_tag",
        ):
            self.__setattr__(name, kwargs[name])

        self._logger.info("Constructing radial build...")

    @property
    def toroidal_angles(self):
        return self._toroidal_angles

    @toroidal_angles.setter
    def toroidal_angles(self, angle_list):
        if hasattr(self, "toroidal_angles"):
            e = AttributeError(
                '"toroidal_angles" cannot be set after class initialization. '
                "Please create new class instance to alter this attribute."
            )
            self._logger.error(e.args[0])
            raise e

        self._toroidal_angles = angle_list
        if self._toroidal_angles[0] != 0.0:
            e = ValueError("The first entry in toroidal_angles must be 0.0.")
            self._logger.error(e.args[0])
            raise e
        if self._toroidal_angles[-1] > 360.0:
            e = ValueError("Toroidal extent cannot exceed 360.0 degrees.")
            self._logger.error(e.args[0])
            raise e

    @property
    def poloidal_angles(self):
        return self._poloidal_angles

    @poloidal_angles.setter
    def poloidal_angles(self, angle_list):
        if hasattr(self, "poloidal_angles"):
            e = AttributeError(
                '"poloidal_angles" cannot be set after class initialization. '
                "Please create new class instance to alter this attribute."
            )
            self._logger.error(e.args[0])
            raise e

        self._poloidal_angles = angle_list
        if self._poloidal_angles[-1] - self._poloidal_angles[0] > 360.0:
            e = AssertionError(
                "Poloidal extent must span exactly 360.0 degrees."
            )
            self._logger.error(e.args[0])
            raise e

    @property
    def wall_s(self):
        return self._wall_s

    @wall_s.setter
    def wall_s(self, s):
        if hasattr(self, "wall_s"):
            e = AttributeError(
                '"wall_s" cannot be set after class initialization. Please '
                "create new class instance to alter this attribute."
            )
            self._logger.error(e.args[0])
            raise e

        self._wall_s = s
        if self._wall_s < 1.0:
            e = ValueError("wall_s must be greater than or equal to 1.0.")
            self._logger.error(e.args[0])
            raise e

    @property
    def radial_build(self):
        return self._radial_build

    @radial_build.setter
    def radial_build(self, build_dict):
        self._radial_build = build_dict

        for name, component in self._radial_build.items():
            component["thickness_matrix"] = np.array(
                component["thickness_matrix"]
            )
            if component["thickness_matrix"].shape != (
                len(self._toroidal_angles),
                len(self._poloidal_angles),
            ):
                e = AssertionError(
                    f"The dimensions of {name}'s thickness matrix "
                    f"{component['thickness_matrix'].shape} must match the "
                    "dimensions defined by the toroidal and poloidal angle "
                    "lists "
                    f"{len(self._toroidal_angles), len(self._poloidal_angles)}, "
                    "which define the rows and columns of the matrix, "
                    "respectively."
                )
                self._logger.error(e.args[0])
                raise e

            if np.any(component["thickness_matrix"] < 0):
                e = ValueError(
                    "Component thicknesses must be greater than or equal to 0. "
                    "Check thickness inputs for negative values."
                )
                self._logger.error(e.args[0])
                raise e

            if "mat_tag" not in component:
                self._set_mat_tag(name, name)

        if not hasattr(self, "_user_layer_names"):
            self._user_layer_names = tuple(self._radial_build.keys())

    @property
    def split_chamber(self):
        return self._split_chamber

    @split_chamber.setter
    def split_chamber(self, value):
        if hasattr(self, "split_chamber"):
            e = AttributeError(
                '"split_chamber" cannot be set after class initialization. '
                "Please create new class instance to alter this attribute."
            )
            self._logger.error(e.args[0])
            raise e

        self._split_chamber = value

        if self._split_chamber:
            if self._wall_s > 1.0 and "sol" not in self._radial_build:
                self.radial_build = {
                    "sol": {
                        "thickness_matrix": np.zeros(
                            (
                                len(self._toroidal_angles),
                                len(self._poloidal_angles),
                            )
                        )
                    },
                    **self.radial_build,
                }
                if not hasattr(self, "sol_mat_tag"):
                    self.sol_mat_tag = "Vacuum"

            inner_volume_name = "plasma"
            inner_volume_tag = "plasma_mat_tag"
        else:
            inner_volume_name = "chamber"
            inner_volume_tag = "chamber_mat_tag"

        self.radial_build = {
            inner_volume_name: {
                "thickness_matrix": np.zeros(
                    (len(self._toroidal_angles), len(self._poloidal_angles))
                )
            },
            **self.radial_build,
        }
        if not hasattr(self, inner_volume_tag):
            self.__setattr__(inner_volume_tag, "Vacuum")

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger_object):
        self._logger = log.check_init(logger_object)

    @property
    def plasma_mat_tag(self):
        return self._plasma_mat_tag

    @plasma_mat_tag.setter
    def plasma_mat_tag(self, mat_tag):
        self._plasma_mat_tag = mat_tag
        self._set_mat_tag("plasma", self._plasma_mat_tag)

    @property
    def sol_mat_tag(self):
        return self._sol_mat_tag

    @sol_mat_tag.setter
    def sol_mat_tag(self, mat_tag):
        self._sol_mat_tag = mat_tag
        self._set_mat_tag("sol", self._sol_mat_tag)

    @property
    def chamber_mat_tag(self):
        return self._chamber_mat_tag

    @chamber_mat_tag.setter
    def chamber_mat_tag(self, mat_tag):
        self._chamber_mat_tag = mat_tag
        self._set_mat_tag("chamber", self._chamber_mat_tag)

    def _set_mat_tag(self, name, mat_tag):
        """Sets DAGMC material tag for a given component.
        (Internal function not intended to be called externally)

        Arguments:
            name (str): name of component.
            mat_tag (str): DAGMC material tag.
        """
        self.radial_build[name]["mat_tag"] = mat_tag

    @property
    def user_layer_names(self):
        """User-supplied radial build component names before inner regions are added."""
        return self._user_layer_names


def parse_args():
    """Parser for running as a script."""
    parser = argparse.ArgumentParser(prog="invessel_build")

    parser.add_argument(
        "filename",
        help="YAML file defining ParaStell in-vessel component configuration",
    )
    parser.add_argument(
        "-e",
        "--export_dir",
        default="",
        help=(
            "Directory to which output files are exported (default: working directory)"
        ),
        metavar="",
    )
    parser.add_argument(
        "-l",
        "--logger",
        default=False,
        help=(
            "Flag to indicate whether to instantiate a logger object (default: False)"
        ),
        metavar="",
    )

    return parser.parse_args()


def generate_invessel_build():
    """Main method when run as a command line script."""
    args = parse_args()

    all_data = read_yaml_config(args.filename)

    if args.logger == True:
        logger = log.init()
    else:
        logger = log.NullLogger()

    vmec_file = all_data["vmec_file"]
    vmec_obj = read_vmec.VMECData(vmec_file)

    invessel_build_dict = all_data["invessel_build"]

    radial_build = RadialBuild(
        invessel_build_dict["toroidal_angles"],
        invessel_build_dict["poloidal_angles"],
        invessel_build_dict["wall_s"],
        invessel_build_dict["radial_build"],
        logger=logger,
        **invessel_build_dict,
    )

    invessel_build = InVesselBuild(
        vmec_obj, radial_build, logger=logger, **invessel_build_dict
    )

    invessel_build.populate_surfaces()
    invessel_build.calculate_loci()
    invessel_build.generate_components()

    invessel_build.export_step(export_dir=args.export_dir)


if __name__ == "__main__":
    generate_invessel_build()
