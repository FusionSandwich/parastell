import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Restored imports for pymoab-based DAGMC generation
import pymoab
from pymoab import core, types
import dagmc

# For geometry and meshing
import cubit
import cadquery as cq
import cad_to_dagmc
import pystell.read_vmec as read_vmec

from . import log
from . import cubit_io
from .utils import (
    normalize,
    expand_list,
    read_yaml_config,
    filter_kwargs,
    m2cm,
)

###############################################################################
#                NEW: Adaptive Refinement Helper Functions                    #
###############################################################################

def subdivide_if_needed(t1, t2, o1, o2, coords_func, max_curvature,
                        curvature_estimate, current_level, max_level):
    """Recursively subdivide the angle range [t1, t2] if an approximate
    curvature is too high, up to `max_level` times."""
    if curvature_estimate < max_curvature or current_level >= max_level:
        # No need to subdivide further
        return [t1, t2], [o1, o2]
    else:
        # Subdivide the interval
        tm = 0.5 * (t1 + t2)
        om = 0.5 * (o1 + o2)

        # Evaluate curvature on the two new segments
        p1 = coords_func(t1)
        pm = coords_func(tm)
        p2 = coords_func(t2)

        v1 = pm - p1
        length1 = np.linalg.norm(v1)
        curv1 = 0 if length1 == 0 else 1.0 / length1

        v2 = p2 - pm
        length2 = np.linalg.norm(v2)
        curv2 = 0 if length2 == 0 else 1.0 / length2

        # Recursively subdivide each half
        left_theta, left_offset = subdivide_if_needed(
            t1, tm, o1, om, coords_func, max_curvature,
            curv1, current_level + 1, max_level
        )
        right_theta, right_offset = subdivide_if_needed(
            tm, t2, om, o2, coords_func, max_curvature,
            curv2, current_level + 1, max_level
        )

        # Combine them, removing the repeated middle point from the left side:
        return left_theta[:-1] + right_theta, left_offset[:-1] + right_offset


def refine_angles_by_curvature(theta_list, phi, vmec_obj, s, offset_list, scale,
                               max_curvature=0.05, max_subdivisions=3):
    """Returns updated theta angles + offset_list, refined where curvature is high."""
    def coords(theta):
        # Evaluate reference (x,y,z) at this poloidal angle:
        return scale * np.array(vmec_obj.vmec2xyz(s, theta, phi))

    refined_theta = [theta_list[0]]
    refined_offset = [offset_list[0]]

    for i in range(len(theta_list) - 1):
        t1, t2 = theta_list[i], theta_list[i+1]
        o1, o2 = offset_list[i], offset_list[i+1]

        p1 = coords(t1)
        p2 = coords(t2)
        segment = p2 - p1
        length = np.linalg.norm(segment)

        # Simple "curvature" measure ~ 1.0 / chord length
        curvature_estimate = 0 if length == 0 else 1.0 / length

        sub_thetas, sub_offsets = subdivide_if_needed(
            t1, t2, o1, o2, coords, max_curvature, curvature_estimate,
            current_level=0, max_level=max_subdivisions
        )
        refined_theta += sub_thetas[:-1]
        refined_offset += sub_offsets[:-1]

    refined_theta.append(theta_list[-1])
    refined_offset.append(offset_list[-1])

    return np.array(refined_theta), np.array(refined_offset)


###############################################################################
#                         MOAB Tri-Facet Helpers                              #
###############################################################################

def create_moab_tris_from_corners(corners, mbc, get_or_create_vertices):
    """Create 2 MOAB triangles from 4 corner points [corners 0..3 in xyz]."""
    # corners: [corner1, corner2, corner3, corner4] in xyz
    tri_1_verts = get_or_create_vertices([corners[2], corners[1], corners[0]])
    tri_2_verts = get_or_create_vertices([corners[3], corners[2], corners[0]])

    tri_1 = mbc.create_element(types.MBTRI, tri_1_verts)
    tri_2 = mbc.create_element(types.MBTRI, tri_2_verts)
    return [tri_1, tri_2]


###############################################################################
#                          Geometry & Build Classes                           #
###############################################################################

export_allowed_kwargs = ["export_cad_to_dagmc", "dagmc_filename"]


def orient_spline_surfaces(volume_id):
    """Extracts the inner and outer surface IDs for a given volume in Cubit."""
    surfaces = cubit.get_relatives("volume", volume_id, "surface")

    spline_surfaces = []
    for surface in surfaces:
        if cubit.get_surface_type(surface) == "spline surface":
            spline_surfaces.append(surface)

    if len(spline_surfaces) == 1:
        outer_surface_id = spline_surfaces[0]
        inner_surface_id = None
    else:
        # Outer surface will have larger XY bounding box
        if (
            cubit.get_bounding_box("surface", spline_surfaces[1])[4]
            > cubit.get_bounding_box("surface", spline_surfaces[0])[4]
        ):
            outer_surface_id = spline_surfaces[1]
            inner_surface_id = spline_surfaces[0]
        else:
            outer_surface_id = spline_surfaces[0]
            inner_surface_id = spline_surfaces[1]

    return inner_surface_id, outer_surface_id


class InVesselBuild(object):
    """Parametrically models fusion stellarator in-vessel components using
    plasma equilibrium VMEC data and a user-defined radial build."""

    def __init__(self, vmec_obj, radial_build, logger=None, **kwargs):
        self.logger = logger
        self.vmec_obj = vmec_obj
        self.radial_build = radial_build

        # Original numeric defaults
        self.repeat = 0
        self.num_ribs = 61
        self.num_rib_pts = 67
        self.scale = m2cm

        # For DAGMC (pymoab)
        self.mbc = core.Core()
        self.dag_model = dagmc.DAGModel(self.mbc)
        self._vertex_map = {}  # used by MOAB for deduplication

        # Adaptive refinement flags
        self.adaptive_refinement = kwargs.get("adaptive_refinement", False)
        self.max_curvature = kwargs.get("max_curvature", 0.05)
        self.max_subdivisions = kwargs.get("max_subdivisions", 3)

        # Overwrite defaults from kwargs if present
        for name in kwargs.keys() & (
            "repeat",
            "num_ribs",
            "num_rib_pts",
            "scale",
        ):
            self.__setattr__(name, kwargs[name])

        self.Surfaces = {}
        self.Components = {}

    @property
    def vmec_obj(self):
        return self._vmec_obj

    @vmec_obj.setter
    def vmec_obj(self, vmec_object):
        self._vmec_obj = vmec_object

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
                'exceeds 360 degrees. Check "repeat" and "toroidal_angles".'
            )
            self._logger.error(e.args[0])
            raise e

    def _interpolate_offset_matrix(self, offset_mat):
        """Interpolates offset for expanded angle lists using pchip/cubic splines."""
        interpolator = RegularGridInterpolator(
            (
                self.radial_build.toroidal_angles,
                self.radial_build.poloidal_angles,
            ),
            offset_mat,
            method="pchip",
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
        """Create Surface objects for each radial_build component."""
        self._logger.info("Populating surface objects for in-vessel components...")

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
            interpolated_offset_mat = self._interpolate_offset_matrix(offset_mat)

            surf = Surface(
                self._vmec_obj,
                s,
                self._poloidal_angles_exp,
                self._toroidal_angles_exp,
                interpolated_offset_mat,
                self.scale,
                adaptive_refinement=self.adaptive_refinement,
                max_curvature=self.max_curvature,
                max_subdivisions=self.max_subdivisions,
            )
            self.Surfaces[name] = surf

        [surface.populate_ribs() for surface in self.Surfaces.values()]

    def calculate_loci(self):
        """Compute the 3D loci for each surface's Ribs."""
        self._logger.info("Computing point cloud for in-vessel components...")
        [surface.calculate_loci() for surface in self.Surfaces.values()]

    ###########################################################################
    #                CadQuery-based geometry (generate_components)            #
    ###########################################################################
    def generate_components(self):
        """Construct a CAD solid for each component by lofting surfaces
        and optionally cutting interior surfaces."""
        self._logger.info("Constructing CadQuery solids for in-vessel components...")

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

            # fuse repeated segments if self.repeat > 0
            for angle in segment_angles:
                rot_segment = segment.rotate((0, 0, 0), (0, 0, 1), angle)
                component = component.fuse(rot_segment)

            self.Components[name] = component
            interior_surface = outer_surface

    ###########################################################################
    #               MOAB-based geometry (generate_components_pydagmc)         #
    ###########################################################################
    def generate_components_pydagmc(self):
        """Generates all components in the DAGMC model (pymoab-based).
        Triangulates surfaces between consecutive Ribs, sets surface senses,
        and writes out a .h5m DAGMC file.
        """
        self._logger.info("Generating DAGMC surfaces and volumes via PyMOAB...")

        # We'll store each surface in the DAGModel
        curved_surface_ids = []

        # Step 1: Triangulate each "outer" surface in self.Surfaces
        for i, surface in enumerate(self.Surfaces.values()):
            # We make the new surface
            # Triangulate using the logic from surface.generate_pydagmc_surface
            mb_tris = []
            # We'll connect Ribs[i] to Ribs[i+1] across poloidal angles
            # The user had a similar function previously. Let's replicate it here:

            def get_or_create_vertices(coord_list):
                """Helper that returns or creates MB vertices, avoiding duplicates."""
                vert_handles = []
                for c in coord_list:
                    # Round to mitigate floating precision issues
                    c_tup = tuple(np.round(c, decimals=7))
                    if c_tup in self._vertex_map:
                        vert_handles.append(self._vertex_map[c_tup])
                    else:
                        new_v = self.mbc.create_vertices([c_tup])[0]
                        self._vertex_map[c_tup] = new_v
                        vert_handles.append(new_v)
                return vert_handles

            # Triangulate the "loft" between adjacent ribs
            for rib, next_rib in zip(surface.Ribs[:-1], surface.Ribs[1:]):
                for rib_pt_index in range(len(rib.rib_loci) - 1):
                    corner1 = rib.rib_loci[rib_pt_index]
                    corner2 = rib.rib_loci[rib_pt_index + 1]
                    corner3 = next_rib.rib_loci[rib_pt_index + 1]
                    corner4 = next_rib.rib_loci[rib_pt_index]
                    corners = [corner1, corner2, corner3, corner4]

                    # Add 2 triangles from these 4 corners
                    # If you want to reverse the winding for some surfaces, you can:
                    # corners = corners[::-1]  # if needed
                    mb_tris += create_moab_tris_from_corners(
                        corners, self.mbc, get_or_create_vertices
                    )

            # Add new MOAB surface
            surface_set = dagmc.Surface.create(self.dag_model)
            self.mbc.add_entities(surface_set.handle, mb_tris)
            curved_surface_ids.append(surface_set.id)

        # Step 2: Optionally add "end caps" or side surfaces between surfaces
        # (If your geometry has multiple surfaces to connect. If not needed, skip.)
        end_cap_surface_ids = []
        all_surfaces_list = list(self.Surfaces.values())
        for s_idx in range(len(all_surfaces_list) - 1):
            current_surface = all_surfaces_list[s_idx]
            next_surface = all_surfaces_list[s_idx + 1]

            # connect start ribs:
            mb_tris = self.connect_ribs_with_tris_moab(
                current_surface.Ribs[0], next_surface.Ribs[0], reverse=False
            )
            end_cap_start = dagmc.Surface.create(self.dag_model)
            self.mbc.add_entities(end_cap_start.handle, mb_tris)

            # connect end ribs:
            mb_tris = self.connect_ribs_with_tris_moab(
                current_surface.Ribs[-1], next_surface.Ribs[-1], reverse=True
            )
            end_cap_end = dagmc.Surface.create(self.dag_model)
            self.mbc.add_entities(end_cap_end.handle, mb_tris)

            end_cap_surface_ids.append(
                list(self.dag_model.surfaces_by_id.keys())[-2:]
            )

        # Step 3: Create Volumes for each surface (plus one extra, e.g. plasma interior)
        total_surfaces = len(all_surfaces_list)
        for _ in range(total_surfaces + 1):  # an extra volume
            dagmc.Volume.create(self.dag_model)

        # Step 4: Assign surface senses for the main "curved" surfaces
        for i, sid in enumerate(curved_surface_ids):
            if i == 0:
                # e.g. plasma (Volume 1) to next layer (Volume 2)
                self.dag_model.surfaces_by_id[sid].surf_sense = [
                    self.dag_model.volumes_by_id[1],
                    self.dag_model.volumes_by_id[2],
                ]
            elif i != total_surfaces - 1:
                # Middle surfaces: Volume (i+1) to Volume (i+2)
                self.dag_model.surfaces_by_id[sid].surf_sense = [
                    self.dag_model.volumes_by_id[i + 1],
                    self.dag_model.volumes_by_id[i + 2],
                ]
            else:
                # Last surface: Volume N to None
                self.dag_model.surfaces_by_id[sid].surf_sense = [
                    self.dag_model.volumes_by_id[i + 1],
                    None,
                ]

        # Step 5: Assign senses to end caps
        volume_handles = list(self.dag_model.volumes_by_id.values())
        for i, pair_ids in enumerate(end_cap_surface_ids):
            for sid in pair_ids:
                if i + 2 < len(volume_handles):
                    self.dag_model.surfaces_by_id[sid].surf_sense = [
                        volume_handles[i + 1],
                        volume_handles[i + 2],
                    ]
                else:
                    msg = (f"Cannot assign surf_sense for surface ID {sid}. "
                           f"Need at least {i+2} volumes, only have {len(volume_handles)}.")
                    self._logger.error(msg)
                    raise IndexError(msg)

        # Step 6: Apply materials to volumes
        for vol, (layer_name, layer_data) in zip(
            self.dag_model.volumes,
            list(self.radial_build.radial_build.items()),
        ):
            mat = layer_data.get("mat_tag", layer_name)
            group = dagmc.Group.create(self.dag_model, name="mat:" + mat)
            group.add_set(vol)

        # Optionally export to file(s) here, or let the user call dag_model.write_file()
        self._logger.info("DAGMC surfaces & volumes built successfully.")

    def connect_ribs_with_tris_moab(self, rib1, rib2, reverse=False):
        """Helper to connect two Ribs with triangular facets in MOAB."""
        mb_tris = []

        def get_or_create_vertices(coord_list):
            vert_handles = []
            for c in coord_list:
                c_tup = tuple(np.round(c, decimals=7))
                if c_tup in self._vertex_map:
                    vert_handles.append(self._vertex_map[c_tup])
                else:
                    new_v = self.mbc.create_vertices([c_tup])[0]
                    self._vertex_map[c_tup] = new_v
                    vert_handles.append(new_v)
            return vert_handles

        for rib_loci_index in range(len(rib1.rib_loci) - 1):
            corner1 = rib1.rib_loci[rib_loci_index]
            corner2 = rib1.rib_loci[rib_loci_index + 1]
            corner3 = rib2.rib_loci[rib_loci_index + 1]
            corner4 = rib2.rib_loci[rib_loci_index]
            corners = [corner1, corner2, corner3, corner4]
            if reverse:
                mb_tris += create_moab_tris_from_corners(
                    corners[::-1], self.mbc, get_or_create_vertices
                )
            else:
                mb_tris += create_moab_tris_from_corners(
                    corners, self.mbc, get_or_create_vertices
                )
        return mb_tris

    ###########################################################################
    #                 Export & Meshing-related methods                        #
    ###########################################################################
    def get_loci(self):
        return np.array([surface.get_loci() for surface in self.Surfaces.values()])

    def merge_layer_surfaces(self):
        """Merges surfaces in Cubit by ID instead of imprinting all."""
        prev_outer_surface_id = None
        for data in self.radial_build.radial_build.values():
            inner_surf, outer_surf = orient_spline_surfaces(data["vol_id"])
            if prev_outer_surface_id is None:
                prev_outer_surface_id = outer_surf
            else:
                cubit.cmd(f"merge surface {inner_surf} {prev_outer_surface_id}")
                prev_outer_surface_id = outer_surf

    def import_step_cubit(self):
        """Imports STEP files for each in-vessel component volume into Cubit."""
        for name, data in self.radial_build.radial_build.items():
            vol_id = cubit_io.import_step_cubit(name, self.export_dir)
            data["vol_id"] = vol_id

    def export_step(self, export_dir=""):
        """Export CAD solids as STEP files via CadQuery."""
        self._logger.info("Exporting STEP files for in-vessel components...")
        self.export_dir = export_dir
        for name, component in self.Components.items():
            export_path = Path(self.export_dir) / Path(name).with_suffix(".step")
            cq.exporters.export(component, str(export_path))

    def export_cad_to_dagmc(self, dagmc_filename="dagmc", export_dir=""):
        """Exports DAGMC neutronics H5M file of in-vessel components (via CAD-to-DAGMC)."""
        self._logger.info("Exporting DAGMC model (CAD-to-DAGMC) for in-vessel components...")
        model = cad_to_dagmc.CadToDagmc()
        for name, component in self.Components.items():
            model.add_cadquery_object(
                component,
                material_tags=[self.radial_build.radial_build[name]["mat_tag"]],
            )
        export_path = Path(export_dir) / Path(dagmc_filename).with_suffix(".h5m")
        model.export_dagmc_h5m_file(filename=str(export_path))

    def export_component_mesh(self, components, mesh_size=5, import_dir="", export_dir=""):
        """Creates a tetrahedral mesh in Cubit and exports it as an H5M file."""
        for comp in components:
            vol_id = cubit_io.import_step_cubit(comp, import_dir)
            cubit.cmd(f"volume {vol_id} scheme tetmesh")
            cubit.cmd(f"volume {vol_id} size auto factor {mesh_size}")
            cubit.cmd(f"mesh volume {vol_id}")
            cubit_io.export_mesh_cubit(
                filename=comp,
                export_dir=export_dir,
                delete_upon_export=True,
            )


class Surface(object):
    """An object representing a surface formed by lofting across Ribs at
    different toroidal angles, offset from a reference surface."""

    def __init__(
        self,
        vmec_obj,
        s,
        theta_list,
        phi_list,
        offset_mat,
        scale,
        adaptive_refinement=False,
        max_curvature=0.05,
        max_subdivisions=3,
    ):
        self.vmec_obj = vmec_obj
        self.s = s
        self.theta_list = theta_list
        self.phi_list = phi_list
        self.offset_mat = offset_mat
        self.scale = scale

        self.adaptive_refinement = adaptive_refinement
        self.max_curvature = max_curvature
        self.max_subdivisions = max_subdivisions

        self.surface = None
        self.Ribs = []

    def populate_ribs(self):
        """Populates Rib objects for each toroidal angle in phi_list."""
        self.Ribs = [
            Rib(
                self.vmec_obj,
                self.s,
                self.theta_list,
                phi,
                self.offset_mat[i, :],
                self.scale,
                adaptive_refinement=self.adaptive_refinement,
                max_curvature=self.max_curvature,
                max_subdivisions=self.max_subdivisions,
            )
            for i, phi in enumerate(self.phi_list)
        ]

    def calculate_loci(self):
        """Compute final coordinates for each Rib."""
        [rib.calculate_loci() for rib in self.Ribs]

    def generate_surface(self):
        """Loft across Rib splines to form a CadQuery Solid."""
        if not self.surface:
            self.surface = cq.Solid.makeLoft(
                [rib.generate_rib() for rib in self.Ribs]
            )
        return self.surface

    def get_loci(self):
        """Return array of (N_ribs, N_pts, 3) for all Rib loci."""
        return np.array([rib.rib_loci for rib in self.Ribs])


class Rib(object):
    """A single 'rib' in a toroidal plane at phi, offset from flux surface s."""

    def __init__(
        self,
        vmec_obj,
        s,
        theta_list,
        phi,
        offset_list,
        scale,
        adaptive_refinement=False,
        max_curvature=0.05,
        max_subdivisions=3,
    ):
        self.vmec_obj = vmec_obj
        self.s = s
        self.theta_list = theta_list
        self.phi = phi
        self.offset_list = offset_list
        self.scale = scale

        self.adaptive_refinement = adaptive_refinement
        self.max_curvature = max_curvature
        self.max_subdivisions = max_subdivisions

        self.rib_loci = None

    def _vmec2xyz(self, poloidal_offset=0):
        """Return Nx3 array for s, phi, over poloidal angles with an offset."""
        return self.scale * np.array(
            [
                self.vmec_obj.vmec2xyz(self.s, theta, self.phi)
                for theta in (self.theta_list + poloidal_offset)
            ]
        )

    def _normals(self):
        """Approximate normal by crossing plane normal with local tangents."""
        eps = 1e-4
        next_pt_loci = self._vmec2xyz(eps)
        tangent = next_pt_loci - self.rib_loci
        plane_norm = np.array([-np.sin(self.phi), np.cos(self.phi), 0])
        norm = np.cross(plane_norm, tangent)
        return normalize(norm)

    def calculate_loci(self):
        """Compute final rib_loci, refining angles if adaptive_refinement is True."""
        if self.adaptive_refinement:
            refined_theta, refined_offset = refine_angles_by_curvature(
                self.theta_list,
                self.phi,
                self.vmec_obj,
                self.s,
                self.offset_list,
                self.scale,
                max_curvature=self.max_curvature,
                max_subdivisions=self.max_subdivisions,
            )
            self.theta_list = refined_theta
            self.offset_list = refined_offset

        self.rib_loci = self._vmec2xyz()
        if not np.all(self.offset_list == 0):
            self.rib_loci += self.offset_list[:, np.newaxis] * self._normals()

    def generate_rib(self):
        """Construct a spline (Wire) from the rib_loci points."""
        rib_pts = [cq.Vector(tuple(r)) for r in self.rib_loci]
        spline = cq.Edge.makeSpline(rib_pts).close()
        rib_spline = cq.Wire.assembleEdges([spline]).close()
        return rib_spline


class RadialBuild(object):
    """Parametrically defines ParaStell in-vessel component geometries."""

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
                '"toroidal_angles" cannot be set after init. '
                "Create a new class instance to alter this attribute."
            )
            self._logger.error(e.args[0])
            raise e

        self._toroidal_angles = angle_list
        if self._toroidal_angles[0] != 0.0:
            e = ValueError("The first entry in toroidal_angles must be 0.0.")
            self._logger.error(e.args[0])
            raise e
        if self._toroidal_angles[-1] > 360.0:
            e = ValueError("Toroidal extent cannot exceed 360.0 deg.")
            self._logger.error(e.args[0])
            raise e

    @property
    def poloidal_angles(self):
        return self._poloidal_angles

    @poloidal_angles.setter
    def poloidal_angles(self, angle_list):
        if hasattr(self, "poloidal_angles"):
            e = AttributeError(
                '"poloidal_angles" cannot be set after init.'
            )
            self._logger.error(e.args[0])
            raise e
        self._poloidal_angles = angle_list
        if self._poloidal_angles[-1] - self._poloidal_angles[0] > 360.0:
            e = AssertionError("Poloidal extent must span exactly 360.0 deg.")
            self._logger.error(e.args[0])
            raise e

    @property
    def wall_s(self):
        return self._wall_s

    @wall_s.setter
    def wall_s(self, s):
        if hasattr(self, "wall_s"):
            e = AttributeError(
                '"wall_s" cannot be set after init. '
                "Create new class instance to alter this attribute."
            )
            self._logger.error(e.args[0])
            raise e
        self._wall_s = s
        if self._wall_s < 1.0:
            e = ValueError("wall_s must be >= 1.0.")
            self._logger.error(e.args[0])
            raise e

    @property
    def radial_build(self):
        return self._radial_build

    @radial_build.setter
    def radial_build(self, build_dict):
        self._radial_build = build_dict
        for name, component in self._radial_build.items():
            component["thickness_matrix"] = np.array(component["thickness_matrix"])
            if component["thickness_matrix"].shape != (
                len(self._toroidal_angles),
                len(self._poloidal_angles),
            ):
                e = AssertionError(
                    f"{name}'s thickness matrix shape "
                    f"{component['thickness_matrix'].shape} must match "
                    f"{(len(self._toroidal_angles), len(self._poloidal_angles))}."
                )
                self._logger.error(e.args[0])
                raise e
            if np.any(component["thickness_matrix"] < 0):
                e = ValueError("Thickness must be >= 0. Negative found.")
                self._logger.error(e.args[0])
                raise e
            if "mat_tag" not in component:
                self._set_mat_tag(name, name)

    @property
    def split_chamber(self):
        return self._split_chamber

    @split_chamber.setter
    def split_chamber(self, value):
        if hasattr(self, "split_chamber"):
            e = AttributeError(
                '"split_chamber" cannot be set after init.'
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
        """Sets DAGMC material tag for a given component."""
        self._radial_build[name]["mat_tag"] = mat_tag


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
        help="Directory for output files (default: working directory)",
        metavar="",
    )
    parser.add_argument(
        "-l",
        "--logger",
        default=False,
        help="Flag to instantiate a logger object (default: False)",
        metavar="",
    )

    return parser.parse_args()


def generate_invessel_build():
    """Main method when run as a command line script."""
    args = parse_args()

    all_data = read_yaml_config(args.filename)

    if args.logger in [True, "True", "true"]:
        logger = log.init()
    else:
        logger = log.NullLogger()

    vmec_file = all_data["vmec_file"]
    vmec_obj = read_vmec.VMECData(vmec_file)

    invessel_build_dict = all_data["invessel_build"]

    # Create radial build
    radial_build = RadialBuild(
        invessel_build_dict["toroidal_angles"],
        invessel_build_dict["poloidal_angles"],
        invessel_build_dict["wall_s"],
        invessel_build_dict["radial_build"],
        logger=logger,
        **invessel_build_dict,
    )

    # Create InVesselBuild object (with optional adaptive refinement flags)
    invessel_build = InVesselBuild(
        vmec_obj,
        radial_build,
        logger=logger,
        **invessel_build_dict,
    )

    invessel_build.populate_surfaces()
    invessel_build.calculate_loci()

    # Example usage:
    # - If you want a CadQuery solid: 
    invessel_build.generate_components()
    invessel_build.export_step(export_dir=args.export_dir)

    # - If you want PyMOAB-based DAGMC geometry:
    #   (Check if "generate_components_pydagmc" is requested)
    if invessel_build_dict.get("generate_components_pydagmc", False):
        invessel_build.generate_components_pydagmc()
        # Then if you want to export the resulting MB geometry:
        invessel_build.dag_model.write_file("all_surfaces.vtk")
        invessel_build.dag_model.write_file("dagmc.h5m")

    # - If also requested to do CAD-to-DAGMC:
    if invessel_build_dict.get("export_cad_to_dagmc", False):
        invessel_build.export_cad_to_dagmc(
            export_dir=args.export_dir,
            **(filter_kwargs(invessel_build_dict, ["dagmc_filename"])),
        )


if __name__ == "__main__":
    generate_invessel_build()
