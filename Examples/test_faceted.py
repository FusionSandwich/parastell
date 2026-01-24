from parastell.invessel_build import *
import openmc
import time
import os


def extract_ss(ss_file):
    """Extracts list of source strengths for each tetrahedron from input file.

    Arguments:
        ss_file (str): path to source strength input file.

    Returns:
        strengths (list): list of source strengths for each tetrahedron (1/s).
            Returned only if source mesh is generated.
    """
    strengths = []

    file_obj = open(ss_file, "r")
    data = file_obj.readlines()
    for line in data:
        strengths.append(float(line))

    return strengths


def nwl_transport(
    dagmc_geom, source_mesh, tor_ext, ss_file, num_parts, surface_ids
):
    """Performs neutron transport on first wall geometry via OpenMC.

    Arguments:
        dagmc_geom (str): path to DAGMC geometry file.
        source_mesh (str): path to source mesh file.
        tor_ext (float): toroidal extent of model (deg).
        ss_file (str): source strength input file.
        num_parts (int): number of source particles to simulate.
    """
    tor_ext = np.deg2rad(tor_ext)

    strengths = extract_ss(ss_file)

    # Initialize OpenMC model
    model = openmc.model.Model()

    dag_univ = openmc.DAGMCUniverse(dagmc_geom, auto_geom_ids=False)

    # Define problem boundaries
    vac_surf = openmc.Sphere(r=10000, surface_id=99999, boundary_type="vacuum")
    per_init = openmc.YPlane(boundary_type="periodic", surface_id=99998)
    per_fin = openmc.Plane(
        a=np.sin(tor_ext),
        b=-np.cos(tor_ext),
        c=0,
        d=0,
        boundary_type="periodic",
        surface_id=99997,
    )

    # Define first period of geometry
    region = -vac_surf & +per_init & +per_fin
    period = openmc.Cell(cell_id=9996, region=region, fill=dag_univ)
    geometry = openmc.Geometry([period])
    model.geometry = geometry

    # Define run settings
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = num_parts
    settings.batches = 1

    # Define neutron source
    mesh = openmc.UnstructuredMesh(source_mesh, "moab")
    src = openmc.IndependentSource()
    src.space = openmc.stats.MeshSpatial(
        mesh, strengths=strengths, volume_normalized=False
    )
    src.angle = openmc.stats.Isotropic()
    src.energy = openmc.stats.Discrete([14.1e6], [1.0])
    settings.source = [src]

    # Track surface crossings
    settings.surf_source_write = {
        "cellfrom": 1,
        "surface_ids": surface_ids,
        "max_particles": num_parts,
    }

    model.settings = settings

    model.export_to_model_xml()

    os.system("openmc model.xml")


print(time.time)
vmec_file = "plasma_wistelld.nc"
vmec = read_vmec.VMECData(vmec_file)

radial_build_dict = {}

toroidal_angles = np.linspace(0, 90, 72)
poloidal_angles = np.linspace(0, 360, 96)
wall_s = 1.08

radial_build = RadialBuild(
    toroidal_angles, poloidal_angles, wall_s, radial_build_dict
)
build = InVesselBuild(vmec, radial_build, split_chamber=False)
build.populate_surfaces()
build.calculate_loci()

test_surf = build.Surfaces["chamber"]

test_surf.build_faceted_surface()
print(time.time)

surface_ids = test_surf.bin_surface_ids
bin_areas = test_surf.bin_areas

np.save("surface_ids.npy", surface_ids)
np.save("bin_areas.npy", bin_areas)

nwl_transport(
    "test_facets.h5m",
    "source_mesh.h5m",
    90,
    "strengths.txt",
    1000000,
    [int(surf_id) for surf_id in surface_ids.flatten()],
)
