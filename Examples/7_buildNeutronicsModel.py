import openmc

model = openmc.model.Model()
dagmc_model_file = "dagmc.h5m"

materials = openmc.Materials()

iron = openmc.Material(name="iron")
iron.add_element("Fe", 1.0)
iron.set_density("g/cm3", 7.874)
materials.append(iron)

tungsten = openmc.Material(name="tungsten")
tungsten.add_element("W", 1.0)
tungsten.set_density("g/cm3", 19.25)
materials.append(tungsten)

model.materials = materials

dag_uni = openmc.DAGMCUniverse(filename=dagmc_model_file)
bounding_sphere = openmc.Sphere(
    r=10000, boundary_type="vacuum", surface_id=10000
)
vac_cell = openmc.Cell(region=-bounding_sphere, fill=dag_uni, cell_id=10000)
geometry = openmc.Geometry([vac_cell])
model.geometry = geometry

source = openmc.Source()
source.space = openmc.stats.Point((5, 5, 5))  # Location of the source
source.angle = openmc.stats.Isotropic()  # Isotropic source

settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.source = [source]
settings.particles = 10000
settings.batches = 10

model.settings = settings

mesh = openmc.RegularMesh()
mesh.dimension = [20, 10, 10]
mesh.lower_left = [0, 0, 0]
mesh.upper_right = [20, 10, 10]
mesh_filter = openmc.MeshFilter(mesh)

flux_tally = openmc.Tally(name="flux_tally")
flux_tally.filters = [mesh_filter]
flux_tally.scores = ["flux"]

model.tallies = [flux_tally]

model.run()
