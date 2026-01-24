import numpy as np
import openmc
import parastell.parastell as ps
import time  # Import the time module for timing

# Define directory to export all output files to
export_dir = ""
# Define plasma equilibrium VMEC file
vmec_file = "wout_vmec.nc"

# Instantiate ParaStell build
stellarator = ps.Stellarator(vmec_file)

# Define build parameters for in-vessel components
toroidal_angles = [0.0, 11.25, 22.5, 33.75, 45.0, 56.25, 67.5, 78.75, 90.0]
poloidal_angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0]
wall_s = 1.08

# Define a matrix of uniform unit thickness
uniform_unit_thickness = np.ones((len(toroidal_angles), len(poloidal_angles)))

radial_build_dict = {
    "first_wall": {
        "thickness_matrix": uniform_unit_thickness * 5,
        "mat_tag": "iron",
    },
    "breeder": {
        "thickness_matrix": (
            [
                [75.0, 75.0, 75.0, 25.0, 25.0, 25.0, 75.0, 75.0, 75.0],
                [75.0, 75.0, 75.0, 25.0, 25.0, 75.0, 75.0, 75.0, 75.0],
                [75.0, 75.0, 25.0, 25.0, 75.0, 75.0, 75.0, 75.0, 75.0],
                [65.0, 25.0, 25.0, 65.0, 75.0, 75.0, 75.0, 75.0, 65.0],
                [45.0, 45.0, 75.0, 75.0, 75.0, 75.0, 75.0, 45.0, 45.0],
                [65.0, 75.0, 75.0, 75.0, 75.0, 65.0, 25.0, 25.0, 65.0],
                [75.0, 75.0, 75.0, 75.0, 75.0, 25.0, 25.0, 75.0, 75.0],
                [75.0, 75.0, 75.0, 75.0, 25.0, 25.0, 75.0, 75.0, 75.0],
                [75.0, 75.0, 75.0, 25.0, 25.0, 25.0, 75.0, 75.0, 75.0],
            ]
        ),
        "mat_tag": "iron",
    },
    "back_wall": {
        "thickness_matrix": uniform_unit_thickness * 5,
        "mat_tag": "iron",
    },
    "shield": {
        "thickness_matrix": uniform_unit_thickness * 50,
        "mat_tag": "iron",
    },
    "vacuum_vessel": {
        "thickness_matrix": uniform_unit_thickness * 10,
        "mat_tag": "tungsten",
    },
}

# Start timing before constructing in-vessel components
start_time = time.perf_counter()

# Construct in-vessel components
stellarator.construct_invessel_build(
    toroidal_angles,
    poloidal_angles,
    wall_s,
    radial_build_dict,
    use_pydagmc=True,
)

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"Construction of in-vessel components completed in {elapsed_time:.2f} seconds.\n")

for surf in stellarator.invessel_build.dag_model.surfaces:
    print(surf)
    print(surf.surf_sense)

print(stellarator.invessel_build.dag_model.volumes)

print(stellarator.invessel_build.dag_model.groups)
