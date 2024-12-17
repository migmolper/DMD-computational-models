#!/usr/local/bin/python3.11
import sys
import os
import h5py
import numpy as np
from ovito.data import *
from ovito.pipeline import *
from ovito.io import import_file, export_file

input_file_name = sys.argv[1]

# Verify that the input file exists and has the correct file extension.
if not os.path.isfile(input_file_name) or not input_file_name.lower().endswith(".hdf5"):
    print("Error: The input file does not exist or does not have the extension .hdf5")
    sys.exit(1)

output_file_name = os.path.splitext(input_file_name)[0] + ".dump"

# Read file
hf = h5py.File(input_file_name, "r")

# Create the data collection containing a Particles object:
data = DataCollection()
particles = data.create_particles()
n_sites = hf["specie"].size

particles = data.create_particles(count=n_sites, vis_params={"radius": 1.4})

# Read fields
spc = hf["specie"][:].squeeze()
spc_prop = particles.create_property("Particle Type", data=spc)

mean_q = hf["particle_fields/DMSwarmSharedField_DMSwarmPIC_coor"][:]
mean_q_prop = particles.create_property("Position", data=mean_q)

stdv_q = hf["particle_fields/DMSwarmSharedField_stdv-q"][:]
stdv_q_prop = particles.create_property("Stdv q", data=stdv_q[0])

xi = hf["particle_fields/DMSwarmSharedField_molar-fraction"][:]
xi_prop = particles.create_property("Molar fraction", data=xi[0])

gamma = hf["particle_fields/DMSwarmSharedField_gamma"][:]
gamma_prop = particles.create_property("chemical-multp", data=gamma[0])

beta = hf["particle_fields/DMSwarmSharedField_beta"][:]
beta_prop = particles.create_property("thermal-multp", data=beta[0])

# gamma_bcc = hf["gamma-bcc"][:].squeeze()
# if gamma_bcc.shape[0] > 0:
#    gamma_bcc_prop = particles.create_property("chemical-multp-bcc", data=gamma_bcc)
# else:
#    gamma_bcc_prop = particles.create_property(
#        "chemical-multp-bcc", data=np.zeros(n_sites, dtype=np.int32)
#    )

# beta_bcc = hf["beta-bcc"][:].squeeze()

# if beta_bcc.shape[0] > 0:
#    beta_bcc_prop = particles.create_property("thermal-multp-bcc", data=beta_bcc)
# else:
#    beta_bcc_prop = particles.create_property(
#        "thermal-multp-bcc", data=np.zeros(n_sites, dtype=np.int32)
#    )


# Create the simulation box:
cell = SimulationCell(pbc=(True, True, True))
cell[...] = [
    [hf.attrs["Xmax"] - hf.attrs["Xmin"], 0, 0, hf.attrs["Xmin"]],
    [0, hf.attrs["Ymax"] - hf.attrs["Ymin"], 0, hf.attrs["Ymin"]],
    [0, 0, hf.attrs["Zmax"] - hf.attrs["Zmin"], hf.attrs["Zmin"]],
]
cell.vis.line_width = 0.1
data.objects.append(cell)


# Save file
export_file(
    data,
    output_file_name,
    "lammps/dump",
    columns=[
        "Particle Type",
        "Position.X",
        "Position.Y",
        "Position.Z",
        "Stdv q",
        "Molar fraction",
        "chemical-multp",
        "thermal-multp",
        #        "chemical-multp-bcc",
        #        "thermal-multp-bcc",
    ],
)

hf.close()
