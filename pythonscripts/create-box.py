#!/usr/local/bin/python3.11

#!/usr/local/bin/python3.11

import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import (
    ReplicateModifier,
    AffineTransformationModifier,
    SliceModifier,
)
from ovito.data import DataCollection
import random

# Input parameters
Num_periodic_x = 5
Num_periodic_y = 5
Num_periodic_z = 5

D = 40  # nm
tau_H = 0.06 * D  # Thickness of the hydrogen layer
tau_a = 0.06 * D  # 0.06 * D  # Thickness of the amorphous layer
a = 5.5235100253371492  # Lattice parameter


def create_amorphous_inclusion(frame, data):
    """
    Function to amorphize a surface layer of a nanwire based on the paper:
    "Modeling thermal conductivity in silicon nanowires"
    """

    p_max = a / 5  # a / 10.0
    rad = D / 2.0 - tau_a

    sel = (
        (data.particles["Position"][..., 1]) ** 2
        + (data.particles["Position"][..., 2]) ** 2
    ) > rad * rad

    p_i = np.random.normal(p_max / 2.0, (p_max / 8.0) ** 2, data.particles_.count)
    d_i_x = np.zeros(data.particles_.count)
    d_i_y = np.random.uniform(1.0, -1.0, data.particles_.count)
    d_i_z = np.random.uniform(1.0, -1.0, data.particles_.count)

    rndm_d = np.matrix([d_i_x, d_i_y, d_i_z]).transpose()

    norm_rndm_d = np.linalg.norm(rndm_d, axis=1)

    rndm_displacement = np.matrix(
        [
            sel * p_i * d_i_x * (1.0 / norm_rndm_d),
            sel * p_i * d_i_y * (1.0 / norm_rndm_d),
            sel * p_i * d_i_z * (1.0 / norm_rndm_d),
        ]
    ).transpose()

    new_coordinates = data.particles["Position"] + rndm_displacement
    data.particles_.positions_[:] = new_coordinates

    data.particles_.create_property("amorphous-layer", dtype=int, data=sel)


#input_file_name = "../data/Mg-unitcell-hcp.dump"
#output_file_name = "Mg-hcp-cube-x{}-x{}-x{}.dump".format(
#    (int)(Num_periodic_x), (int)(Num_periodic_y), (int)(Num_periodic_z)
#)

#input_file_name = "../data/MgHx-unitcell-hcp.dump"
#output_file_name = "MgHx-hcp-cube-x{}-x{}-x{}.dump".format(
#    (int)(Num_periodic_x), (int)(Num_periodic_y), (int)(Num_periodic_z)
#)

input_file_name = "../data/MgHx-unitcell-rutile.dump"
output_file_name = "MgH2-rutile-cube-x{}-x{}-x{}.dump".format(
    (int)(Num_periodic_x), (int)(Num_periodic_y), (int)(Num_periodic_z)
)

pipeline = import_file(input_file_name)


# Create big block
pipeline.modifiers.append(
    ReplicateModifier(num_x=Num_periodic_x, num_y=Num_periodic_y, num_z=Num_periodic_z)
)

# Amorphization of the cell core
#pipeline.modifiers.append(create_amorphous_inclusion)

# Update changes in data
data = pipeline.compute()

# Get the number of positions
nsites = data.particles.count

site_type_value = np.int_(data.particles["Particle Type"][...])

# Define initial occupancy
molar_fraction_value = np.zeros(nsites)
site_value = np.zeros(nsites, dtype=np.int32)

# Define initial chemical potential
k_B = 8.617332478e-5
Temperature = 300
xi_0 = 1e-2
chm_value = np.ones(nsites)
thermal_value = np.ones(nsites)
stddev_q_0 = 0.01
stddev_q_value = stddev_q_0 * np.ones(nsites)
chm_bcc_value =  np.zeros(nsites,dtype=int)
thermal_bcc_value =  np.zeros(nsites,dtype=int)

for i in range(0, nsites):
    thermal_value[i] = 1.0 / (300 * k_B)
    if (data.particles["site"][i] == -1) and (data.particles["Particle Type"][i] == 1):
        # Mg
        site_value[i] = -1
        molar_fraction_value[i] = 1.0
        chm_value[i] = 0.0
        stddev_q_value[i] = 0.01
    elif (data.particles["site"][i] == 1) and (data.particles["Particle Type"][i] == 2):
        # Hx - T
        site_value[i] = 1
        molar_fraction_value[i] = 0.999
        chm_value[i] = 0.0
        stddev_q_value[i] = 0.01
    elif (data.particles["site"][i] == 2) and (data.particles["Particle Type"][i] == 2):
        # Hx - O
        site_value[i] = 2
        molar_fraction_value[i] = 0.0001
        chm_value[i] = 0.0
        stddev_q_value[i] = 0.01
    else:
        print("wrong kind of site")

# Define initial value of the standard deviation of q

data.particles_.create_property("Stdv q", data=stddev_q_value)
data.particles_.create_property("Molar fraction", data=molar_fraction_value)
data.particles_.create_property("chemical-multp", data=chm_value)
data.particles_.create_property("thermal-multp", data=thermal_value)
data.particles_.create_property("chemical-multp-bcc", data=chm_bcc_value)
data.particles_.create_property("thermal-multp-bcc", data=thermal_bcc_value)


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
        "chemical-multp-bcc",
        "thermal-multp-bcc"
    ],
)
