#!/usr/local/bin/python3.11

#!/usr/local/bin/python3.11

import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import (
    ReplicateModifier,
    AffineTransformationModifier,
    SliceModifier, DeleteSelectedModifier
)
from ovito.data import DataCollection
import random

R = 10
tau_H = 3  # Thickness of the hydrogen layer
tau_a = 0.1 * (2 * R)  # 0.06 * (2*R) # Thickness of the amorphous layer
a = 5.5235100253371492  # Lattice parameter


def select_free_surface_H(frame, data):


    p_max = a / 5  # a / 10.0
    R_tau_H = R - tau_H  

    center = (0, 0, 0)

    sel_1 = (
        np.sqrt(
            (data.particles["Position"][..., 0] - center[0]) ** 2
            + (data.particles["Position"][..., 1] - center[1]) ** 2
            + (data.particles["Position"][..., 2] - center[2]) ** 2            
        )
    ) < R_tau_H

    amorph_sel = np.invert(sel_1)

    p_i = 0.0;# np.random.normal(p_max / 2.0, (p_max / 8.0) ** 2, data.particles_.count)
    d_i_x = np.zeros(data.particles_.count)
    d_i_y = np.random.uniform(1.0, -1.0, data.particles_.count)
    d_i_z = np.random.uniform(1.0, -1.0, data.particles_.count)

    rndm_d = np.matrix([d_i_x, d_i_y, d_i_z]).transpose()

    norm_rndm_d = np.linalg.norm(rndm_d, axis=1)

    rndm_displacement = np.matrix(
        [
            amorph_sel * p_i * d_i_x * (1.0 / norm_rndm_d),
            amorph_sel * p_i * d_i_y * (1.0 / norm_rndm_d),
            amorph_sel * p_i * d_i_z * (1.0 / norm_rndm_d),
        ]
    ).transpose()

    new_coordinates = data.particles["Position"] + rndm_displacement
    data.particles_.positions_[:] = new_coordinates

    sel_2 = data.particles["Particle Type"][...] == 2

    data.particles_.create_property("chemical-multp-bcc", dtype=int, data=amorph_sel * sel_2)


def Shrink_wrap_simulation_box(frame, data):

    # There's nothing we can do if there are no input particles.
    if not data.particles or data.particles.count == 0:
        return

    # Compute min/max range of particle coordinates.
    coords_min = np.amin(data.particles.positions, axis=0) - 3 * np.ones(3)
    coords_max = np.amax(data.particles.positions, axis=0) + 3 * np.ones(3)

    # Build the new 3x4 cell matrix:
    #   (x_max-x_min  0            0            x_min)
    #   (0            y_max-y_min  0            y_min)
    #   (0            0            z_max-z_min  z_min)
    matrix = np.empty((3, 4))
    matrix[:, :3] = np.diag(coords_max - coords_min)
    matrix[:, 3] = coords_min

    # Assign the cell matrix - or create whole new SimulationCell object in
    # the DataCollection if there isn't one already.
    data.create_cell(matrix, (False, False, False))

#input_file_name = "../data/Mg-unitcell-hcp.dump"
#output_file_name = "Mg-hcp-sphere-R-{}.dump".format(R)

input_file_name = "../data/MgHx-unitcell-hcp.dump"
output_file_name = "MgHx-hcp-sphere-R-{}.dump".format(R)

#input_file_name = "../data/MgHx-unitcell-rutile.dump"
#output_file_name = "MgH2-rutile-sphere-R-{}.dump".format(R)

pipeline = import_file(input_file_name)

# Cell periodicity
Num_periodic_x = 30
Num_periodic_y = 30
Num_periodic_z = 30

# Create big block
pipeline.modifiers.append(
    ReplicateModifier(num_x=Num_periodic_x,
                      num_y=Num_periodic_y, num_z=Num_periodic_z)
)

# Sculpt sphere using cut-planes
def cut_sphere(frame, data):
    sel = (
        np.sqrt(
            data.particles["Position"][..., 0] ** 2
            + data.particles["Position"][..., 1] ** 2
            + data.particles["Position"][..., 2] ** 2
        )
    ) < R

    data.particles_.create_property("Selection", data=np.invert(sel))

# Select Mg particles in the boundary
pipeline.modifiers.append(select_free_surface_H)
pipeline.modifiers.append(cut_sphere)
pipeline.modifiers.append(DeleteSelectedModifier())

# Update changes in data
data = pipeline.compute()

# Select Mg particles in the boundary
pipeline.modifiers.append(Shrink_wrap_simulation_box)
data = pipeline.compute()

# Get the number of positions
nsites = data.particles.count

site_type_value = np.int_(data.particles["Particle Type"][...])

# Define initial occupancy
molar_fraction_value = np.zeros(nsites)
site_value = np.zeros(nsites, dtype=np.int32)

# Define initial chemical potential
k_B = 8.617332478E-5
Temperature = 300
xi_0 = 1e-2
chm_value = np.ones(nsites)
thermal_value = np.ones(nsites)
stddev_q_0 = 0.01
stddev_q_value = stddev_q_0 * np.ones(nsites)
thermal_bcc_value = np.zeros(nsites,dtype=int)
chemical_bcc_value = np.zeros(nsites,dtype=int)

for i in range(0, nsites):
    thermal_value[i] = 1.0/(300 * k_B)
    if (data.particles["site"][i] == -1) and (data.particles["Particle Type"][i] == 1):
        # Mg
        site_value[i] = -1
        molar_fraction_value[i] = 1.0
        chm_value[i] = 0.0
        stddev_q_value[i] = 0.01
    elif (data.particles["site"][i] == 1) and (data.particles["Particle Type"][i] == 2):
        # Hx - T
        site_value[i] = 1
        molar_fraction_value[i] = 0.0
        if data.particles["chemical-multp-bcc"][i] == 1:
            molar_fraction_value[i] = 0.999
        chm_value[i] = 0.0
        stddev_q_value[i] = 0.01
    elif (data.particles["site"][i] == 2) and (data.particles["Particle Type"][i] == 2):
        # Hx - O
        site_value[i] = 2
        molar_fraction_value[i] = 0.0
#        if data.particles["chemical-multp-bcc"][i] == 1:
#            molar_fraction_value[i] = 0.999
        chm_value[i] = 0.0
        stddev_q_value[i] = 0.01
    else:
        print("wrong kind of site")

# Define initial value of the standard deviation of q

data.particles_.create_property("Stdv q", data=stddev_q_value)
data.particles_.create_property("Molar fraction", data=molar_fraction_value)
data.particles_.create_property("chemical-multp", data=chm_value)
#data.particles_.create_property("chemical-multp-bcc", data=chemical_bcc_value)
data.particles_.create_property("thermal-multp", data=thermal_value)
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
