import math
import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import (
    ReplicateModifier,
    ExpressionSelectionModifier,
    DeleteSelectedModifier,
)


# Input parameters
R = 40  # nm
tau_H = 3  # Thickness of the hydrogen layer
tau_a = 0.1 * (2 * R)  # 0.06 * (2*R) # Thickness of the amorphous layer
a = 5.5235100253371492  # Lattice parameter

def select_surface_layer(frame, data):
    """
    Function to select the atoms in the surface of the nanowire
    """

    p_max = a / 5  # a / 10.0
    R_tau_H = R - tau_H

    sel_1 = data.particles["Position"][..., 2] < R_tau_H * np.cos(math.radians(30))

    sel_2 = data.particles["Position"][..., 2] > -R_tau_H * np.cos(math.radians(30))

    sel_3 = data.particles["Position"][..., 2] > -R_tau_H * np.cos(math.radians(30))

    sel_4 = data.particles["Position"][..., 2] < np.tan(math.radians(60)) * (
        R_tau_H - data.particles["Position"][..., 1]
    )

    sel_5 = data.particles["Position"][..., 2] < np.tan(math.radians(60)) * (
        R_tau_H + data.particles["Position"][..., 1]
    )

    sel_6 = data.particles["Position"][..., 2] > np.tan(math.radians(60)) * (
        -R_tau_H - data.particles["Position"][..., 1]
    )

    sel_7 = data.particles["Position"][..., 2] > np.tan(math.radians(60)) * (
        -R_tau_H + data.particles["Position"][..., 1]
    )

    sel = sel_1 * sel_2 * sel_3 * sel_4 * sel_5 * sel_6 * sel_7
    
    amorph_sel = np.invert(sel)

    p_i = np.random.normal(p_max / 2.0, (p_max / 8.0) ** 2, data.particles_.count)
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

    sel_H = data.particles["Particle Type"][...] == 2

    data.particles_.create_property(
        "chemical-multp-bcc",
        dtype=int,
        data=sel * sel_H,
    )


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

    a = data.cell[:, 0]
    b = data.cell[:, 1]
    c = data.cell[:, 2]
    o = data.cell[:, 3]

    matrix = np.empty((3, 4))
    matrix[0, 0] = a[0]
    matrix[1, 1] = (coords_max - coords_min)[1]
    matrix[2, 2] = (coords_max - coords_min)[2]

    matrix[0, 3] = o[0]
    matrix[1, 3] = coords_min[1]
    matrix[2, 3] = coords_min[2]

    # Assign the cell matrix - or create whole new SimulationCell object in
    # the DataCollection if there isn't one already.
    data.create_cell(matrix, (False, False, False))


# input_file_name = "../data/Mg-unitcell-hcp.dump"
# output_file_name = "Mg-hcp-hex-nanowire-R-{}-A.dump".format((int)(R))

input_file_name = "../data/MgHx-unitcell-hcp.dump"
output_file_name = "MgHx-hcp-hex-nanowire-R-{}-A.dump".format((int)(R))

# input_file_name = "../data/MgHx-unitcell-rutile.dump"
# output_file_name = "MgH2-rutile-hex-nanowire-R-{}.dump".format((int)(R))

pipeline = import_file(input_file_name)

# Cell periodicity
Num_periodic_x = 3
Num_periodic_y = 50
Num_periodic_z = 50


# Create big block of Mg
pipeline.modifiers.append(
    ReplicateModifier(num_x=Num_periodic_x, num_y=Num_periodic_y, num_z=Num_periodic_z)
)

# Sculpt hexagon using cut-planes
def cut_hexagon(frame, data):
    sel_1 = data.particles["Position"][..., 2] < R * np.cos(math.radians(30))
    sel_2 = data.particles["Position"][..., 2] > -R * np.cos(math.radians(30))
    sel_3 = data.particles["Position"][..., 2] > -R * np.cos(math.radians(30))
    sel_4 = data.particles["Position"][..., 2] < np.tan(math.radians(60)) * (
        R - data.particles["Position"][..., 1]
    )
    sel_5 = data.particles["Position"][..., 2] < np.tan(math.radians(60)) * (
        R + data.particles["Position"][..., 1]
    )
    sel_6 = data.particles["Position"][..., 2] > np.tan(math.radians(60)) * (
        -R - data.particles["Position"][..., 1]
    )
    sel_7 = data.particles["Position"][..., 2] > np.tan(math.radians(60)) * (
        -R + data.particles["Position"][..., 1]
    )

    sel = sel_1 * sel_2 * sel_3 * sel_4 * sel_5 * sel_6 * sel_7


    data.particles_.create_property("Selection", data=np.invert(sel))


# Amorphization of the surface layer
# pipeline.modifiers.append(create_amorphous_surface_layer)

# Sculpt the block using a hexagon

# Select Mg particles in the boundary
pipeline.modifiers.append(select_surface_layer)

pipeline.modifiers.append(cut_hexagon)
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
k_B = 8.617332478e-5
Temperature = 300
xi_0 = 1e-2
chm_value = np.ones(nsites)
thermal_value = np.ones(nsites)
stddev_q_0 = 0.01
stddev_q_value = stddev_q_0 * np.ones(nsites)
thermal_bcc_value = np.zeros(nsites, dtype=int)
chemical_bcc_value = np.zeros(nsites, dtype=int)

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
        molar_fraction_value[i] = 0.0001
        if data.particles["chemical-multp-bcc"][i] == 1:
            molar_fraction_value[i] = 0.999
        chm_value[i] = 0.0
        stddev_q_value[i] = 0.01
    elif (data.particles["site"][i] == 2) and (data.particles["Particle Type"][i] == 2):
        # Hx - O
        site_value[i] = 2
        molar_fraction_value[i] = 0.0001
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
# data.particles_.create_property("chemical-multp-bcc", data=chemical_bcc_value)
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
        "thermal-multp-bcc",
    ],
)
