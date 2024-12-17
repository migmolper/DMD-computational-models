#!/usr/local/bin/python3.11
#!/opt/homebrew/bin/python3.11

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
# It's also possible to use the reduced notation by directly setting font.family:
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})

url_ad_MgHx = "https://www.ctcms.nist.gov/potentials/Download/2018--Smirnova-D-E-Starikov-S-V-Vlasova-A-M--Mg-H/1/Mg_H.adp.alloy.txt"
file_ad_MgHx = requests.get(url_ad_MgHx)
open('Mg_H.adp.alloy.txt', 'wb').write(file_ad_MgHx.content)

reader = open('Mg_H.adp.alloy.txt')
try:
    for i in range(0,3): line = reader.readline()

    # 
    line = reader.readline()
    print(line.split())

    # 
    line = reader.readline()
    info = line.split()
    N_rho = int(info[0])
    d_rho = float(info[1])
    N_r = int(info[2])
    d_r = float(info[3])
    cutoff_r = float(info[4])
    r_ij = np.linspace(0.0,cutoff_r,N_r)    

    # Mg atomistic data
    atomistic_info = reader.readline().split() # 
    Mg_atomic_number = int(atomistic_info[0])
    Mg_mass = float(atomistic_info[1])
    Mg_lattice_constant = float(atomistic_info[2])
    Mg_lattice_type = atomistic_info[3]

    # Mg embedding function and energy density function
    Mg_embedding_function_data = np.zeros(N_rho)
    for i in range(0,N_rho):
        Mg_embedding_function_data[i] = float(reader.readline().split()[0])
    Mg_energy_density_function_data = np.zeros(N_r)
    for i in range(0,N_r):
        Mg_energy_density_function_data[i] = float(reader.readline().split()[0])

    # H atomistic data
    atomistic_info = reader.readline().split() #
    H_atomic_number = int(atomistic_info[0])
    H_mass = float(atomistic_info[1])
    H_lattice_constant = float(atomistic_info[2])
    H_lattice_type = atomistic_info[3]

    # H embedding function and energy density function
    H_embedding_function_data = np.zeros(N_rho)
    for i in range(0,N_rho):
        H_embedding_function_data[i] = float(reader.readline().split()[0])
    H_energy_density_function_data = np.zeros(N_r)
    for i in range(0,N_r):
        H_energy_density_function_data[i] = float(reader.readline().split()[0])

    # Mg-Mg pair potential
    MgMg_pair_potential_data = np.zeros(N_r)
    for i in range(0,N_r):
        MgMg_pair_potential_data[i] = float(reader.readline().split()[0])

    # Mg-H pair potential
    MgH_pair_potential_data = np.zeros(N_r)
    for i in range(0,N_r):
        MgH_pair_potential_data[i] = float(reader.readline().split()[0])

    # H-H pair potential
    HH_pair_potential_data = np.zeros(N_r)
    for i in range(0,N_r):
        HH_pair_potential_data[i] = float(reader.readline().split()[0])

    # Mg-Mg u function (dipole)
    MgMg_dipole_function_u_data = np.zeros(N_r)
    for i in range(0,N_r):
        MgMg_dipole_function_u_data[i] = float(reader.readline().split()[0])

    # Mg-H u function (dipole)
    MgH_dipole_function_u_data = np.zeros(N_r)
    for i in range(0,N_r):
        MgH_dipole_function_u_data[i] = float(reader.readline().split()[0])

    # H-H u function (dipole)
    HH_dipole_function_u_data = np.zeros(N_r)
    for i in range(0,N_r):
        HH_dipole_function_u_data[i] = float(reader.readline().split()[0])

    # Mg-Mg u function (quadrupole)
    MgMg_quadrupole_function_w_data = np.zeros(N_r)
    for i in range(0,N_r):
        MgMg_quadrupole_function_w_data[i] = float(reader.readline().split()[0])

    # Mg-H u function (quadrupole)
    MgH_quadrupole_function_w_data = np.zeros(N_r)
    for i in range(0,N_r):
        MgH_quadrupole_function_w_data[i] = float(reader.readline().split()[0])

    # H-H u function (quadrupole)
    HH_quadrupole_function_w_data = np.zeros(N_r)
    for i in range(0,N_r):
        HH_quadrupole_function_w_data[i] = float(reader.readline().split()[0])

finally:
    reader.close()

# Compute splines and its derivatives (embedding)
cs_embedding_H = CubicSpline(np.linspace(0.0,cutoff_r,N_rho), H_embedding_function_data)
dr_cs_embedding_H = cs_embedding_H.derivative(1)
dr2_cs_embedding_H = cs_embedding_H.derivative(2)

cs_embedding_Mg = CubicSpline(np.linspace(0.0,cutoff_r,N_rho), Mg_embedding_function_data)
dr_cs_embedding_Mg = cs_embedding_Mg.derivative(1)
dr2_cs_embedding_Mg = cs_embedding_Mg.derivative(2)

# Compute splines and its derivatives (energy density)
cs_rho_H = CubicSpline(r_ij, H_energy_density_function_data)
dr_cs_rho_H = cs_rho_H.derivative(1)
dr2_cs_rho_H = cs_rho_H.derivative(2)

cs_rho_Mg = CubicSpline(r_ij, Mg_energy_density_function_data)
dr_cs_rho_Mg = cs_rho_Mg.derivative(1)
dr2_cs_rho_Mg = cs_rho_Mg.derivative(2)

# Compute splines and its derivatives (pairing potential)
cs_pair_MgMg = CubicSpline(r_ij[1:], MgMg_pair_potential_data[1:]/r_ij[1:])
dr_cs_pair_MgMg = cs_pair_MgMg.derivative(1)
dr2_cs_pair_MgMg = cs_pair_MgMg.derivative(2)

cs_pair_MgH = CubicSpline(r_ij[1:], MgH_pair_potential_data[1:]/r_ij[1:])
dr_cs_pair_MgH = cs_pair_MgH.derivative(1)
dr2_cs_pair_MgH = cs_pair_MgH.derivative(2)

cs_pair_HH = CubicSpline(r_ij[1:], HH_pair_potential_data[1:]/r_ij[1:])
dr_cs_pair_HH = cs_pair_HH.derivative(1)
dr2_cs_pair_HH = cs_pair_HH.derivative(2)

# Compute splines and its derivatives (dispole distortion potential)
cs_dipole_MgMg = CubicSpline(r_ij, MgMg_dipole_function_u_data)
dr_cs_dipole_MgMg = cs_dipole_MgMg.derivative(1)
dr2_cs_dipole_MgMg = cs_dipole_MgMg.derivative(2)

cs_dipole_MgH = CubicSpline(r_ij, MgH_dipole_function_u_data)
dr_cs_dipole_MgH = cs_dipole_MgH.derivative(1)
dr2_cs_dipole_MgH = cs_dipole_MgH.derivative(2)

cs_dipole_HH = CubicSpline(r_ij, HH_dipole_function_u_data)
dr_cs_dipole_HH = cs_dipole_HH.derivative(1)
dr2_cs_dipole_HH = cs_dipole_HH.derivative(2)

# Compute splines and its derivatives (quadrupole distortion potential)
cs_quadrupole_MgMg = CubicSpline(r_ij, MgMg_quadrupole_function_w_data)
dr_cs_quadrupole_MgMg = cs_quadrupole_MgMg.derivative(1)
dr2_cs_quadrupole_MgMg = cs_quadrupole_MgMg.derivative(2)

cs_quadrupole_MgH = CubicSpline(r_ij, MgH_quadrupole_function_w_data)
dr_cs_quadrupole_MgH = cs_quadrupole_MgH.derivative(1)
dr2_cs_quadrupole_MgH = cs_quadrupole_MgH.derivative(2)

cs_quadrupole_HH = CubicSpline(r_ij, HH_quadrupole_function_w_data)
dr_cs_quadrupole_HH = cs_quadrupole_HH.derivative(1)
dr2_cs_quadrupole_HH = cs_quadrupole_HH.derivative(2)

xs = np.arange(0.01, 7.5, 0.01)

print(cs_rho_H.c)
#print(dr_cs_rho_Mg.c)
#print(dr2_cs_rho_Mg.c)

## Save data to files
save_data = False
if (save_data):
    # Energy density
    np.savetxt('Files/embedding-Hx.csv', np.column_stack((xs,cs_embedding_H(xs))), delimiter=',')
    np.savetxt('Files/embedding-Mg.csv', np.column_stack((xs, cs_embedding_Mg(xs))), delimiter=',')

    # Energy density
    np.savetxt('Files/rho-Hx.csv', np.column_stack((xs,cs_rho_H(xs))), delimiter=',')
    np.savetxt('Files/rho-Mg.csv', np.column_stack((xs, cs_rho_Mg(xs))), delimiter=',')

    # Pairing term
    np.savetxt('Files/pair-MgMg.csv', np.column_stack((xs, cs_pair_MgMg(xs))), delimiter=',')
    np.savetxt('Files/pair-MgHx.csv', np.column_stack((r_ij[1:], MgH_pair_potential_data[1:]/r_ij[1:])), delimiter=',')
    np.savetxt('Files/pair-HxHx.csv', np.column_stack((xs, cs_pair_HH(xs))), delimiter=',')

    # Dipole term
    np.savetxt('Files/dipole-MgMg.csv', np.column_stack((xs, cs_dipole_MgMg(xs))), delimiter=',')
    np.savetxt('Files/dipole-MgHx.csv', np.column_stack((xs, cs_dipole_MgH(xs))), delimiter=',')
    np.savetxt('Files/dipole-HxHx.csv', np.column_stack((xs, cs_dipole_HH(xs))), delimiter=',')

    # Quadrupole term
    np.savetxt('Files/quadrupole-MgMg.csv', np.column_stack((xs, cs_quadrupole_MgMg(xs))), delimiter=',')
    np.savetxt('Files/quadrupole-MgHx.csv', np.column_stack((xs, cs_quadrupole_MgH(xs))), delimiter=',')
    np.savetxt('Files/quadrupole-HxHx.csv', np.column_stack((xs, cs_quadrupole_HH(xs))), delimiter=',')


## Plot results

plot_energy_density = False
if (plot_energy_density):
    fig, (ax_rho,ax_drho,ax_d2rho) = plt.subplots(3, figsize=(10, 6), constrained_layout=True)

    ax_rho.set_title(r'Energy density $\rho$ $\left[eV \right]$') 

    ax_rho.scatter(r_ij[::100], H_energy_density_function_data[::100], label=r'data H-H')
    ax_rho.plot(xs, cs_rho_H(xs), label=r'spline H-H')

    ax_rho.scatter(r_ij[::100], Mg_energy_density_function_data[::100], label=r'data Mg-H')
    ax_rho.plot(xs, cs_rho_Mg(xs), label=r'spline Mg-Mg')

    #ax_rho.set_ylim([-0.025, 0.15])
    #ax_rho.set_xlim([1.5, 6.0])
    ax_rho.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax_drho.set_title(r'First derivative of the energy density $\rho$ $\left[eV / \AA\right]$') 
    ax_drho.plot(xs, dr_cs_rho_H(xs), label=r'spline H-H')
    ax_drho.plot(xs, dr_cs_rho_Mg(xs), label=r'spline Mg-Mg')
    #ax_drho.set_ylim([-0.025, 0.15])
    #ax_drho.set_xlim([1.5, 6.0])
    ax_drho.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax_d2rho.set_title(r'Second derivative of the energy density $\rho$ $\left[eV / \AA^2\right]$') 
    ax_d2rho.plot(xs, dr2_cs_rho_H(xs), label=r'spline H-H')
    ax_d2rho.plot(xs, dr2_cs_rho_Mg(xs), label=r'spline Mg-Mg')
    #ax_d2rho.set_ylim([-5.0, 3.0])
    #ax_d2rho.set_xlim([1.5, 6.0])
    ax_d2rho.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel(r'$r\ \left[\AA\right]$',fontsize='x-large')
    plt.show()

plot_pairing = False
if (plot_pairing):
    fig, (ax_phi,ax_dphi,ax_d2phi) = plt.subplots(3, figsize=(10, 6), constrained_layout=True)

    ax_phi.set_title(r'Pairing $\phi$') 
    ax_phi.scatter(r_ij[1::100], MgMg_pair_potential_data[1::100]/r_ij[1::100], label=r'data Mg-Mg')
    ax_phi.plot(xs, cs_pair_MgMg(xs), label=r'spline Mg-Mg')

    ax_phi.scatter(r_ij[1::100], MgH_pair_potential_data[1::100]/r_ij[1::100], label=r'data Mg-H')
    ax_phi.plot(xs, cs_pair_MgH(xs), label=r'spline Mg-H')

    ax_phi.scatter(r_ij[1::100], HH_pair_potential_data[1::100]/r_ij[1::100], label=r'data H-H')
    ax_phi.plot(xs, cs_pair_HH(xs), label=r'spline H-H')

    ax_phi.set_ylim([-0.25, 1.0])
    ax_phi.set_xlim([1.5, 6.0])
    ax_phi.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax_dphi.set_title(r'First derivative of the pairing $\phi$') 
    ax_dphi.plot(xs, dr_cs_pair_MgMg(xs), label=r'spline Mg-Mg')

    ax_dphi.plot(xs, dr_cs_pair_MgH(xs), label=r'spline Mg-H')

    ax_dphi.plot(xs, dr_cs_pair_HH(xs), label=r'spline H-H')

    ax_dphi.set_ylim([-0.25, 0.25])
    ax_dphi.set_xlim([1.5, 6.0])
    ax_dphi.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax_d2phi.set_title(r'Second derivative of the pairing $\phi$') 
    ax_d2phi.plot(xs, dr2_cs_pair_MgMg(xs), label=r'spline Mg-Mg')

    ax_d2phi.plot(xs, dr2_cs_pair_MgH(xs), label=r'spline Mg-H')

    ax_d2phi.plot(xs, dr2_cs_pair_HH(xs), label=r'spline H-H')

    ax_d2phi.set_ylim([-25, 27])
    ax_d2phi.set_xlim([1.6, 6.0])
    ax_d2phi.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel(r'$r_{ij}\ \left[\AA\right]$',fontsize='x-large')
    plt.show()


plot_dipole = False
if (plot_dipole):

    fig, (ax_u,ax_du,ax_d2u) = plt.subplots(3, figsize=(10, 6), constrained_layout=True)

    ax_u.set_title(r'Dipole function $u$') 
    ax_u.scatter(r_ij[::100], MgMg_dipole_function_u_data[::100], label=r'data Mg-Mg')
    ax_u.plot(xs, cs_dipole_MgMg(xs), label=r'spline Mg-Mg')

    ax_u.scatter(r_ij[::100], MgH_dipole_function_u_data[::100], label=r'data Mg-H')
    ax_u.plot(xs, cs_dipole_MgH(xs), label=r'spline Mg-H')

    ax_u.scatter(r_ij[::100], HH_dipole_function_u_data[::100], label=r'data H-H')
    ax_u.plot(xs, cs_dipole_HH(xs), label=r'spline H-H')

    ax_u.set_ylim([-0.2, 0.2])
    ax_u.set_xlim([1.5, 6.0])
    ax_u.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax_du.set_title(r'First derivative of the dipole function $u$') 
    ax_du.plot(xs, dr_cs_dipole_HH(xs), label=r'spline H-H')

    ax_du.plot(xs, dr_cs_dipole_MgMg(xs), label=r'spline Mg-Mg')

    ax_du.plot(xs, dr_cs_dipole_MgH(xs), label=r'spline Mg-H')

    #ax_du.set_ylim([-0.2, 0.2])
    ax_du.set_xlim([1.5, 6.0])
    ax_du.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax_d2u.set_title(r'Second derivative of the dipole function $u$') 
    ax_d2u.plot(xs, dr2_cs_dipole_MgMg(xs), label=r'spline Mg-Mg')

    ax_d2u.plot(xs, dr2_cs_dipole_HH(xs), label=r'spline H-H')

    ax_d2u.plot(xs, dr2_cs_dipole_MgH(xs), label=r'spline Mg-H')

    ax_d2u.set_ylim([-2.0, 2.0])
    ax_d2u.set_xlim([1.5, 6.0])
    ax_d2u.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel(r'$r_{ij}\ \left[\AA\right]$',fontsize='x-large')
    plt.show()


plot_quadrupol = False
if (plot_quadrupol):
    fig, (ax_w,ax_dw,ax_d2w) = plt.subplots(3, figsize=(10, 6), constrained_layout=True)

    ax_w.set_title(r'Quadrupole function $w$') 
    ax_w.scatter(r_ij[::100], MgMg_quadrupole_function_w_data[::100], label=r'data Mg-Mg')
    ax_w.plot(xs, cs_quadrupole_MgMg(xs), label=r'spline Mg-Mg')

    ax_w.scatter(r_ij[::100], MgH_quadrupole_function_w_data[::100], label=r'data Mg-H')
    ax_w.plot(xs, cs_quadrupole_MgH(xs), label=r'spline Mg-H')

    ax_w.scatter(r_ij[::100], HH_quadrupole_function_w_data[::100], label=r'data H-H')
    ax_w.plot(xs, cs_quadrupole_HH(xs), label=r'spline H-H')

    ax_w.set_ylim([-0.2, 0.2])
    ax_w.set_xlim([1.5, 6.0])
    ax_w.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax_dw.set_title(r'First derivative of the quadrupole function $w$') 
    ax_dw.plot(xs, dr_cs_quadrupole_HH(xs), label=r'spline H-H')

    ax_dw.plot(xs, dr_cs_quadrupole_MgMg(xs), label=r'spline Mg-Mg')

    ax_dw.plot(xs, dr_cs_quadrupole_MgH(xs), label=r'spline Mg-H')

    #ax_dw.set_ylim([-0.2, 0.2])
    ax_dw.set_xlim([1.5, 6.0])
    ax_dw.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax_d2w.set_title(r'Second derivative of the quadrupole function $w$') 
    ax_d2w.plot(xs, dr2_cs_quadrupole_MgMg(xs), label=r'spline Mg-Mg')

    ax_d2w.plot(xs, dr2_cs_quadrupole_HH(xs), label=r'spline H-H')

    ax_d2w.plot(xs, dr2_cs_quadrupole_MgH(xs), label=r'spline Mg-H')

    ax_d2w.set_ylim([-2, 2])
    ax_d2w.set_xlim([1.5, 6.0])
    ax_d2w.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel(r'$r_{ij}\ \left[\AA\right]$',fontsize='x-large')
    plt.show()





