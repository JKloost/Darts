import darts.models.physics.chemical
import numpy as np
import pandas as pd

import sys
sys.path.append('C:/Users/Jaro/Documents/inversemodelling/code_thesis/DARTS_1D_model/physics_sup_2/')  # Change this

from model_Reaktoro import Model
# from model_phreeqc import Model
from darts.engines import value_vector, redirect_darts_output
import matplotlib.pyplot as plt
from matplotlib import cm
import os
# import cProfile

redirect_darts_output('run.log')
n = Model()
n.init()
nu, x, z_c, z_e, density, pH, poro_diff, poro_diff2, poro_diff_tot, poro_diff_og, gas_list, gas_list2 = [], [], [], [], [], [], [], [], [], [], [], []
H2O, Ca, Na, Cl, OH, H, NaCO3, CO3, HCO3, Calcite, NaOH, H2CO3, K, CO2, Halite = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
n.run_python(1000, timestep_python=True)
# for i in range(300):
#     if i > 0:
#         n.load_restart_data()
#     n.run_python(1, timestep_python=True)
#     n.save_restart_data()
#     Xn = np.array(n.physics.engine.X, copy=False)
#     nc = n.property_container.nc + n.thermal
#     ne = n.property_container.n_e
#     nb = n.reservoir.nb
#     log_flag = n.property_container.log_flag
#     P = Xn[0:ne * (nb):ne]
#     poro = n.poro
#     z_darts = np.zeros((ne, len(P)))
#     for i in range(1, ne):
#         if log_flag == 1:
#             z_darts[i - 1] = np.exp(Xn[i:ne * (nb):ne])
#         else:
#             z_darts[i - 1] = Xn[i:ne * (nb):ne]
#     z_darts[-1] = np.ones(len(P)) - list(map(sum, zip(*z_darts[:-1])))
#     nu_output, x_output, z_c_output, density_output, pH_output, gas = n.flash_properties(z_darts[:, 0], 320,
#                                                                                              P[i])  # itor
#     H2O.append(z_c_output[0]-gas[0])
#     Na.append(z_c_output[2])
#     Cl.append(z_c_output[3])
#     CO2.append(z_c_output[1]-gas[1])
#     Halite.append(z_c_output[4])
#     gas_list.append(gas[0])
#     gas_list2.append(gas[1])
#     #print(z_darts[:,2])
#     z_e.append(z_darts[:, 2])
#     pH.append(pH_output)
#     if density_output[1] > 1050 or density_output[1] < 950:
#         density_output[1] = 1012
#     density.append(density_output)
#     nu.append(nu_output)
#     poro_diff_og.append((poro[i] * (1 - (nu_output[-1] * density_output[-1]) / np.sum(density_output))) - poro[i])
#
# # print(z_e)
# # print(density)
# # print(nu)
# print(poro_diff_og)
# plt.figure(3)
# plt.plot(H2O, label='H2O')
# plt.plot(gas_list, '--', label='H2O(g)')
# plt.plot(CO2, label='CO2')
# plt.plot(gas_list2, '--', label='CO2(g)')
# plt.plot(Na, label='Ca+2')
# plt.plot(Cl, label='CO3-2')
# plt.plot(Halite, label='Calcite')
# #plt.yscale('log')
# plt.legend()
# # plt.legend()
# plt.ylabel('z_c')
# plt.xlabel('time [days]')
# plt.title('Composition', y=1)
# plt.show()
#
# plt.figure(4)
# plt.plot([item[0] for item in z_e], label='H2O')
# plt.plot([item[1] for item in z_e], label='CO2')
# plt.plot([item[2] for item in z_e], label='Ca+2')
# plt.plot([item[3] for item in z_e], label='CO3-2')
# # plt.yscale('log')
# plt.xlabel('time [days]')
# plt.ylabel('z_e')
# plt.legend()
# plt.show()
# plt.figure(5)
# plt.plot(pH)
# plt.ylabel('SI')
# plt.xlabel('time [days]')
# plt.show()
#
# plt.figure(6)
# # plt.plot(dx, poro_diff, label='Halite')
# plt.plot(poro_diff_og, label='Calcite')
# # plt.plot(poro_diff_og, label='Total')
# plt.ylabel('$\Delta$ porosity')
# #plt.yscale('log')
# #plt.ylim([0,1])
# plt.xlabel('time [days]')
# plt.title('$\Delta$ porosity', y=1)
# plt.tight_layout()
# plt.legend()
# plt.show()
# exit()
# n.print_timers()
# n.print_stat()
# time_data = pd.DataFrame.from_dict(n.physics.engine.time_data)
# time_data.to_pickle("darts_time_data.pkl")
# n.save_restart_data()
# writer = pd.ExcelWriter('time_data.xlsx')
# time_data.to_excel(writer, 'Sheet1')
# writer.save()

# n.load_restart_data()
# time_data = pd.read_pickle("darts_time_data.pkl")

# if 0:
#     n.run_python(1, timestep_python=True)
#     n.print_timers()
#     n.print_stat()
#     time_data = pd.DataFrame.from_dict(n.physics.engine.time_data)
#     time_data.to_pickle("darts_time_data.pkl")
#     n.save_restart_data()
#     writer = pd.ExcelWriter('time_data.xlsx')
#     time_data.to_excel(writer, 'Sheet1')
#     writer.save()
#     n.print_and_plot('time_data.xlsx')
# else:
#     n.load_restart_data()
#     time_data = pd.read_pickle("darts_time_data.pkl")
n.print_timers()
n.print_stat()
""" plot results 2D """
Xn = np.array(n.physics.engine.X, copy=False)
nc = n.property_container.nc + n.thermal
ne = n.property_container.n_e
nb = n.reservoir.nb
log_flag = n.property_container.log_flag
P = Xn[0:ne*nb:ne]
dx = n.reservoir.global_data['dx']
dx = np.cumsum(dx)
poro = n.poro
z_darts = np.zeros((ne, len(P)))
for i in range(1, ne):
    if log_flag == 1:
        z_darts[i-1] = np.exp(Xn[i:ne*nb:ne])
    else:
        z_darts[i - 1] = Xn[i:ne * nb:ne]
z_darts[-1] = np.ones(len(P)) - list(map(sum, zip(*z_darts[:-1])))
gas_sat, water_sat, solid_sat = [], [], []
nu, x, z_c, density, pH, poro_diff, poro_diff2, poro_diff_tot, poro_diff_og, gas_list, gas_list2 = [], [], [], [], [], [], [], [], [], [], []
H2O, Ca, Na, Cl, OH, H, NaCO3, CO3, HCO3, Calcite, NaOH, H2CO3, K, CO2, Halite = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for i in range(len(P)):
    nu_output, x_output, z_c_output, density_output, pH_output, gas = n.flash_properties(z_darts[:, i], 320, P[i])  # itor
    if i == 0:
        print(z_c_output)
    # OH.append(z_c_output[1])
    # H.append(z_c_output[2])
    Na.append(z_c_output[2])
    Cl.append(z_c_output[3])
    CO2.append(z_c_output[1]-gas[1])
    # HCO3.append(z_c_output[5])
    H2O.append(z_c_output[0]-gas[0])
    # Ca.append(z_c_output[4])
    # CO3.append(z_c_output[5])
    # H2CO3.append(z_c_output[7])
    # NaCO3.append(z_c_output[8])
    # NaOH.append(z_c_output[9])
    # Calcite.append(z_c_output[7])
    # NaOH.append(z_c_output[5])
    Halite.append(z_c_output[4])
    nu.append(nu_output[1])
    x.append(x_output)
    z_c.append(z_c_output)
    pH.append(pH_output)
    # gas_sat.append(pH_output[0])
    # water_sat.append(pH_output[1])
    # solid_sat.append(pH_output[2])
    if nu_output[1] < 1e-11:  # if water phase does not exist
        density_output[1] = 1
    if nu_output[0] < 1e-11:
        density_output[0] = 1
    density.append(density_output)
    #print(nu_output)
    # # print(len(nu_output), nu_output, density_output)
    #poro_diff.append((poro[i] * (1-(nu_output[-1]*2156.560885608856*x_output[-1][-1])/np.sum(density_output)))-poro[i])
    #poro_diff2.append((poro[i] * (1-(nu_output[-1]*2712.4959349593496*x_output[-1][-1])/np.sum(density_output)))-poro[i])
    poro_diff_og.append((poro[i] * (1-(nu_output[-1]*density_output[-1])/np.sum(density_output)))-poro[i])
    gas_list.append(gas[0])
    gas_list2.append(gas[1])

# print(P)
# print(Halite)
print(nu)
plt.figure(1)
plt.plot(dx, P, label='pressure')
plt.xlabel('x [m]')
plt.ylabel('Pressure [bar]')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(dx, z_darts[0], label='H2O')
plt.plot(dx, z_darts[1], label='CO2')
plt.plot(dx, z_darts[2], label='Na+')
plt.plot(dx, z_darts[3], label='Cl-')
# plt.plot(z_darts[4], label='Ca+2')
# plt.plot(z_darts[5], label='CO3-2')
# plt.plot(z_darts[4], label='CO3-2')
plt.legend()
plt.ylabel('z_e')
# plt.yscale('log')
plt.xlabel('x dimensionless')
plt.title('Composition', y=1)
plt.show()

plt.figure(3)
plt.plot(dx, H2O, label='H2O')
plt.plot(gas_list, '--', label='H2O(g)')

# plt.plot(OH, label='OH-')
# plt.plot(H, label='H+')
plt.plot(dx, CO2, label='CO2')
plt.plot(dx, gas_list2, '--', label='CO2(g)')
# plt.plot(nu, ':', label='saturation')
plt.plot(dx, Na, label='Na+')
plt.plot(dx, Cl, label='Cl-')
# plt.plot(Ca, label='Ca+2')
# plt.plot(CO3, label='CO3-2')
# plt.plot(NaCO3, label='NaCO3')
# plt.plot(NaHCO3, label='NaHCO3-')
# plt.plot(NaOH, label='NaOH')
# plt.plot(K, label='K+')
plt.plot(dx, Halite, label='Halite')
# plt.plot(Calcite, label='Calcite')
# plt.yscale('log')
plt.legend()
# plt.legend()
plt.ylabel('z_c')
plt.xlabel('x dimensionless')
plt.title('Composition', y=1)
plt.show()

# plt.figure(4)
# plt.plot(dx, pH)
# plt.ylabel('SI')
# plt.xlabel('x dimensionless')
# plt.title('Saturation index', y=1)
# plt.show()

# plt.figure(4)
# plt.plot(dx, gas_sat, label='CO2(aq)')
# plt.plot(dx, water_sat, label='Na+')
# plt.plot(dx, solid_sat, label='Cl-')
# plt.legend()
# plt.ylabel('Molal')
# plt.xlabel('x dimensionless')
# plt.show()

plt.figure(5)
plt.plot(dx, nu)
plt.ylabel('Saturation')
#plt.ylim([0,1])
plt.xlabel('x dimensionless')
plt.title('Water saturation', y=1)
plt.show()

plt.figure(6)
# plt.plot(dx, poro_diff, label='Halite')
# plt.plot(dx, poro_diff2, label='Calcite')
plt.plot(poro_diff_og, label='Halite')
plt.ylabel('$\Delta$ porosity')
#plt.yscale('log')
#plt.ylim([0,1])
plt.xlabel('x dimensionless')
plt.title('$\Delta$ porosity', y=1)
plt.tight_layout()
plt.legend()
plt.show()