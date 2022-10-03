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
n.run_python(25000, timestep_python=True)
n.print_timers()
n.print_stat()
time_data = pd.DataFrame.from_dict(n.physics.engine.time_data)
time_data.to_pickle("darts_time_data.pkl")
n.save_restart_data()
writer = pd.ExcelWriter('time_data.xlsx')
time_data.to_excel(writer, 'Sheet1')
writer.save()

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

""" plot results 2D """
Xn = np.array(n.physics.engine.X, copy=False)
nc = n.property_container.nc + n.thermal
ne = n.property_container.n_e
nb = n.reservoir.nb
log_flag = n.property_container.log_flag
P = Xn[0:ne*nb:ne]
poro = n.poro
z_darts = np.zeros((ne, len(P)))
for i in range(1, ne):
    if log_flag == 1:
        z_darts[i-1] = np.exp(Xn[i:ne*nb:ne])
    else:
        z_darts[i - 1] = Xn[i:ne * nb:ne]
z_darts[-1] = np.ones(len(P)) - list(map(sum, zip(*z_darts[:-1])))
nu, x, z_c, density, pH, poro_diff = [], [], [], [], [], []
H2O, Ca, Na, Cl, OH, H, NaCO3, CO3, HCO3, NaHCO3, NaOH, H2CO3, K, CO2, Halite = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for i in range(len(P)):
    nu_output, x_output, z_c_output, density_output, pH_output = n.flash_properties(z_darts[:, i], 320, P[i])  # itor
    # CO3.append(z_c_output[0])
    # OH.append(z_c_output[1])
    # H.append(z_c_output[2])
    Na.append(z_c_output[2])
    Cl.append(z_c_output[3])
    CO2.append(z_c_output[1])
    # HCO3.append(z_c_output[5])
    H2O.append(z_c_output[0])
    # K.append(z_c_output[5])
    # H2CO3.append(z_c_output[7])
    # NaCO3.append(z_c_output[8])
    # NaOH.append(z_c_output[9])
    # NaHCO3.append(z_c_output[10])
    # NaOH.append(z_c_output[5])
    Halite.append(z_c_output[4])
    nu.append(nu_output[1])
    x.append(x_output)
    z_c.append(z_c_output)
    density.append(density_output)
    pH.append(pH_output)
    poro_diff.append((poro[i] * (1-(nu_output[-1]*density_output[-1])/np.sum(density_output)))-poro[i])

plt.figure(1)
plt.plot(P, label='pressure')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(z_darts[0], label='H2O')
plt.plot(z_darts[1], label='CO2')
plt.plot(z_darts[2], label='Ca+2')
plt.plot(z_darts[3], label='CO3-2')
# plt.plot(z_darts[4], label='CO3-2')
plt.legend()
plt.ylabel('z_e')
# plt.yscale('log')
plt.xlabel('x dimensionless')
plt.title('Composition', y=1)
plt.show()

plt.figure(3)
plt.plot(H2O, label='H2O')
# plt.plot(OH, label='OH-')
# plt.plot(H, label='H+')
plt.plot(CO2, label='CO2')
plt.plot(Na, label='Ca+2')
plt.plot(Cl, label='CO3-2')
# plt.plot(HCO3, label='HCO3-')
# plt.plot(H2CO3, label='H2CO3')
# plt.plot(NaCO3, label='NaCO3')
# plt.plot(NaHCO3, label='NaHCO3-')
# plt.plot(NaOH, label='NaOH')
# plt.plot(K, label='K+')
plt.plot(Halite, label='Calcite')
#plt.yscale('log')
plt.legend(loc=4)
# plt.legend()
plt.ylabel('z_c')
plt.xlabel('x dimensionless')
plt.title('Composition', y=1)
plt.show()

# plt.figure(4)
# plt.plot(pH)
# plt.ylabel('pH')
# plt.xlabel('x dimensionless')
# plt.title('pH', y=1)
# plt.show()

plt.figure(4)
plt.plot(nu)
plt.ylabel('Saturation')
#plt.ylim([0,1])
plt.xlabel('x dimensionless')
plt.title('Water saturation', y=1)
plt.show()

plt.figure(5)
plt.plot(poro_diff)
plt.ylabel('$\Delta$ porosity')
#plt.ylim([0,1])
plt.xlabel('x dimensionless')
plt.title('$\Delta$ porosity', y=1)
plt.tight_layout()
plt.show()
