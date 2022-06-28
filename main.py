import numpy as np
import pandas as pd

import sys
sys.path.append('C:/Users/Jaro/Documents/inversemodelling/code_thesis/DARTS_1D_model/physics_sup_2/')

from model_Reaktoro import Model
from darts.engines import value_vector, redirect_darts_output
import matplotlib.pyplot as plt
from matplotlib import cm
import os
#import cProfile

redirect_darts_output('run.log')
n = Model()  # 1220s, 1140 point generation
n.init()
n.run_python(400, timestep_python=True)
n.print_timers()
n.print_stat()
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

""" plot results 2D """
Xn = np.array(n.physics.engine.X, copy=False)
nc = n.property_container.nc + n.thermal
ne = n.property_container.n_e
nb = n.reservoir.nb
P = Xn[0:ne*nb:ne]
z1_darts = Xn[1:ne*nb:ne]
z2_darts = Xn[2:ne*nb:ne]
z3_darts = Xn[3:ne*nb:nc-2]
z4_darts = Xn[4:ne*nb:nc-2]
z5_darts = Xn[5:ne*nb:nc-2]
z6_darts = np.zeros(len(z1_darts))
for i in range(len(z1_darts)):
    z6_darts[i] = 1 - z1_darts[i]-z2_darts[i] - z3_darts[i]-z4_darts[i]-z5_darts[i]
# z_e = [z1_darts, z2_darts, z3_darts, z4_darts, z5_darts, z6_darts]
# print(z_e)
# print(z_e[1])
# nu, x, z_c, density = np.zeros(nb), np.zeros(nb), np.zeros(nb), np.zeros(nb)
nu, x, z_c, density = [], [], [], []
H2O, CO2, Ca, CO3, Na, Cl, Calcite, Halite = [], [], [], [], [], [], [], []
for i in range(len(P)):
    z_e = [float(z1_darts[i]), float(z2_darts[i]), float(z3_darts[i]), float(z4_darts[i]), float(z5_darts[i]), float(z6_darts[i])]
    nu_output, x_output, z_c_output, density_output = n.flash_properties(z_e, 320, P[i]) # itor
    H2O.append(z_c_output[0])
    CO2.append(z_c_output[1])
    Ca.append(z_c_output[2])
    CO3.append(z_c_output[3])
    Na.append(z_c_output[4])
    Cl.append(z_c_output[5])
    Calcite.append(z_c_output[6])
    Halite.append(z_c_output[7])
    nu.append(nu_output[1])
    x.append(x_output)
    z_c.append(z_c_output)
    density.append(density_output)

plt.figure(1)
plt.plot(P, label='pressure')
plt.legend()
plt.show()
plt.figure(2)
plt.plot(z1_darts, label='H2O')
plt.plot(z2_darts, label='CO2')
plt.plot(z3_darts, label='Ca++')
plt.plot(z4_darts, label='CO3--')
plt.plot(z5_darts, label='Na+')
plt.plot(z6_darts, label='Cl-')
plt.legend()
plt.ylabel('z_e')
plt.xlabel('x dimensionless')
plt.title('Composition', y=1)
plt.show()
plt.figure(3)
plt.plot(H2O, label='H2O')
plt.plot(CO2, label='CO2')
plt.plot(Ca, label='Ca++')
plt.plot(CO3, label='CO3--')
plt.plot(Na, label='Na+')
plt.plot(Cl, label='Cl-')
plt.plot(Calcite, label='Calcite')
plt.plot(Halite, label='Halite')
plt.legend()
plt.ylabel('z_c')
plt.xlabel('x dimensionless')
plt.title('Composition', y=1)
plt.show()
plt.figure(4)
plt.plot(nu)
plt.ylabel('saturation')
plt.xlabel('x dimensionless')
plt.title('Water saturation', y=1)
plt.show()