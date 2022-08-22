import darts.models.physics.chemical
import numpy as np
import pandas as pd

import sys
sys.path.append('C:/Users/Jaro/Documents/inversemodelling/code_thesis/DARTS_1D_model/physics_sup_2/')

from model_Reaktoro import Model
from darts.engines import value_vector, redirect_darts_output
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cProfile

redirect_darts_output('run.log')
n = Model()
n.init()
n.run_python(300, timestep_python=True)
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
poro = n.poro
P = Xn[0:ne*nb:ne]

z_darts = np.zeros((ne, len(P)))
for i in range(1, ne):
    z_darts[i-1] = Xn[i:ne*nb:ne]
z_darts[-1] = np.ones(len(P)) - list(map(sum, zip(*z_darts[:-1])))

nu, x, z_c, density = [], [], [], []
H2O, CO2, Na, Cl, Ca, CO3, Halite, Calcite = [], [], [], [], [], [], [], []
for i in range(len(P)):
    nu_output, x_output, z_c_output, density_output, kinetic_rate = n.flash_properties(z_darts[:, i], 350, P[i])  # itor
    H2O.append(z_c_output[0])
    CO2.append(z_c_output[1])
    Na.append(z_c_output[2])
    Cl.append(z_c_output[3])
    Ca.append(z_c_output[4])
    CO3.append(z_c_output[5])
    Halite.append(z_c_output[6])
    Calcite.append(z_c_output[7])
    nu.append(nu_output[1])
    x.append(x_output)
    z_c.append(z_c_output)
    density.append(density_output)

df = pd.DataFrame({'Pressure': P, 'Water sat': np.array(nu), 'Porosity': np.array(poro), 'H2O': np.array(H2O),
                   'CO2': np.array(CO2), 'Na+': np.array(Na), 'Cl-': np.array(Cl), 'Ca+2': np.array(Ca),
                   'CO3-2': np.array(CO3), 'Halite': np.array(Halite), 'Calcite': np.array(Calcite)})
df.to_csv('data.csv')
# df = pd.read_csv('data.csv')

# plt.figure(1)
# plt.plot(P, label='pressure')
# plt.legend()
# plt.show()
# plt.figure(2)
# plt.plot(H2O, label='H2O')
# plt.plot(CO2, label='CO2')
# plt.plot(Na, label='Na+')
# plt.plot(Cl, label='Cl-')
# plt.legend()
# plt.ylabel('z_e')
# plt.xlabel('x dimensionless')
# plt.title('Composition', y=1)
# plt.show()
# plt.figure(3)
# plt.plot(H2O, label='H2O')
# plt.plot(CO2, label='CO2')
# plt.plot(Na, label='Na+')
# plt.plot(Cl, label='Cl-')
# plt.plot(Halite, label='Halite')
# plt.legend()
# plt.ylabel('z_c')
# plt.xlabel('x dimensionless')
# plt.title('Composition', y=1)
# plt.show()
# plt.figure(4)
# plt.plot(nu)
# plt.ylabel('saturation')
# plt.xlabel('x dimensionless')
# plt.title('Water saturation', y=1)
# plt.show()
font_dict_title = {'family': 'sans-serif', 'color': 'black', 'weight': 'bold', 'size': 16, }
font_dict_axes = {'family': 'monospace', 'color': 'black', 'weight': 'normal', 'size': 14, }

extent = [0, 1, 0, 0.3]
fig, axs = plt.subplots(3, 1, figsize=(10, 7), dpi=400, facecolor='w', edgecolor='k')
im0 = axs[0].imshow(P.reshape(n.nz, n.nx), extent=extent)
axs[0].set_xlabel('x [-]', font_dict_axes)
axs[0].set_ylabel('y [-]', font_dict_axes)
axs[0].set_title('Pressure', fontdict=font_dict_title)
divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im0, cax=cax)

im1 = axs[1].imshow(np.array(nu).reshape(n.nz, n.nx), extent=extent)
axs[1].set_xlabel('x [-]', font_dict_axes)
axs[1].set_ylabel('y [-]', font_dict_axes)
axs[1].set_title('Water saturation', fontdict=font_dict_title)
divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)

im2 = axs[2].imshow(np.array(poro).reshape(n.nz, n.nx), extent=extent)
axs[2].set_xlabel('x [-]', font_dict_axes)
axs[2].set_ylabel('y [-]', font_dict_axes)
axs[2].set_title('Porosity', fontdict=font_dict_title)
divider = make_axes_locatable(axs[2])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax)
plt.show()


fig2, axs2 = plt.subplots(3, 2, figsize=(10, 7), dpi=400, facecolor='w', edgecolor='k')
im0 = axs2[0][0].imshow(np.array(H2O).reshape(n.nz, n.nx), extent=extent)
axs2[0][0].set_xlabel('x [-]', font_dict_axes)
axs2[0][0].set_ylabel('y [-]', font_dict_axes)
axs2[0][0].set_title('H2O', fontdict=font_dict_title)
divider = make_axes_locatable(axs2[0][0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im0, cax=cax)

im1 = axs2[0][1].imshow(np.array(CO2).reshape(n.nz, n.nx), extent=extent)
axs2[0][1].set_xlabel('x [-]', font_dict_axes)
axs2[0][1].set_ylabel('y [-]', font_dict_axes)
axs2[0][1].set_title('CO2', fontdict=font_dict_title)
divider = make_axes_locatable(axs2[0][1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)

im2 = axs2[1][0].imshow(np.array(Na).reshape(n.nz, n.nx), extent=extent)
axs2[1][0].set_xlabel('x [-]', font_dict_axes)
axs2[1][0].set_ylabel('y [-]', font_dict_axes)
axs2[1][0].set_title('Na+', fontdict=font_dict_title)
divider = make_axes_locatable(axs2[1][0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax)

im3 = axs2[1][1].imshow(np.array(Halite).reshape(n.nz, n.nx), extent=extent)
axs2[1][1].set_xlabel('x [-]', font_dict_axes)
axs2[1][1].set_ylabel('y [-]', font_dict_axes)
axs2[1][1].set_title('Halite', fontdict=font_dict_title)
divider = make_axes_locatable(axs2[1][1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im3, cax=cax)

im4 = axs2[2][0].imshow(np.array(Ca).reshape(n.nz, n.nx), extent=extent)
axs2[2][0].set_xlabel('x [-]', font_dict_axes)
axs2[2][0].set_ylabel('y [-]', font_dict_axes)
axs2[2][0].set_title('Ca+2', fontdict=font_dict_title)
divider = make_axes_locatable(axs2[2][0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im4, cax=cax)

im5 = axs2[2][1].imshow(np.array(Calcite).reshape(n.nz, n.nx), extent=extent)
axs2[2][1].set_xlabel('x [-]', font_dict_axes)
axs2[2][1].set_ylabel('y [-]', font_dict_axes)
axs2[2][1].set_title('Calcite', fontdict=font_dict_title)
divider = make_axes_locatable(axs2[2][1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im5, cax=cax)

plt.tight_layout()
plt.show()
