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

# redirect_darts_output('run.log')
n = Model()
# n.init()
# n.run_python(1000, timestep_python=True)

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
#saturation = n.property_container.sat

P = Xn[0::nc-2]
z1_darts = Xn[1::nc-2]
z2_darts = Xn[2::nc-2]
z3_darts = Xn[3::nc-2]
z4_darts = Xn[4::nc-2]
z5_darts = Xn[5::nc-2]
z6_darts = np.zeros(len(z1_darts))
for i in range(len(z1_darts)):
    z6_darts[i] = 1 - z1_darts[i]-z2_darts[i]-z3_darts[i]-z4_darts[i]-z5_darts[i]
# P = Xn[0:n.reservoir.nb*nc:nc]
# z1_darts = Xn[1:n.reservoir.nb*nc:nc]
# z2_darts = Xn[2:n.reservoir.nb*nc:nc]
# z3_darts = Xn[3:n.reservoir.nb*nc:nc]
# z4_darts = Xn[2:n.reservoir.nb*nc:nc]
# z5_darts = np.zeros(len(z2_darts))
# for i in range(len(z2_darts)):
#     z5_darts[i] = 1 - z1_darts[i] - z2_darts[i] - z3_darts[i] - z4_darts[i]

plt.figure(1)
plt.plot(P, label='pressure')
plt.legend()
plt.show()
plt.figure(2)
plt.plot(z1_darts, label='H2O')
plt.plot(z2_darts, label='CO2')
plt.plot(z3_darts,label='Ca++')
plt.plot(z4_darts,label='CO3--')
plt.plot(z5_darts,label='Na+')
plt.plot(z6_darts,label='Cl-')
plt.legend()
plt.ylabel('z_e')
plt.xlabel('x dimensionless')
plt.title('Composition', y=1)
#
# # plt.subplot(212)
# # plt.plot(P)
# # plt.title('Pressure', y=1)
#
#
plt.show()
# plt.plot(saturation)
# plt.show()