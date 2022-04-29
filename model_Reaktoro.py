from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import sim_params, value_vector
import numpy as np
from properties_basic import *
from property_container import *
from reaktoro import *
from physics_comp_sup import Compositional

import matplotlib.pyplot as plt

# TODO: properties_basic: Flash line 61
#       properties_basic: Kinetic line 265

# Model class creation here!
class Model(DartsModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.zero = 1e-11
        init_ions = 0.5
        solid_init = 0.7
        equi_prod = (init_ions / 2) ** 2
        solid_inject = self.zero
        trans_exp = 3
        perm = 100 #/ (1 - solid_init) ** trans_exp
        """Reservoir"""
        nx = 1000
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=1, dy=1, dz=1, permx=perm, permy=perm,
                                         permz=perm/10, poro=1, depth=1000)


        """well location"""
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1, j=1, k=1, multi_segment=False)

        self.reservoir.add_well("P1")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=nx, j=1, k=1, multi_segment=False)

        """Physical properties"""
        # Create property containers:
        components_name = ['H2O', 'CO2', 'Ca++', 'CO3--', 'CaCO3']
        elements_name = ['H2O', 'CO2', 'Ca++', 'CO3--']
        aq_phase = ['H2O(l)', 'CO2(aq)', 'Ca++', 'CO3--', 'CaCO3(aq)']
        gas_phase = ['H2O(g)', 'CO2(g)']
        mineral_phase = ['Calcite']
        self.thermal = 0
        Mw = [18.015, 44.01, 40.078, 60.008, 100.086]
        self.property_container = model_properties(phases_name=['gas', 'wat', 'sol'],
                                                   components_name=components_name, elements_name=elements_name,
                                                   diff_coef=1e-9, rock_comp=1e-7,
                                                   Mw=Mw, min_z=self.zero / 10, solid_dens=[2000])
        self.components = self.property_container.components_name
        self.elements = self.property_container.elements_name
        self.phases = self.property_container.phases_name

        """ properties correlations """
        #self.property_container.flash_ev = Flash(self.components[:-1], [10, 1e-12, 1e-1], self.zero) #return [y, x], [V, 1-V]
        self.property_container.density_ev = dict([('gas', Density(compr=1e-4, dens0=100)),
                                                   ('wat', Density(compr=1e-6, dens0=1000)),
                                                   ('sol', Density(compr=0, dens0=2700))])
        self.property_container.viscosity_ev = dict([('gas', ViscosityConst(0.1)),
                                                     ('wat', ViscosityConst(1)),
                                                     ('sol', ViscosityConst(1))])
        self.property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                                    ('wat', PhaseRelPerm("wat")),
                                                    ('sol', PhaseRelPerm('sol'))])


        ne = self.property_container.nc + self.thermal
        self.property_container.kinetic_rate_ev = kinetic_basic(equi_prod, 1e-0, ne)

        """ Activate physics """
        self.physics = Compositional(self.property_container, self.timer, n_points=101, min_p=1, max_p=1000,
                                     min_z=self.zero/10, max_z=1-self.zero/10)

        # zc_fl_init = [self.zero / (1 - solid_init), init_ions]
        # zc_fl_init = zc_fl_init + [1 - sum(zc_fl_init)]
        # self.ini_stream = [x * (1 - solid_init) for x in zc_fl_init]
        # zc_fl_inj_stream_gas = [1 - 2 * self.zero / (1 - solid_inject), self.zero / (1 - solid_inject)]
        # zc_fl_inj_stream_gas = zc_fl_inj_stream_gas + [1 - sum(zc_fl_inj_stream_gas)]
        # self.inj_stream = [x * (1 - solid_inject) for x in zc_fl_inj_stream_gas]
        # self.inj_stream = [self.zero,0.99,self.zero,self.zero,self.zero]

        self.inj_stream = [0.99, self.zero, self.zero, self.zero, self.zero]
        self.ini_stream = [0.9, self.zero, 0.03, 0.03, 0.03]

        # Some newton parameters for non-linear solution:
        self.params.first_ts = 0.001
        self.params.max_ts = 1
        self.params.mult_ts = 2

        self.params.tolerance_newton = 1e-5
        self.params.tolerance_linear = 1e-6
        self.params.max_i_newton = 10
        self.params.max_i_linear = 50
        self.params.newton_type = sim_params.newton_local_chop
        # self.params.newton_params[0] = 0.2

        self.runtime = 1000

        self.timer.node["initialization"].stop()

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        """ initialize conditions for all scenarios"""
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, 95, self.ini_stream)

        # composition = np.array(self.reservoir.mesh.composition, copy=False)
        # n_half = int(self.reservoir.nx * self.reservoir.ny * self.reservoir.nz / 2)
        # composition[2*n_half:] = 1e-6

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                w.control = self.physics.new_rate_inj(0.2, self.inj_stream, 0)
                #w.control = self.physics.new_bhp_inj(150, self.inj_stream)
            else:
                w.control = self.physics.new_bhp_prod(50)

    def set_op_list(self):
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks
        self.op_num[n_res:] = 1
        self.op_list = [self.physics.acc_flux_itor, self.physics.acc_flux_w_itor]

    def properties(self, state):

        (sat, x, rho, rho_m, mu, kr, ph) = self.property_container.evaluate(state)

        return sat[0]

    def print_and_plot(self, filename):

        nc = self.property_container.nc
        Sg = np.zeros(self.reservoir.nb)
        Ss = np.zeros(self.reservoir.nb)
        X = np.zeros((self.reservoir.nb, nc - 1, 2))

        rel_perm = np.zeros((self.reservoir.nb, 2))
        visc = np.zeros((self.reservoir.nb, 2))
        density = np.zeros((self.reservoir.nb, 3))
        density_m = np.zeros((self.reservoir.nb, 3))

        Xn = np.array(self.physics.engine.X, copy=True)


        P = Xn[0:self.reservoir.nb * nc:nc]
        z_caco3 = 1 - (Xn[1:self.reservoir.nb * nc:nc] + Xn[2:self.reservoir.nb * nc:nc] + Xn[3:self.reservoir.nb * nc:nc])

        z_co2 = Xn[1:self.reservoir.nb * nc:nc] / (1 - z_caco3)
        z_inert = Xn[2:self.reservoir.nb * nc:nc] / (1 - z_caco3)
        z_h2o = Xn[3:self.reservoir.nb * nc:nc] / (1 - z_caco3)

        for ii in range(self.reservoir.nb):
            x_list = Xn[ii*nc:(ii+1)*nc]
            state = value_vector(x_list)
            (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property_container.evaluate(state)

            rel_perm[ii, :] = kr
            visc[ii, :] = mu
            density[ii, :2] = rho
            density_m[ii, :2] = rho_m

            density[2] = self.property_container.solid_dens[-1]

            X[ii, :, 0] = x[1][:-1]
            X[ii, :, 1] = x[0][:-1]
            Sg[ii] = sat[0]
            Ss[ii] = z_caco3[ii]

        # Write all output to a file:
        with open(filename, 'w+') as f:
            # Print headers:
            print('//Gridblock\t gas_sat\t pressure\t C_m\t poro\t co2_liq\t co2_vap\t h2o_liq\t h2o_vap\t ca_plus_co3_liq\t liq_dens\t vap_dens\t solid_dens\t liq_mole_dens\t vap_mole_dens\t solid_mole_dens\t rel_perm_liq\t rel_perm_gas\t visc_liq\t visc_gas', file=f)
            print('//[-]\t [-]\t [bar]\t [kmole/m3]\t [-]\t [-]\t [-]\t [-]\t [-]\t [-]\t [kg/m3]\t [kg/m3]\t [kg/m3]\t [kmole/m3]\t [kmole/m3]\t [kmole/m3]\t [-]\t [-]\t [cP]\t [cP]', file=f)
            for ii in range (self.reservoir.nb):
                print('{:d}\t {:6.5f}\t {:7.5f}\t {:7.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:8.5f}\t {:8.5f}\t {:8.5f}\t {:7.5f}\t {:7.5f}\t {:7.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}'.format(
                    ii, Sg[ii], P[ii], Ss[ii] * density_m[ii, 2], 1 - Ss[ii], X[ii, 0, 0], X[ii, 0, 1], X[ii, 2, 0], X[ii, 2, 1], X[ii, 1, 0],
                    density[ii, 0], density[ii, 1], density[ii, 2], density_m[ii, 0], density_m[ii, 1], density_m[ii, 2],
                    rel_perm[ii, 0], rel_perm[ii, 1], visc[ii, 0], visc[ii, 1]), file=f)

        """ start plots """

        font_dict_title = {'family': 'sans-serif',
                           'color': 'black',
                           'weight': 'normal',
                           'size': 14,
                           }

        font_dict_axes = {'family': 'monospace',
                          'color': 'black',
                          'weight': 'normal',
                          'size': 14,
                          }

        fig, axs = plt.subplots(3, 3, figsize=(12, 10), dpi=200, facecolor='w', edgecolor='k')
        """ sg and x """
        axs[0][0].plot(z_co2, 'b')
        axs[0][0].set_xlabel('x [m]', font_dict_axes)
        axs[0][0].set_ylabel('$z_{CO_2}$ [-]', font_dict_axes)
        axs[0][0].set_title('Fluid composition', fontdict=font_dict_title)

        axs[0][1].plot(z_h2o, 'b')
        axs[0][1].set_xlabel('x [m]', font_dict_axes)
        axs[0][1].set_ylabel('$z_{H_2O}$ [-]', font_dict_axes)
        axs[0][1].set_title('Fluid composition', fontdict=font_dict_title)

        axs[0][2].plot(z_inert, 'b')
        axs[0][2].set_xlabel('x [m]', font_dict_axes)
        axs[0][2].set_ylabel('$z_{w, Ca+2} + z_{w, CO_3-2}$ [-]', font_dict_axes)
        axs[0][2].set_title('Fluid composition', fontdict=font_dict_title)

        axs[1][0].plot(X[:, 0, 0], 'b')
        axs[1][0].set_xlabel('x [m]', font_dict_axes)
        axs[1][0].set_ylabel('$x_{w, CO_2}$ [-]', font_dict_axes)
        axs[1][0].set_title('Liquid mole fraction', fontdict=font_dict_title)

        axs[1][1].plot(X[:, 2, 0], 'b')
        axs[1][1].set_xlabel('x [m]', font_dict_axes)
        axs[1][1].set_ylabel('$x_{w, H_2O}$ [-]', font_dict_axes)
        axs[1][1].set_title('Liquid mole fraction', fontdict=font_dict_title)

        axs[1][2].plot(X[:, 1, 0], 'b')
        axs[1][2].set_xlabel('x [m]', font_dict_axes)
        axs[1][2].set_ylabel('$x_{w, Ca+2} + x_{w, CO_3-2}$ [-]', font_dict_axes)
        axs[1][2].set_title('Liquid mole fraction', fontdict=font_dict_title)

        axs[2][0].plot(P, 'b')
        axs[2][0].set_xlabel('x [m]', font_dict_axes)
        axs[2][0].set_ylabel('$p$ [bar]', font_dict_axes)
        axs[2][0].set_title('Pressure', fontdict=font_dict_title)

        axs[2][1].plot(Sg, 'b')
        axs[2][1].set_xlabel('x [m]', font_dict_axes)
        axs[2][1].set_ylabel('$s_g$ [-]', font_dict_axes)
        axs[2][1].set_title('Gas saturation', fontdict=font_dict_title)

        axs[2][2].plot(1 - Ss, 'b')
        axs[2][2].set_xlabel('x [m]', font_dict_axes)
        axs[2][2].set_ylabel('$\phi$ [-]', font_dict_axes)
        axs[2][2].set_title('Porosity', fontdict=font_dict_title)

        left = 0.05  # the left side of the subplots of the figure
        right = 0.95  # the right side of the subplots of the figure
        bottom = 0.05  # the bottom of the subplots of the figure
        top = 0.95  # the top of the subplots of the figure
        wspace = 0.25  # the amount of width reserved for blank space between subplots
        hspace = 0.25  # the amount of height reserved for white space between subplots
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        for ii in range(3):
            for jj in range(3):
                for tick in axs[ii][jj].xaxis.get_major_ticks():
                    tick.label.set_fontsize(20)

                for tick in axs[ii][jj].yaxis.get_major_ticks():
                    tick.label.set_fontsize(20)

        plt.tight_layout()
        plt.savefig("results_kinetic_brief.pdf")
        plt.show()



class model_properties(property_container):
    def __init__(self, phases_name, components_name, elements_name, Mw, min_z=1e-11,
                 diff_coef=0, rock_comp=1e-6, solid_dens=[]):
        # Call base class constructor
        super().__init__(phases_name, components_name, elements_name, Mw, min_z=min_z, diff_coef=diff_coef,
                         rock_comp=rock_comp, solid_dens=solid_dens)
        # Check: here change density from Reaktoro

    def run_density(self, pressure, zc):
        nu,x,y,zc,density = Flash_Reaktoro(zc, 320, pressure*100000, self.elements_name, self.min_z)
        return density

        # introduce temperature

    def run_flash(self, pressure, ze):  # Change this to include 3 phases
        # Make every value that is the min_z equal to 0, as Reaktoro can work with 0, but not transport
        for i in range(len(ze)):
            if ze[i] < self.min_z:
                ze[i] = 0
                ze = [float(i) / sum(ze) for i in ze]
        nu, x, y, zc, density = Flash_Reaktoro(ze, 320, pressure*100000, self.elements_name, self.min_z)
        # z_c Output of reaktoro has no values less than z_c and has been renormalised
        self.x = x
        self.nu = nu
        #print('nu',nu)
        '''
        Reaktoro input z_e, will output the components. Input also is the temp in K and pressure in Pa
        Output is the phase fractions (L,V,S, gotten from volume occupied), liq and vapor fraction (x,y)
        z_c (components) and density (kg/m3)
        
        ph gives the phases that are inside the cell 0 - gas, 1 - liquid, 2- solid
        Check whether there is a solid phase first, then for other phases
        '''
        # check with self.zero instead 0
        # Check if S<minimum, otherwise, set it to minimum value
        # Solid phase always present

        ph = list(range(3))  # ph = range(number of total phases)

        for i in range(len(self.nu)):
            if i > 2 and self.nu[i] < self.min_z:
                self.nu[i] = self.min_z
                self.nu = [float(i) / sum(self.nu) for i in self.nu]

        if self.nu[0] < self.min_z:     # if vapor phase is less than min_z, it does not exist
            del ph[0]                   # remove entry of vap
            self.nu[0] = 0
            self.nu = [float(i) / sum(self.nu) for i in self.nu]

        elif self.nu[1] < self.min_z:   # if liq phase is less than min_z, it does not exist
            del ph[1]
            self.nu[1] = 0
            self.nu = [float(i) / sum(self.nu) for i in self.nu]
        # Check: this is only for negative flash
        # for i in range(self.nc - 1):
        #     for j in range(2):
        #         self.x[j][i] = xr[j][i]

        # self.nu[0] = V
        # self.nu[1] = L
        # self.nu[2] = S
        #print('nu after corr', self.nu)
        return ph


def Flash_Reaktoro(z_e, T, P, elem, min_z):
    m = Reaktoro()
    m.addingproblem(T, P, z_e, elem)
    nu, x, y, z_c, density = m.output(min_z)  # this outputs comp h20, co2, ca, co3, caco3
    return nu, x, y, z_c, density

class Reaktoro():
    def __init__(self):
        editor = ChemicalEditor(Database('supcrt98.xml'))              # Database that Reaktoro uses
        editor.addAqueousPhase("H2O(l) CO2(aq) Ca++ CO3-- CaCO3(aq)")  # Aqueous phase
        editor.addGaseousPhase('H2O(g) CO2(g)')
        editor.addMineralPhase('Calcite')

        self.system = ChemicalSystem(editor)        # Set the system
        self.solver = EquilibriumSolver(self.system)    # solves the eq system

        self.properties = ChemicalProperties(self.system)

        self.states = []  # In case multiple adding problem are used, they are stored within a list, for easy output

    def addingproblem(self, temp, pres, z_e, elem):
        problem = EquilibriumProblem(self.system)
        problem.setTemperature(temp, 'kelvin')
        problem.setPressure(pres, 'pascal')
        for i in range(len(elem)):
            problem.add(str(elem[i]), z_e[i], 'mol')  # Add all components/elements

        state = ChemicalState(self.system)
        self.solver.solve(state, problem)
        self.properties.update(temp, pres, z_e)
        self.states.append(state)           # append the solved eq state to states

    def output_clean(self):
        density = self.properties.phaseDensities().val
        mole_frac_phases = self.properties.moleFractions().val
        phase_vol = self.properties.phaseVolumes().val
        total_vol = self.properties.volume().val

    def output(self, min_z):
        n_states = len(self.states)
        volume_tot = [ChemicalQuantity(state).value('volume(units=m3)') for state in self.states]  # m3

        mass_aq = [ChemicalQuantity(state).value("phaseMass(Aqueous)") for state in self.states]  # kg
        volume_aq = [ChemicalQuantity(state).value("phaseVolume(Aqueous)") for state in self.states]  # m3

        density_aq = np.zeros(n_states)
        for i in range(n_states):
            density_aq[i] = mass_aq[i]/volume_aq[i]
        mol_total_aq = [ChemicalQuantity(state).value("phaseAmount(Aqueous)") for state in self.states]  # mol
        H2O = [state.speciesAmount("H2O(l)") for state in self.states]
        CO2 = [state.speciesAmount("CO2(aq)") for state in self.states]
        CaCO3 = [state.speciesAmount("CaCO3(aq)") for state in self.states]
        Ca = [state.speciesAmount("Ca++") for state in self.states]
        CO3 = [state.speciesAmount("CO3--") for state in self.states]

        mass_gas = [ChemicalQuantity(state).value("phaseMass(Gaseous)") for state in self.states]  # kg
        volume_gas = [ChemicalQuantity(state).value("phaseVolume(Gaseous)") for state in self.states]  # m3
        density_gas = np.zeros(n_states)
        for i in range(n_states):
            density_gas[i] = mass_gas[i]/volume_gas[i]
        mol_total_gas = [ChemicalQuantity(state).value("phaseAmount(Gaseous)") for state in self.states]  # mol
        H2O_gas = [state.speciesAmount("H2O(g)") for state in self.states]
        CO2_gas = [state.speciesAmount("CO2(g)") for state in self.states]

        mass_solid = [ChemicalQuantity(state).value("phaseMass(Calcite)") for state in self.states]  # kg
        volume_solid = [ChemicalQuantity(state).value("phaseVolume(Calcite)") for state in self.states]  # m3
        density_solid = np.zeros(n_states)
        for i in range(n_states):
            density_solid[i] = mass_solid[i]/volume_solid[i]
        mol_total_solid = [ChemicalQuantity(state).value("phaseAmount(Calcite)") for state in self.states]  # mol
        Calcite = [state.speciesAmount("Calcite") for state in self.states]

        S_w, S_g, S_s = np.zeros(n_states), np.zeros(n_states), np.zeros(n_states)
        L, V, S = np.zeros(n_states), np.zeros(n_states), np.zeros(n_states)
        mol_total = np.zeros(n_states)
        z_c, x, y, x_calcite, density = [], [], [], [], []
        for i in range(n_states):
            if volume_aq[i] < 0:
                print('######################NEGATIVE AQUEOUS VOLUME WARNING##########################################')
            mol_total[i] = mol_total_aq[i] + mol_total_gas[i] + mol_total_solid[i]
            zc = [(H2O[i]+H2O_gas[i])/mol_total[i], (CO2[i]+CO2_gas[i])/mol_total[i], Ca[i]/mol_total[i], CO3[i]/mol_total[i], (CaCO3[i]+Calcite[i])/mol_total[i]]
            for q in range(len(zc)):
                if zc[q] < min_z:
                    zc[q] = min_z
                    zc = [float(q) / sum(zc) for q in zc]
            S_w[i] = volume_aq[i] / volume_tot[i]
            S_g[i] = volume_gas[i] / volume_tot[i]
            S_s[i] = volume_solid[i] / volume_tot[i]  # * density_solid = solid mass
            L[i] = (density_aq[i] * S_w[i]) / (density_gas[i] * S_g[i] + density_aq[i] * S_w[i] + density_solid[i] * S_s[i])
            V[i] = (density_gas[i] * S_g[i]) / (density_gas[i] * S_g[i] + density_aq[i] * S_w[i] + density_solid[i] * S_s[i])
            S[i] = (density_solid[i] * S_s[i]) / (density_gas[i] * S_g[i] + density_aq[i] * S_w[i] + density_solid[i] * S_s[i])
            x = np.append(x, [H2O[i]/mol_total_aq[i], CO2[i]/mol_total_aq[i], Ca[i]/mol_total_aq[i], CO3[i]/mol_total_aq[i], CaCO3[i]/mol_total_aq[i]])
            y = np.append(y, [H2O_gas[i]/mol_total_gas[i], CO2_gas[i]/mol_total_gas[i], 0, 0, 0])
            x_calcite = np.append(x_calcite, [0, 0, 0, 0, 1])
            z_c = np.append(z_c, zc)
            density = np.append(density, [density_gas, density_aq, density_solid])
        nu = [V, L, S]
        x = [y, x, x_calcite]
        return nu, x, y, z_c, density


# def Flash_Reaktoro_kin(z_beta, T, P):
#     m = Reaktoro()
#     m.addingproblem(m.T, P, z_beta)
#     #amount_elements = m.states[0].elementAmounts()  # element amounts in alphabet order: C, Ca, H, O
#     #m.solver.solve(m.states[0], m.T, m.p, amount_elements)
#     L, V, S, x, y, z_c, density, kinetic_rate = m.output()  # this outputs comp h20, co2, ca, co3, caco3
#     return L, x, y, z_c, density, kinetic_rate
#
# class Reaktoro(Variables):
#     def __init__(self):
#         super().__init__()
#         editor = ChemicalEditor(Database('supcrt98.xml'))              # Database that Reaktoro uses
#         editor.addAqueousPhase("H2O(l) CO2(aq) Ca++ CO3-- CaCO3(aq)")  # Aqueous phase with elem
#         editor.addGaseousPhase('H2O(g) CO2(g)')
#         editor.addMineralPhase('Calcite')
#
#         # Kinetics
#         editor.addMineralReaction("Calcite") \
#             .setEquation("Ca++ + CO3-- = Calcite") \
#             .addMechanism("logk = -5.81 mol/(m2*s); Ea = 23.5 kJ/mol") \
#             .setSpecificSurfaceArea(10, "cm2/g")
#
#         self.system = ChemicalSystem(editor)        # Set the system
#         self.reactions = ReactionSystem(editor)     # Set reaction system for kinetics
#
#         self.solver = EquilibriumSolver(self.system)    # solves the eq system
#         self.kinetic_solver = KineticSolver(self.reactions)
#
#         self.partition = Partition(self.system)  # Partition the system into equilibrium and kinetic reactions
#         self.partition.setKineticSpecies(["Calcite"])  # Set which species needs to be defined kinetically
#         self.kinetic_solver.setPartition(self.partition)
#         self.states = []
#         # Two version can be coded with the same solution. One uses the EquilibriumProblem function,
#         # Other will use the state function. Both need to output a state in order to be solved by EquilibriumSolver
#
#     def addingproblem(self, temp, pres, z_e):
#         self.problem = EquilibriumProblem(self.system)
#         self.problem.setPartition(self.partition)
#         self.problem.setTemperature(temp, 'kelvin')
#         self.problem.setPressure(pres, 'pascal')
#         self.problem.add('H2O', z_e[0], 'mol')
#         self.problem.add('CO2', z_e[1], 'mol')
#         self.problem.add('Ca++', z_e[2], 'mol')
#         self.problem.add('CO3--', z_e[3], 'mol')
#         #self.problem.add('Calcite', z_e[4], 'mol')
#         t0, t1 = 0, 60
#
#         self.state = ChemicalState(self.system)
#         #self.state = equilibrate(self.problem)  # Equilibrate the problem in order to write to state
#         self.solver.solve(self.state, self.problem)
#         self.state.scaleVolume(1.0, "m3")
#         self.state.setSpeciesAmount('Calcite', z_e[4], 'mol')
#         self.kinetic_solver.solve(self.state, t0, t1)  # t: start time [sec], dt: step to be used from t to t + dt[sec]
#         self.rate = (ChemicalQuantity(self.state).value('phaseAmount(Calcite)') - z_e[4]) / (t1-t0)
#         self.states.append(self.state)
#
#
#     def addingstates(self, temp, pres, z_e):
#         self.state = ChemicalState(self.system)
#         self.state.setTemperature(temp, 'kelvin')
#         self.state.setPressure(pres, 'pascal')
#         self.state.setSpeciesAmount('H2O(l)', z_e[0], 'mol')
#         self.state.setSpeciesAmount('CO2(aq)', z_e[1], 'mol')
#         self.state.setSpeciesAmount('Ca++', z_e[2], 'mol')
#         self.state.setSpeciesAmount('CO3--', z_e[3], 'mol')
#         self.states.append(self.state)
#
#     def output(self):
#         n_states = len(self.states)
#         volume_tot = [ChemicalQuantity(state).value('volume(units=m3)') for state in self.states]  # m3
#
#         mass_aq = [ChemicalQuantity(state).value("phaseMass(Aqueous)") for state in self.states]  # kg
#         volume_aq = [ChemicalQuantity(state).value("phaseVolume(Aqueous)") for state in self.states]  # m3
#
#         density_aq = np.zeros(n_states)
#         for i in range(n_states):
#             density_aq[i] = mass_aq[i]/volume_aq[i]
#         mol_total_aq = [ChemicalQuantity(state).value("phaseAmount(Aqueous)") for state in self.states]  # mol
#         H2O = [state.speciesAmount("H2O(l)") for state in self.states]
#         CO2 = [state.speciesAmount("CO2(aq)") for state in self.states]
#         CaCO3 = [state.speciesAmount("CaCO3(aq)") for state in self.states]
#         Ca = [state.speciesAmount("Ca++") for state in self.states]
#         CO3 = [state.speciesAmount("CO3--") for state in self.states]
#
#         mass_gas = [ChemicalQuantity(state).value("phaseMass(Gaseous)") for state in self.states]  # kg
#         volume_gas = [ChemicalQuantity(state).value("phaseVolume(Gaseous)") for state in self.states]  # m3
#         density_gas = np.zeros(n_states)
#         for i in range(n_states):
#             density_gas[i] = mass_gas[i]/volume_gas[i]
#         mol_total_gas = [ChemicalQuantity(state).value("phaseAmount(Gaseous)") for state in self.states]  # mol
#         H2O_gas = [state.speciesAmount("H2O(g)") for state in self.states]
#         CO2_gas = [state.speciesAmount("CO2(g)") for state in self.states]
#
#         mass_solid = [ChemicalQuantity(state).value("phaseMass(Calcite)") for state in self.states]  # kg
#         volume_solid = [ChemicalQuantity(state).value("phaseVolume(Calcite)") for state in self.states]  # m3
#         density_solid = np.zeros(n_states)
#         for i in range(n_states):
#             density_solid[i] = mass_solid[i]/volume_solid[i]
#         mol_total_solid = [ChemicalQuantity(state).value("phaseAmount(Calcite)") for state in self.states]  # mol
#         Calcite = [state.speciesAmount("Calcite") for state in self.states]
#
#
#         S_w, S_g, S_s = np.zeros(n_states), np.zeros(n_states), np.zeros(n_states)
#         L, V, S = np.zeros(n_states), np.zeros(n_states), np.zeros(n_states)
#         mol_total = np.zeros(n_states)
#         z_c, x, y, density = [],[],[], []
#         for i in range(n_states):
#             if volume_aq[i] < 0:
#                 print('######################NEGATIVE AQUEOUS VOLUME WARNING##########################################')
#             mol_total[i] = mol_total_aq[i] + mol_total_gas[i] + mol_total_solid[i]
#             S_w[i] = volume_aq[i] / volume_tot[i]
#             S_g[i] = volume_gas[i] / volume_tot[i]
#             S_s[i] = volume_solid[i] / volume_tot[i]
#             S_w_norm = S_w / (S_g+S_w)
#             S_g_norm = S_g / (S_g+S_w)
#             #L[i] = (density_aq[i] * S_w[i]) / (density_gas[i] * S_g[i] + density_aq[i] * S_w[i] + density_solid[i] * S_s[i])
#             L[i] = (density_aq[i] * S_w_norm[i]) / (density_gas[i] * S_g_norm[i] + density_aq[i] * S_w_norm[i])
#             V[i] = (density_gas[i] * S_g[i]) / (density_gas[i] * S_g[i] + density_aq[i] * S_w[i] + density_solid[i] * S_s[i])
#             S[i] = (density_solid[i] * S_s[i]) / (density_gas[i] * S_g[i] + density_aq[i] * S_w[i] + density_solid[i] * S_s[i])
#             x = np.append(x,[H2O[i]/mol_total_aq[i], CO2[i]/mol_total_aq[i], Ca[i]/mol_total_aq[i], CO3[i]/mol_total_aq[i], CaCO3[i]/mol_total_aq[i]])
#             y = np.append(y,[H2O_gas[i]/mol_total_gas[i], CO2_gas[i]/mol_total_gas[i], 0, 0, 0])
#             z_c = np.append(z_c,[(H2O[i]+H2O_gas[i])/mol_total[i], (CO2[i]+CO2_gas[i])/mol_total[i], Ca[i]/mol_total[i], CO3[i]/mol_total[i], (CaCO3[i]+Calcite[i])/mol_total[i]])
#             density = np.append(density, [density_aq, density_gas, density_solid])
#         return L, V, S, x, y, z_c, density,self.rate