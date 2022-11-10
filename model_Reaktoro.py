from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import sim_params
import numpy as np
from properties_basic import *
from property_container import *
from physics_comp_sup import Compositional
import matplotlib.pyplot as plt
from reaktoro import *  # reaktoro v2.0.0rc22

class Model(DartsModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        log_flag = 0  # zero for normal, 1 for log
        kinetic_flag = 0  # zero for eq, 1 for kinetic

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.zero = 1e-11
        perm = 100  # np.array([1,1,0,0])  # mD # / (1 - solid_init) ** trans_exp
        nx = 100

        # self.dx = np.array([0.308641975, 0.617283951, 0.925925926, 1.234567901, 1.543209877, 1.851851852, 2.160493827,
        #                2.469135802, 2.777777778, 3.086419753, 3.395061728, 3.703703704, 4.012345679, 4.320987654,
        #                4.62962963, 4.938271605, 5.24691358, 5.555555556, 5.864197531, 6.172839506, 6.481481481,
        #                6.790123457, 7.098765432, 7.407407407, 7.716049383, 8.024691358, 8.333333333, 8.641975309,
        #                8.950617284, 9.259259259, 9.567901235, 9.87654321, 10.18518519, 10.49382716, 10.80246914,
        #                11.11111111, 11.41975309, 11.72839506, 12.03703704, 12.34567901, 12.65432099, 12.96296296,
        #                13.27160494, 13.58024691, 13.88888889, 14.19753086, 14.50617284, 14.81481481, 15.12345679,
        #                15.43209877, 15.74074074, 16.04938272, 16.35802469, 16.66666667, 16.97530864, 17.28395062,
        #                17.59259259, 17.90123457, 18.20987654, 18.51851852, 18.82716049, 19.13580247, 19.44444444,
        #                19.75308642, 20.0617284, 20.37037037, 20.67901235, 20.98765432, 21.2962963, 21.60493827,
        #                21.91358025, 22.22222222, 22.5308642, 22.83950617, 23.14814815, 23.45679012, 23.7654321,
        #                24.07407407, 24.38271605, 24.69135802])  # totals 1000

        self.dx = np.ones(nx)
        #nx = 500
        self.poro = np.ones(nx) * 0.2
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=self.dx, dy=10, dz=1, permx=perm, permy=perm,
                                         permz=perm, poro=self.poro, depth=2000)
        """well location"""
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1, j=1, k=1, multi_segment=False)

        self.reservoir.add_well("P1")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=nx, j=1, k=1, multi_segment=False)

        """Physical properties"""
        # Create property containers:
        # components_name = ['H2O', 'CO2', 'Ca+2', 'CO3-2', 'Calcite']
        components_name = ['H2O(l)', 'CO2(aq)', 'Ca+2', 'CO3-2', 'Calcite']

        if kinetic_flag == 0:
            elements_name = components_name[:4]
        else:
            elements_name = components_name[:5]
        # elements_name = ['H', 'C', 'O', 'Ca', 'Z']
        self.reaktoro = Reaktoro(components_name, len(elements_name), kinetic_flag)  # Initialise Reaktoro

        # self.db = PhreeqcDatabase.fromFile('C:/Users/Jaro/Documents/inversemodelling/code_thesis/DARTS_1D_model/Comp4George/phreeqc_cut.dat')
        # self.db = PhreeqcDatabase('phreeqc.dat')
        self.db = SupcrtDatabase('supcrtbl')
        if kinetic_flag == 0:
            E_mat = np.array([[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 1],
                              [0, 0, 0, 1, 1]])
        else:
            E_mat = np.array([[1, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1]])

        # E_mat = np.array([[2, 0, 0, 0, 0],
        #                   [0, 1, 0, 1, 1],
        #                   [1, 2, 0, 3, 3],
        #                   [0, 0, 1, 0, 0],
        #                   [0, 0, 2, -2, 0]])
        # E_mat = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
        #                   [0, 1, 0, 0, 0, 0, 0, 0],
        #                   [0, 0, 1, 0, 0, 0, 1, 0],
        #                   [0, 0, 0, 1, 0, 0, 1, 0],
        #                   [0, 0, 0, 0, 1, 0, 0, 1],
        #                   [0, 0, 0, 0, 0, 1, 0, 1]])


        E_mat_ini = E_mat
        components_name_Mw = ['H2O', 'CO2', 'Ca+2', 'CO3-2', 'CaCO3']
        Mw = np.zeros(len(components_name_Mw))
        for i in range(len(Mw)):
            component = Species(str(components_name_Mw[i]))
            Mw[i] = component.molarMass() * 1000
        # Mw = [1,1,1,1,1]
        # aq_list = components_name
        # aq_species = StringList(aq_list)
        # sol_species = StringList([])
        # aq = AqueousPhase(aq_species)
        # # aq.setActivityModel(ActivityModelHKF())
        # # sol = MineralPhase(sol_species.data()[0])
        # system_inj, system_ini = ChemicalSystem(self.db, aq), ChemicalSystem(self.db, aq)
        # state_inj, state_ini = ChemicalState(system_inj), ChemicalState(system_ini)
        # specs_inj, specs_ini = EquilibriumSpecs(system_inj), EquilibriumSpecs(system_ini)
        # specs_inj.temperature(), specs_ini.temperature()
        # specs_inj.pressure(), specs_ini.pressure()
        # specs_inj.pH(), specs_ini.pH()
        # specs_inj.charge(), specs_ini.charge()
        # specs_inj.openTo('Cl-'), specs_ini.openTo('Cl-')
        # solver_inj = EquilibriumSolver(specs_inj)
        # solver_ini = EquilibriumSolver(specs_ini)
        #
        # conditions_inj, conditions_ini = EquilibriumConditions(specs_inj), EquilibriumConditions(specs_ini)
        #
        # conditions_inj.temperature(320, 'kelvin'),      conditions_ini.temperature(320, 'kelvin')
        # conditions_inj.pressure(100, 'bar'),            conditions_ini.pressure(100, 'bar')
        # conditions_inj.pH(11.1),                        conditions_ini.pH(7)
        # conditions_inj.charge(0),                       conditions_ini.charge(0)
        #
        # state_inj.set('H2O', 1, 'kg'),                  state_ini.set('H2O', 1, 'kg')
        # state_inj.set('Na+', 8270.6, 'mg'),             state_ini.set('Na+', 3931, 'mg')  # 8270.6, 3931# ppm
        # state_inj.set('CO3-2', 5660, 'mg'),             state_ini.set('CO3-2', 17.8, 'mg')  # 5660, 17.8
        # state_inj.set('Cl-', 1, 'mg'),                  state_ini.set('Cl-', 1, 'mg')  # 33.5, 6068  / 5900, 6051.75
        #
        # solver_inj.solve(state_inj, conditions_inj)
        # solver_ini.solve(state_ini, conditions_ini)
        # cp_inj = ChemicalProps(state_inj)
        # z_c_inj = np.zeros(aq_species.size() + sol_species.size())
        # for i in range(len(z_c_inj)):
        #     z_c_inj[i] = float(cp_inj.speciesAmount(i))
        # for i in range(len(z_c_inj)):
        #     if z_c_inj[i] < self.zero:
        #         z_c_inj[i] = 0
        # z_c_inj = [float(i) / sum(z_c_inj) for i in z_c_inj]
        # z_e_inj = np.zeros(E_mat_ini.shape[0])
        # for i in range(E_mat_ini.shape[0]):
        #     z_e_inj[i] = np.divide(np.sum(np.multiply(E_mat_ini[i], z_c_inj)), np.sum(np.multiply(E_mat_ini, z_c_inj)))
        #
        # cp_ini = ChemicalProps(state_ini)
        # z_c_ini = np.zeros(aq_species.size() + sol_species.size())
        # for i in range(len(z_c_ini)):
        #     z_c_ini[i] = float(cp_ini.speciesAmount(i))
        # for i in range(len(z_c_ini)):
        #     if z_c_ini[i] < self.zero:
        #         z_c_ini[i] = 0
        # z_c_ini = [float(i) / sum(z_c_ini) for i in z_c_ini]
        # z_e_ini = np.zeros(E_mat_ini.shape[0])
        # for i in range(E_mat_ini.shape[0]):
        #     z_e_ini[i] = np.divide(np.sum(np.multiply(E_mat_ini[i], z_c_ini)), np.sum(np.multiply(E_mat_ini, z_c_ini)))

        # print(state_inj)
        # print(state_ini)
        # print(AqueousProps(state_ini).pH())

        self.thermal = 0

        # fill in density for amount of solids present
        solid_density = [2000]
        self.property_container = model_properties(phases_name=['gas', 'wat', 'sol'],
                                                   components_name=components_name, elements_name=elements_name,
                                                   reaktoro=self.reaktoro, E_mat=E_mat, diff_coef=1e-9, rock_comp=1e-5,
                                                   Mw=Mw, log_flag=log_flag, min_z=self.zero / 10, solid_dens=solid_density)

        """ properties correlations """
        # self.property_container.flash_ev = Flash(self.components[:-1], [10, 1e-12, 1e-1], self.zero)
        # return [y, x], [V, 1-V]
        self.property_container.density_ev = dict([('gas', Density(compr=1e-4, dens0=100)),
                                                   ('wat', Density(compr=1e-6, dens0=1000)),
                                                   ('sol', Density(compr=0, dens0=2630))])
        self.property_container.viscosity_ev = dict([('gas', ViscosityConst(0.1)),
                                                     ('wat', ViscosityConst(1)),
                                                     ('sol', ViscosityConst(1))])
        self.property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                                    ('wat', PhaseRelPerm("wat")),
                                                    ('sol', PhaseRelPerm('sol'))])

        # ne = self.property_container.nc + self.thermal
        # self.property_container.kinetic_rate_ev = kinetic_basic(equi_prod, 1e-0, ne)
        self.property_container.kinetic_rate_ev = Reaktoro_kinetics(self.reaktoro)

        H2O = 55.5  # mol
        Na = 0.8  # 10 # 4.5 # 0.8 # 3.44 # mol/kgW
        Cl = 0.8  # 10 # 4.5 # 0.8 # mol/kgW

        # CO2 = 10000000
        # H2O_2 = 1
        # Na_2 = 10/55
        # Cl_2 = 10/55
        # Ca = 1
        # CO3 = 1
        if kinetic_flag == 0:
            z_e_ini = [H2O - 4 * self.zero, self.zero, Na, Cl]
            z_e_ini = [float(i) / sum(z_e_ini) for i in z_e_ini]
            z_e_inj = [self.zero, 1-3*self.zero, self.zero, self.zero]
            #z_e_ini = [111, 0.8, 57.9, 0.8, 0]
            #z_e_inj = [0, 50, 100, 0, 0]
            #z_e_ini = [float(i) / sum(z_e_ini) for i in z_e_ini]
            #z_e_inj = [float(i) / sum(z_e_inj) for i in z_e_inj]
        else:
            z_e_ini = [H2O - 4 * self.zero, self.zero, Na, Cl, self.zero]
            z_e_ini = [float(i) / sum(z_e_ini) for i in z_e_ini]
            z_e_inj = [self.zero, 1-4*self.zero, self.zero, self.zero, self.zero]

        z_diff = np.zeros(len(z_e_ini))
        min_z = np.zeros(len(z_e_ini))
        max_z = np.zeros(len(z_e_ini))
        for i in range(len(z_e_ini)):
            z_diff[i] = abs(z_e_ini[i] - z_e_inj[i])  # 0.005
            # if abs(z_e_ini[i] - z_e_inj[i]) < z_diff[i]:
            #     z_diff[i] = abs(z_e_ini[i] - z_e_inj[i])
            min_z[i] = min(z_e_ini[i], z_e_inj[i]) - z_diff[i]
            max_z[i] = max(z_e_ini[i], z_e_inj[i]) + z_diff[i]
            if min_z[i] < self.zero:
                min_z[i] = self.zero
            if max_z[i] > 1 - self.zero:
                max_z[i] = 1 - self.zero
        min_z = min_z[:-1]
        max_z = max_z[:-1]
        if log_flag == 1:
            self.ini_stream = np.log(z_e_ini[:-1])
            self.inj_stream = np.log(z_e_inj[:-1])
            print('initial: ', np.exp(self.ini_stream))
            print('injection: ', np.exp(self.inj_stream))
            min_z = np.log(min_z)
            max_z = np.log(max_z)
        else:
            self.ini_stream = z_e_ini[:-1]
            self.inj_stream = z_e_inj[:-1]
            print('initial: ', self.ini_stream)
            print('injection: ', self.inj_stream)

        """ Activate physics """
        #n_points = [101, 10001, 10001, 10001, 100001]
        n_points = [1001]*len(elements_name)
        min_z = [self.zero / 10] * (len(elements_name)-1)
        max_z = [1 - self.zero / 10] * (len(elements_name)-1)
        #max_z[-1] = 0.3
        # min_z[0] = self.zero/1000
        self.physics = Compositional(self.property_container, self.timer, n_points=n_points, min_p=1, max_p=1000,
                                     min_z=min_z, max_z=max_z, cache=0)

        # print(self.ini_stream)
        # print(self.inj_stream)
        # exit()
        # self.ini_stream = [0.48, 0.48, 0.009, 0.009]
        # self.inj_stream = [0.47, 0.47, 0.02, 0.02]
        # print(z_e_inj)
        # print(z_e_ini)
        # exit()
        self.params.first_ts = 1e-2
        self.params.max_ts = 1
        self.params.mult_ts = 2
        self.params.log_transform = log_flag

        self.params.tolerance_newton = 1e-5
        self.params.tolerance_linear = 1e-6
        self.params.max_i_newton = 10
        self.params.max_i_linear = 50
        self.params.newton_type = sim_params.newton_local_chop
        self.params.nonlinear_norm_type = sim_params.LINF
        # self.params.newton_params[0] = 0.2

        self.timer.node["initialization"].stop()

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        """ initialize conditions for all scenarios"""
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, 100, self.ini_stream)
        # volume = np.array(self.reservoir.volume, copy=False)
        # volume.fill(100)
        # volume[0] = 1e10
        # volume[-1] = 1e10
        # pressure = np.array(self.reservoir.mesh.pressure, copy=False)
        # pressure.fill(200)
        # pressure[0] = 205
        # pressure[-1] = 195
        # # #pressure[-1] = 100
        # comp = np.array(self.reservoir.mesh.composition, copy=False)
        # comp[0] = self.inj_stream[0]
        # comp[1] = self.inj_stream[1]
        # comp[2] = self.inj_stream[2]
        # comp[3] = self.inj_stream[3]
        # comp[4] = self.inj_stream[4]
        # composition = np.array(self.reservoir.mesh.composition, copy=False)
        # n_half = int(self.reservoir.nx * self.reservoir.ny * self.reservoir.nz / 2)
        # composition[2*n_half:] = 1e-6

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                # w.control = self.physics.new_rate_inj(0.2, self.inj_stream, 0)
                # w.control = self.physics.new_bhp_inj(105, self.inj_stream)
                w.control = self.physics.new_bhp_inj(105, self.inj_stream)
                #w.constraint = self.physics.new_rate_inj(500, self.inj_stream, 0)
            else:
                w.control = self.physics.new_bhp_prod(95)

    def set_op_list(self):
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks
        self.op_num[n_res:] = 1
        self.op_list = [self.physics.acc_flux_itor, self.physics.acc_flux_w_itor]

    def properties(self, state):
        (sat, x, rho, rho_m, mu, kr, ph) = self.property_container.evaluate(state)
        return sat[0]

    def flash_properties(self, ze, T, P):
        nu, x, zc, density, pH, gas = model_properties.Flash_Reaktoro(self, ze, T, P, self.reaktoro)
        return nu, x, zc, density, pH, gas


class model_properties(property_container):
    def __init__(self, phases_name, components_name, elements_name, reaktoro, E_mat, Mw, log_flag, min_z=1e-12,
                 diff_coef=float(0), rock_comp=1e-6, solid_dens=None):
        if solid_dens is None:
            solid_dens = []
        # Call base class constructor
        super().__init__(phases_name, components_name, Mw, min_z=min_z, diff_coef=diff_coef,
                         rock_comp=rock_comp, solid_dens=solid_dens)
        self.n_e = len(elements_name)
        self.elements_name = elements_name
        self.reaktoro = reaktoro
        self.E_mat = E_mat
        self.log_flag = log_flag
        self.Mw = Mw

    def run_flash_R(self, pressure, ze):
        # if self.log_flag == 1:
        #     ze = np.exp(ze)
        # Make every value that is the min_z equal to 0, as Reaktoro can work with 0, but not transport
        # ze = comp_extension(ze, self.min_z)
        nu, x, zc, density, pH, gas = self.Flash_Reaktoro(ze, 320, pressure, self.reaktoro)
        # zc = comp_correction(zc, self.min_z)

        # Solid phase always needs to be present
        ph = list(range(len(nu)))  # ph = range(number of total phases)
        min_ph = 0.005  # min value to be considered inside the phase
        # for i in range(len(self.nu)):
        #     if density[i] < 0 and min_ph < 0.1:
        #         min_ph_new = self.nu[i]
        #         min_ph = max(min_ph, min_ph_new)
        #     if density[1] > 1500 and min_ph < 0.1:
        #         min_ph_new = self.nu[1]
        #         min_ph = max(min_ph, min_ph_new)
        # if nu[0] <= min_ph:  # if vapor phase is less than min_z, it does not exist
        #     del ph[0]  # remove entry of vap
        #     density[0] = 0
        # elif nu[1] <= min_ph:  # if liq phase is less than min_z, it does not exist
        #     del ph[1]
        #     density[1] = 0
        # solid phase always present
        for i in range(len(nu)):
            if i > len(nu)-1 and nu[i] < self.min_z:
                nu[i] = self.min_z
        # for i in range(len(self.nu)):
        #     if density[i] < 0 or density[1] > 2000:
        #         print('Partial molar problems, likely, density is below 0 or above 2000 for aqueous phase')
        #         print('ze', ze)
        #         print('zc', zc)
        #         print('nu', self.nu)
        #         print(density)
        return x, nu, density

    def Flash_Reaktoro(self, z_e, T, P, reaktoro):
        # if z_e[-1] > 1e-15:
        #     z_e[-1] = 1e-16
        # if z_e[0] > 0.8:
        #     z_e[0] = 0.8

        if z_e[2] != z_e[3]:
            ze_new = (z_e[2] + z_e[3]) / 2
            z_e[2] = ze_new
            z_e[3] = ze_new
            # z_e = [float(i) / sum(z_e) for i in z_e]
        # if z_e[4] != z_e[5]:
        #     ze_new = (z_e[4] + z_e[5]) / 2
        #     z_e[4] = ze_new
        #     z_e[5] = ze_new
        # if z_e[0] != z_e[1]:
        #     ze_new = (z_e[1] + z_e[0]) / 2
        #     z_e[0] = ze_new
        #     z_e[1] = ze_new
        # ['if reaktoro.kinetic_flag == 0:
        # rho = np.zeros(len(z_e))
        # c_e = np.zeros(len(z_e))
        # for i in range(len(rho)):
        #     rho[i] = z_e[i]*self.Mw[i]
        #     c_e[i] = z_e[i] * rho[i]
        # rho_t_e = sum(rho)
        #
        # print(rho)
        # print(c_e)
        # print(rho_t_e)
        # Mw = [18.01528, 44.0098, 22.98921942, 35.45324858, 58.442468]
        # Mw_tot = sum(Mw)
        # for i in range(len(z_e)):
        #     z_e[i] = z_e[i] * Mw[i]
        # z_e = z_e * Mw_tot
        multiplier = 1
        reaktoro.addingproblem(T, P, z_e, multiplier)
        nu, x, z_c, density, pH, gas, density_tot_e = reaktoro.output(z_e)  # z_c order is determined by user, check if its the same order as E_mat
        #reaktoro.addingproblem(T, P, z_e, sum(density_tot_e))
        #nu, x, z_c, density, pH, gas, density_tot_e = reaktoro.output(z_e)
        # else:
        #     reaktoro.addingproblem_kin(T, P, z_e)
        #     nu, x, z_c, density, pH, rate, gas = reaktoro.output_kin()

        return nu, x, z_c, density, pH, gas


def comp_extension(z, min_z):
    sum_z = 0
    z_correct = False
    C = len(z)
    for c in range(C):
        new_z = z[c]
        if new_z <= min_z:
            new_z = 0
            z_correct = True
        elif new_z >= 1 - min_z:
            new_z = 1
            z_correct = True
        sum_z += new_z
    new_z = 1 - sum_z
    if new_z <= min_z:
        new_z = 0
        z_correct = True
    sum_z += new_z
    if z_correct:
        for c in range(C):
            new_z = z[c]
            if new_z <= min_z:
                new_z = 0
            elif new_z >= 1 - min_z:
                new_z = 1
            new_z = new_z / sum_z  # Rescale
            z[c] = new_z
    return z


def comp_correction(z, min_z):
    sum_z = 0
    z_correct = False
    C = len(z)
    for c in range(C):
        new_z = z[c]
        if new_z < min_z:
            new_z = min_z
            z_correct = True
        elif new_z > 1 - min_z:
            new_z = 1 - min_z
            z_correct = True
        sum_z += new_z  # sum previous z of the loop
    new_z = 1 - sum_z  # Get z_final
    if new_z < min_z:
        new_z = min_z
        z_correct = True
    sum_z += new_z  # Total sum of all z's
    if z_correct:  # if correction is needed
        for c in range(C):
            new_z = z[c]
            new_z = max(min_z, new_z)
            new_z = min(1 - min_z, new_z)  # Check whether z is in between min_z and 1-min_z
            new_z = new_z / sum_z  # Rescale
            z[c] = new_z
    return z

class Reaktoro:
    def __init__(self, components, ne, kinetic_flag):
        self.kinetic_flag = kinetic_flag
        # db = SupcrtDatabase("supcrtbl")
        # db = PhreeqcDatabase.fromFile("phreeqc_cut.dat")
        # db = PhreeqcDatabase.fromFile("phreeqc_cat_ion.dat")
        # db = PhreeqcDatabase.fromFile('logKFrom961_bdotFixedTuned.dat')
        db = PhreeqcDatabase('phreeqc.dat')

        '''Hardcode'''
        self.aq_comp = StringList(['H2O', 'CO2', 'Ca+2', 'CO3-2'])
        self.ne = ne
        self.gas_comp = StringList(["H2O(g)", "CO2(g)"])
        aq = AqueousPhase(StringList(['H2O', 'CO2', 'Ca+2', 'CO3-2']))
        components_name_Mw = ['H2O', 'CO2', 'Ca+2', 'CO3-2', 'CaCO3']
        self.Mw = np.zeros(len(components_name_Mw))
        for i in range(len(self.Mw)):
            component = Species(str(components_name_Mw[i]))
            self.Mw[i] = component.molarMass() * 1000
        #aq.setActivityModel(ActivityModelDebyeHuckel())
        # aq.setActivityModel(chain(
        #     ActivityModelDebyeHuckel(),
        #     ActivityModelDrummond("CO2")
        # ))
        # aq.setActivityModel(ActivityModelHKF())
        gas = GaseousPhase(self.gas_comp)
        #gas.setActivityModel(ActivityModelPengRobinson())
        # gas.setActivityModel(ActivityModelSpycherPruessEnnis())

        # aq.setActivityModel(ActivityModelHKF())
        if kinetic_flag == 0:
            self.sol_comp = ['Calcite']
            for i in range(len(self.sol_comp)):
                globals()['sol%s' % i] = MineralPhase(self.sol_comp[i])
            self.system = ChemicalSystem(db, gas, aq, sol0)
        else:
            self.system = ChemicalSystem(db, gas, aq)

            self.sol_comp = ['Calcite']
            for i in range(len(self.sol_comp)):
                globals()['sol%s' % i] = MineralPhase(self.sol_comp[i])
            self.system_kin = ChemicalSystem(db, gas, aq, sol0)
            self.specs_kin = EquilibriumSpecs(self.system_kin)
            self.specs_kin.temperature()
            self.specs_kin.pressure()
            # self.specs.charge()
            # self.specs.openTo("Cl-")
            self.solver_kin = EquilibriumSolver(self.specs_kin)
            # self.cp = type(object)
            self.cp_kin: ChemicalProps = ChemicalProps(self.system_kin)

        self.specs = EquilibriumSpecs(self.system)
        self.specs.temperature()
        self.specs.pressure()
        # self.specs.charge()
        # self.specs.openTo("Cl-")
        opts = EquilibriumOptions()
        opts.epsilon = 1e-15
        self.solver = EquilibriumSolver(self.specs)
        self.solver.setOptions(opts)
        # self.cp = type(object)
        self.cp: ChemicalProps = ChemicalProps(self.system)
        # self.specs.pH()

    def addingproblem(self, temp, pres, z_e, multiplier):
        self.state = ChemicalState(self.system)
        self.state.temperature(temp, 'kelvin')
        self.state.pressure(pres, 'bar')
        #for i in range(self.aq_comp.size()):
            # if z_e[i] == 0:
            #     z_e[i] = 1e-50
            #self.state.set(self.aq_comp[i], z_e[i]*multiplier, 'mol')  #*density_elements
        # state.set('Kaolinite', z_e[i+1], 'mol')
        # state.set('Quartz', z_e[i + 1], 'mol')
        self.state.set('H2O', z_e[0], 'mol')
        self.state.set('CO2(g)', z_e[1], 'mol')
        self.state.set('Ca+2', z_e[2], 'mol')
        self.state.set('CO3-2', z_e[3], 'mol')
        conditions = EquilibriumConditions(self.specs)
        conditions.temperature(temp, "kelvin")
        conditions.pressure(pres, "bar")
        if self.kinetic_flag == 1:
            # a_Na = float(ChemicalProps(self.state).speciesActivity('Ca+2'))
            # a_Cl = float(ChemicalProps(self.state).speciesActivity('CO3-2'))
            # self.Q = a_Na * a_Cl
            self.state_kin = ChemicalState(self.system_kin)
            self.state_kin.temperature(temp, 'kelvin')
            self.state_kin.pressure(pres, 'bar')
            for i in range(self.aq_comp.size()):
                # if z_e[i] == 0:
                #     z_e[i] = 1e-50
                self.state_kin.set(self.aq_comp[i], z_e[i], 'mol')
            conditions_kin = EquilibriumConditions(self.specs_kin)
            conditions_kin.temperature(temp, "kelvin")
            conditions_kin.pressure(pres, "bar")
            result_kin = self.solver_kin.solve(self.state_kin, conditions_kin)
            self.cp_kin.update(self.state_kin)
            # a_Na = float(ChemicalProps(self.state_kin).speciesActivity('Ca+2'))
            # a_Cl = float(ChemicalProps(self.state_kin).speciesActivity('CO3-2'))
            # self.K_eq = a_Na*a_Cl

        # conditions.charge(0)
        ne = self.system.elements().size()
        self.iH = self.system.elements().index("H")  # the index of component H
        self.iO = self.system.elements().index("O")  # the index of component O
        self.iC = self.system.elements().index("C")  # the index of component C
        self.iCa = self.system.elements().index('Ca')
        self.iZ = ne
        # iZ = self.system.elements().index('z')
        # b = np.zeros(ne + 1)
        # b[self.iH] = z_e[0]
        # b[self.iO] = z_e[1]
        # b[self.iC] = z_e[2]
        # b[self.iCa] = z_e[3]
        # b[self.iZ] = z_e[4]
        result = self.solver.solve(self.state, conditions)
        self.cp.update(self.state)
        self.failure = False
        if not result.optima.succeeded:  # if not found a solution
            # print('Reaktoro did not find solution')
            self.failure = True
            # print('z_e', z_e)
            # print(self.state)
            # exit()

    def output(self, z_e_output):
        gas_props: ChemicalPropsPhaseConstRef = self.cp.phaseProps(0)
        liq_props: ChemicalPropsPhaseConstRef = self.cp.phaseProps(1)
        if self.kinetic_flag == 0:
            sol_props: ChemicalPropsPhaseConstRef = self.cp.phaseProps(2)
        # sol_props2: ChemicalPropsPhaseConstRef = self.cp.phaseProps(3)
        # aq_props = AqueousProps(self.state)
        # print(self.state)
        # print(aq_props.saturationIndices())
        # print(aq_props.saturationSpecies().data()[5].name())
        # pH = aq_props.saturationIndex(5)
        # if pH > 1:
        #     pH = 1
        # if pH == 1:
        # print(self.cp.speciesActivity('Na+')*self.cp.speciesActivity('Cl-'))
        '''Hardcode'''
        H2O_aq = self.cp.speciesAmount('H2O')
        H2O_g = self.cp.speciesAmount('H2O(g)')
        H2O = H2O_aq + H2O_g
        CO2_aq = self.cp.speciesAmount('CO2')
        CO2_g = self.cp.speciesAmount('CO2(g)')
        CO2 = CO2_aq + CO2_g
        if self.kinetic_flag == 0:
            solid = self.cp.speciesAmount('Calcite')
        else:
            solid = 0
        # solid2 = self.cp.speciesAmount('Calcite')
        Na = self.cp.speciesAmount('Ca+2')
        Cl = self.cp.speciesAmount('CO3-2')

        H_aq = self.cp.elementAmountInPhase(self.iH, 1)
        C_aq = self.cp.elementAmountInPhase(self.iC, 1)
        O_aq = self.cp.elementAmountInPhase(self.iO, 1)
        Ca_aq = self.cp.elementAmountInPhase(self.iCa, 1)
        Z_aq = self.cp.elementAmountInPhase(self.iZ, 1)
        H_gas = self.cp.elementAmountInPhase(self.iH, 0)
        C_gas = self.cp.elementAmountInPhase(self.iC, 0)
        O_gas = self.cp.elementAmountInPhase(self.iO, 0)
        Ca_sol = self.cp.elementAmountInPhase(self.iCa, 2)
        C_sol = self.cp.elementAmountInPhase(self.iC, 2)
        O_sol = self.cp.elementAmountInPhase(self.iO, 2)

        H = self.cp.elementAmount(self.iH)
        C = self.cp.elementAmount(self.iC)
        O = self.cp.elementAmount(self.iO)
        Ca = self.cp.elementAmount(self.iCa)
        Z = self.cp.elementAmount(self.iZ)
        total = H+C+O+Ca+Z
        total_aq = H_aq + C_aq + O_aq + Ca_aq + Z_aq
        total_gas = H_gas + C_gas + O_gas
        total_sol = Ca_sol + C_sol + O_sol


        if self.kinetic_flag == 0:
            total_mol_sol = sol_props.amount()  # + sol_props2.amount()
        else:
            total_mol_sol = z_e_output[-1]
        total_mol_eq = self.cp.amount()
        total_mol = liq_props.amount() + gas_props.amount() + total_mol_sol
        mol_frac_gas = gas_props.speciesMoleFractions()
        mol_frac_aq = liq_props.speciesMoleFractions()


        '''Hardcode'''
        mol_frac_gas = [float(mol_frac_gas[0]), float(mol_frac_gas[1]), 0, 0, 0]
        mol_frac_aq = [float(mol_frac_aq[0]), float(mol_frac_aq[1]), float(mol_frac_aq[2]), float(mol_frac_aq[3]),
                       0]
        mol_frac_sol = [0, 0, 0, 0, 1]
        assert len(mol_frac_gas) == len(mol_frac_aq) and len(mol_frac_aq) == len(mol_frac_sol), \
            'mol frac should be same length'

        # Partial molar volume equation: V_tot = total_mol * sum(molar_frac*partial mole volume)
        # partial_mol_vol_aq = np.zeros(len(mol_frac_aq))
        # for i in range(len(mol_frac_aq)):
        #     partial_mol_vol_aq[i] = float(liq_props.speciesStandardVolumes()[i])
        # volume_aq = total_mol_aq * np.sum(np.multiply(mol_frac_aq, partial_mol_vol_aq))

        # FLASH OUTPUT REAKTORO
        # nu, x = reaktoro.evaluate(p, T, z)  # phase fractions (mole fraction of components in

        volume_gas = gas_props.volume()
        volume_aq = liq_props.volume()
        if self.kinetic_flag == 0:
            volume_solid = sol_props.volume() # + sol_props2.volume()
        else:
            volume_solid = 0
        volume_tot = self.cp.volume()

        density_gas = gas_props.density()
        density_aq = liq_props.density()
        if self.kinetic_flag == 0:
            density_solid = sol_props.density()
        else:
            density_solid = 2712
        # print(float(sol_props2.density()), 'Calcite')
        # density_solid = (2 * sol_props.density() * sol_props2.density()) / (sol_props.density() + sol_props2.density())
        # density_solid = sol_props.density() * mol_frac_sol[-2] + sol_props2.density() * mol_frac_sol[-1]
        density_tot = self.cp.density()
        S_g = volume_gas / volume_tot
        S_w = volume_aq / volume_tot
        if self.kinetic_flag == 0:
            S_s = volume_solid / volume_tot
            # V = (density_gas * S_g) / (density_gas * S_g + density_aq * S_w + density_solid * S_s)
            # L = (density_aq * S_w) / (density_gas * S_g + density_aq * S_w + density_solid * S_s)
            # S = (density_solid * S_s) / (density_gas * S_g + density_aq * S_w + density_solid * S_s)
            nu = [float(gas_props.amount())/float(total_mol_eq), float(liq_props.amount()/float(total_mol_eq)), float(sol_props.amount()/float(total_mol_eq))]
        else:
            S_s = 10
            # V = (density_gas * S_g) / (density_gas * S_g + density_aq * S_w)
            # L = (density_aq * S_w) / (density_gas * S_g + density_aq * S_w)
            # S = z_e_output[-1]
            nu = [float(gas_props.amount()/float(total_mol)), float(liq_props.amount()/ float(total_mol)), z_e_output[-1]/float(total_mol)]
            nu = [float(i) / sum(nu) for i in nu]

        # print(float(density_tot), (density_gas * S_g + density_aq * S_w + density_solid * S_s))
        # exit()
        x = [mol_frac_gas, mol_frac_aq, mol_frac_sol]
        # print(nu)
        # print(self.cp.speciesMoleFractions())
        # exit()
        '''Hardcode'''
        if self.kinetic_flag == 0:
            z_c = [float(H2O / total_mol), float(CO2 / total_mol),
               float(Na / total_mol), float(Cl / total_mol),
               float(solid / total_mol)]
        else:
            z_c = [float(H2O / total_mol), float(CO2 / total_mol),
                   float(Na / total_mol), float(Cl / total_mol),
                   float(z_e_output[-1] / total_mol)]
        z_c = [float(i) / sum(z_c) for i in z_c]
        if float(density_aq) < 0:
            density_aq = 1100
        density = [float(density_gas), float(density_aq), float(density_solid)]

        # if len(ph) == 1:
        #     self.sat[ph[0]] = 1
        # elif len(ph) == 2:
        #     denom = self.dens_m[ph[0]] - self.dens_m[ph[0]] * self.nu[ph[0]] + self.dens_m[ph[1]] * self.nu[ph[0]]
        #     self.sat[ph[0]] = self.dens_m[ph[1]] * self.nu[ph[0]] / denom
        #     self.sat[ph[1]] = self.dens_m[ph[0]] * self.nu[ph[1]] / denom
        ph = list(range(len(nu)))
        for i in range(len(nu)-1):
            if nu[i] < 1e-10:
                nu[i] = 0
                del ph[i]
        dens_m = np.zeros(len(density))
        for j in ph:
            M = 0
            # molar weight of mixture
            for i in range(len(x[0])):
                M += self.Mw[i] * x[j][i]
            dens_m[j] = density[j] / M
        sat = [0, 0, 0]
        for i in range(len(nu)-1):
            if nu[i] < 1e-10:
                nu[i] = 0
        #print(dens_m)
        if len(ph) == 2:
            denom = dens_m[ph[0]] - dens_m[ph[0]] * nu[ph[0]] + dens_m[ph[1]] * nu[ph[0]]
            sat[ph[0]] = dens_m[ph[1]] * nu[ph[0]] / denom
            sat[ph[1]] = dens_m[ph[0]] * nu[ph[1]] / denom
        else:
            denom = dens_m[0] * dens_m[1] * nu[2] + dens_m[0] * dens_m[2] * nu[1]\
                  + dens_m[1] * dens_m[2] * nu[0]
            sat[0] = dens_m[1] * dens_m[2] * nu[0] / denom
            sat[1] = dens_m[0] * dens_m[2] * nu[1] / denom
            sat[2] = dens_m[0] * dens_m[1] * nu[2] / denom
        # denom = dens_m[0] * dens_m[1] * nu[2] + dens_m[0] * dens_m[2] * nu[1]\
        #         + dens_m[1] * dens_m[2] * nu[0]
        # #print(denom)
        # sat[0] = dens_m[1] * dens_m[2] * nu[0] / denom
        # sat[1] = dens_m[0] * dens_m[2] * nu[1] / denom
        # sat[2] = dens_m[0] * dens_m[1] * nu[2] / denom
        #print(sat)
        #exit()
        # pH = aprops.pH()
        pH = 7
        #S_s = 1
        # E_mat = np.array([[1, 0, 0, 0, 0],
        #                   [0, 1, 0, 0, 0],
        #                   [0, 0, 1, 0, 1],
        #                   [0, 0, 0, 1, 1]])
        # if self.failure:
        #     print('z_c', z_c)
        # density_tot_e = np.zeros(len(density))
        # for j in range(len(density)):
        #     for i in range(len(z_e_output)):
        #         density_tot_e[j] = np.sum((sat[j] * density[j]) * np.sum(np.multiply(E_mat, x[j])))
        #print('sum', sum(density_tot_e))
        density_tot_e = 1
        # pH = sat
        # print(aq_props.speciesMolalities().asarray())
        # exit()
        # pH = [float(aq_props.speciesMolality('CO2(aq)')), float(aq_props.speciesMolality('Na+')), float(aq_props.speciesMolality('Cl-'))]
        gas = [H2O_g/total_mol, CO2_g/total_mol]
        z = np.dot(np.array(nu), np.array(x))

        # for i in range(len(z)):
        #     if z[i] != z_c[i]:
        #         print('z', z)
        #         print('z_in', z_e_output)
        # print(z_e_output)
        # print(z)
        return nu, x, z_c, density, sat, gas, density_tot_e

    def output_kin(self, z_e_output):
        gas_props: ChemicalPropsPhaseConstRef = self.cp.phaseProps(0)
        liq_props: ChemicalPropsPhaseConstRef = self.cp.phaseProps(1)

        '''Hardcode'''
        H2O_aq = self.cp.speciesAmount('H2O(aq)')
        H2O_g = self.cp.speciesAmount('H2O(g)')
        H2O = H2O_aq + H2O_g
        CO2_aq = self.cp.speciesAmount('CO2(aq)')
        CO2_g = self.cp.speciesAmount('CO2(g)')
        CO2 = CO2_aq + CO2_g
        solid = z_e_output[-1]
        # solid2 = self.cp.speciesAmount('Calcite')
        Na = self.cp.speciesAmount('Na+')
        Cl = self.cp.speciesAmount('Cl-')

        total_mol_sol = z_e_output[-1]  # last z_e output is solid
        total_mol = liq_props.amount() + gas_props.amount() + total_mol_sol  # Total mol

        mol_frac_gas = gas_props.speciesMoleFractions()  # fractions from reaktoro
        mol_frac_aq = liq_props.speciesMoleFractions()

        '''Hardcode'''
        mol_frac_gas = [float(mol_frac_gas[0]), float(mol_frac_gas[1]), 0, 0, 0]
        mol_frac_aq = [float(mol_frac_aq[0]), float(mol_frac_aq[1]), float(mol_frac_aq[2]), float(mol_frac_aq[3]),
                       0]
        mol_frac_sol = [0, 0, 0, 0, 1]
        assert len(mol_frac_gas) == len(mol_frac_aq) and len(mol_frac_aq) == len(mol_frac_sol), \
            'mol frac should be same length'

        density_gas = gas_props.density()
        density_aq = liq_props.density()
        density_solid = 2156

        nu = [float(gas_props.amount()/float(total_mol)), float(liq_props.amount()/float(total_mol)), z_e_output[-1]/float(total_mol)]  # nj / ntot
        nu = [float(i) / sum(nu) for i in nu]  # normalise

        x = [mol_frac_gas, mol_frac_aq, mol_frac_sol]

        '''Hardcode'''
        z_c = [float(H2O / total_mol), float(CO2 / total_mol),
               float(Na / total_mol), float(Cl / total_mol),
               float(solid / total_mol)]                    # solid = z_e_output[-1]
        z_c = [float(i) / sum(z_c) for i in z_c]
        if float(density_aq) < 0:                # If density is below 0 due to weird partial molar volume, set to 1000
            density_aq = 1000
        density = [float(density_gas), float(density_aq), float(density_solid)]

        # pH = aprops.pH()
        pH = 7
        density_tot_e = 1
        # pH = [float(aq_props.speciesMolality('CO2(aq)')), float(aq_props.speciesMolality('Na+')), float(aq_props.speciesMolality('Cl-'))]
        gas = [H2O_g, CO2_g]
        return nu, x, z_c, density, pH, gas, density_tot_e

    def output_eq(self, z_e_output):
        gas_props: ChemicalPropsPhaseConstRef = self.cp.phaseProps(0)
        liq_props: ChemicalPropsPhaseConstRef = self.cp.phaseProps(1)
        sol_props: ChemicalPropsPhaseConstRef = self.cp.phaseProps(2)

        '''Hardcode'''
        H2O_aq = self.cp.speciesAmount('H2O(aq)')
        H2O_g = self.cp.speciesAmount('H2O(g)')
        H2O = H2O_aq + H2O_g
        CO2_aq = self.cp.speciesAmount('CO2(aq)')
        CO2_g = self.cp.speciesAmount('CO2(g)')
        CO2 = CO2_aq + CO2_g
        solid = self.cp.speciesAmount('Calcite')
        Na = self.cp.speciesAmount('Na+')
        Cl = self.cp.speciesAmount('Cl-')

        total_mol_eq = self.cp.amount()

        mol_frac_gas = gas_props.speciesMoleFractions()
        mol_frac_aq = liq_props.speciesMoleFractions()

        '''Hardcode'''
        mol_frac_gas = [float(mol_frac_gas[0]), float(mol_frac_gas[1]), 0, 0, 0]
        mol_frac_aq = [float(mol_frac_aq[0]), float(mol_frac_aq[1]), float(mol_frac_aq[2]), float(mol_frac_aq[3]),
                       0]
        mol_frac_sol = [0, 0, 0, 0, 1]
        assert len(mol_frac_gas) == len(mol_frac_aq) and len(mol_frac_aq) == len(mol_frac_sol), \
            'mol frac should be same length'

        density_gas = gas_props.density()
        density_aq = liq_props.density()
        density_solid = sol_props.density()

        nu = [float(gas_props.amount())/float(total_mol_eq), float(liq_props.amount()/float(total_mol_eq)), float(sol_props.amount()/float(total_mol_eq))]

        x = [mol_frac_gas, mol_frac_aq, mol_frac_sol]

        '''Hardcode'''
        z_c = [float(H2O / total_mol), float(CO2 / total_mol),
               float(Na / total_mol), float(Cl / total_mol),
               float(solid / total_mol)]
        z_c = [float(i) / sum(z_c) for i in z_c]

        if float(density_aq) < 0:
            density_aq = 1000
        density = [float(density_gas), float(density_aq), float(density_solid)]

        # pH = aprops.pH()
        pH = 7
        #S_s = 1
        density_tot_e = 1
        # pH = [float(aq_props.speciesMolality('CO2(aq)')), float(aq_props.speciesMolality('Na+')), float(aq_props.speciesMolality('Cl-'))]
        gas = [H2O_g, CO2_g]
        return nu, x, z_c, density, pH, gas, density_tot_e

# class density():
#     def __init__(self, reaktoro):
#         self.reaktoro = reaktoro
#
#     def evaluate(self, pressure, ze):

# class Reaktoro:
#     def __init__(self, components, ne, kinetic_flag):
#         self.kinetic_flag = kinetic_flag
#         db = SupcrtDatabase("supcrtbl")
#         # db = PhreeqcDatabase('phreeqc.dat')
#
#         '''Hardcode'''
#         self.aq_comp = StringList(['H2O(aq)', 'CO2(aq)', 'Na+', 'Cl-'])
#         self.ne = ne
#         self.gas_comp = StringList(["H2O(g)", "CO2(g)"])
#         aq = AqueousPhase(StringList(['H2O(aq)', 'CO2(aq)', 'Na+', 'Cl-']))
#         # aq.setActivityModel(ActivityModelDebyeHuckel())
#         gas = GaseousPhase(self.gas_comp)
#         #gas.setActivityModel(ActivityModelPengRobinson())
#
#         self.system = ChemicalSystem(db, gas, aq)
#
#         self.specs = EquilibriumSpecs(self.system)
#         self.specs.temperature()
#         self.specs.pressure()
#         # self.specs.charge()
#         # self.specs.openTo("Cl-")
#         opts = EquilibriumOptions()
#         opts.epsilon = 1e-20
#         self.solver = EquilibriumSolver(self.specs)
#         self.solver.setOptions(opts)
#         # self.cp = type(object)
#         self.cp: ChemicalProps = ChemicalProps(self.system)
#         # self.specs.pH()
#
#     def addingproblem(self, temp, pres, z_e):
#         self.state = ChemicalState(self.system)
#         self.state.temperature(temp, 'kelvin')
#         self.state.pressure(pres, 'bar')
#         for i in range(self.aq_comp.size()):
#             if z_e[i] == 0:
#                 z_e[i] = 1e-50
#             self.state.set(self.aq_comp[i], z_e[i], 'mol')  #*density_elements
#         conditions = EquilibriumConditions(self.specs)
#         conditions.temperature(temp, "kelvin")
#         conditions.pressure(pres, "bar")
#
#         # conditions.charge(0)
#         result = self.solver.solve(self.state, conditions)
#         self.cp.update(self.state)
#         self.failure = False
#         if not result.optima.succeeded:  # if not found a solution
#             print('Reaktoro did not find solution')
#             self.failure = True
#             print('z_e', z_e)
#             # print(state)