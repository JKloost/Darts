import numpy as np

# from src.cubic_main import *
# from src.Binary_Interactions import *
# from src.flash_funcs import *
from reaktoro import *

#  dummy function
class const_fun():
    def __init__(self, value=0):
        super().__init__()
        self.ret = value

    def evaluate(self, dummy1=0, dummy2=0, dummy3=0):
        return self.ret

class flash_3phase():
    def __init__(self, components, T):
        self.components = components
        self.T = T
        mixture = Mix(components)
        binary = Kij(components)
        mixture.kij_cubic(binary)

        self.eos = preos(mixture, mixrule='qmr', volume_translation=True)

    def evaluate(self, p, zc):
        nu, x, status = multiphase_flash(self.components, zc, self.T, p, self.eos)

        return x, nu


# Uncomment these two lines if numba package is installed and make things happen much faster:
# from numba import jit
# @jit(nopython=True)
# def RR_func(zc, k, eps):
#
#     a = 1 / (1 - np.max(k)) + eps
#     b = 1 / (1 - np.min(k)) - eps
#
#     max_iter = 200  # use enough iterations for V to converge
#     for i in range(1, max_iter):
#         V = 0.5 * (a + b)
#         r = np.sum(zc * (k - 1) / (V * (k - 1) + 1))
#         if abs(r) < 1e-12:
#             break
#
#         if r > 0:
#             a = V
#         else:
#             b = V
#
#     if i >= max_iter:
#         print("Flash warning!!!")
#
#     x = zc / (V * (k - 1) + 1)
#     y = k * x
#
#     return (x, y, V)
#
# class Flash:
#     def __init__(self, components, ki, min_z=1e-11):
#         self.components = components
#         self.min_z = min_z
#         self.K_values = np.array(ki)
#
#     def evaluate(self, pressure, zc):
#
#         (x, y, V) = RR_func(zc, self.K_values, self.min_z)
#         return [y, x], [V, 1-V]


#  Density dependent on compressibility only
class Density4Ions:
    def __init__(self, density, compressibility=0, p_ref=1, ions_fac=0):
        super().__init__()
        # Density evaluator class based on simple first order compressibility approximation (Taylor expansion)
        self.density_rc = density
        self.cr = compressibility
        self.p_ref = p_ref
        self.ions_fac = ions_fac

    def evaluate(self, pres, ion_liq_molefrac):
        return self.density_rc * (1 + self.cr * (pres - self.p_ref) + self.ions_fac * ion_liq_molefrac)

class Density:
    def __init__(self, dens0=1000, compr=0, p0=1, x_mult=0):
        self.compr = compr
        self.p0 = p0
        self.dens0 = dens0
        self.x_max = x_mult

    def evaluate(self, pressure, x_co2):
        density = (self.dens0 + x_co2 * self.x_max) * (1 + self.compr * (pressure - self.p0))
        return density

class ViscosityConst:
    def __init__(self, visc):
        self.visc = visc

    def evaluate(self):
        return self.visc

class Enthalpy:
    def __init__(self, tref=273.15, hcap=0.0357):
        self.tref = tref
        self.hcap = hcap

    def evaluate(self, temp):
        # methane heat capacity
        enthalpy = self.hcap * (temp - self.tref)
        return enthalpy

class PhaseRelPerm:
    def __init__(self, phase, swc=0, sgr=0):
        self.phase = phase

        self.Swc = swc
        self.Sgr = sgr
        if phase == "oil":
            self.kre = 1
            self.sr = self.Swc
            self.sr1 = self.Sgr
            self.n = 2
        elif phase == 'gas':
            self.kre = 1
            self.sr = self.Sgr
            self.sr1 = self.Swc
            self.n = 2
        else:  # water
            self.kre = 1
            self.sr = 0
            self.sr1 = 0
            self.n = 2
        # self.Swc = swc
        # self.Sgr = sgr
        # if phase == 'oil':
        #     self.kre = 0.32
        #     self.sr = self.Swc
        #     self.sr1 = self.Sgr
        #     self.n = 6
        # elif phase == 'gas':
        #     self.kre = 1
        #     self.sr = self.Swc
        #     self.sr1 = self.Sgr
        #     self.n = 2
        # else:
        #     self.kre = 1
        #     self.sr = self.Sgr
        #     self.sr1 = self.Swc
        #     self.n = 6

    def evaluate(self, sat):

        if sat >= 1 - self.sr1:
            kr = self.kre

        elif sat <= self.sr:
            kr = 0

        else:
            # general Brook-Corey
            kr = self.kre * ((sat - self.sr) / (1 - self.Sgr - self.Swc)) ** self.n

        return kr


class Reaktoro_kinetics():
    def __init__(self, reaktoro):
        self.reaktoro = reaktoro

    def evaluate(self, ze, pressure):
        if self.reaktoro.kinetic_flag == 1:
            state = ChemicalState(self.reaktoro.system)  # no solid
            state.temperature(320, 'kelvin')
            state.pressure(pressure, 'bar')
            for i in range(self.reaktoro.aq_comp.size()):
                # if ze[i] == 0:
                #     ze[i] = 1e-50
                state.set(self.reaktoro.aq_comp[i], ze[i], 'mol')
            result = self.reaktoro.solver.solve(state)
            a_Na = float(ChemicalProps(state).speciesActivity('Na+'))
            a_Cl = float(ChemicalProps(state).speciesActivity('Cl-'))
            # print(a_Na)
            # print(a_Cl)
            # exit()
            Q = a_Na * a_Cl

            state_solid = ChemicalState(self.reaktoro.system_kin)
            state_solid.temperature(320, 'kelvin')
            state_solid.pressure(pressure, 'bar')
            for i in range(self.reaktoro.aq_comp.size()):
                # if ze[i] == 0:
                #     ze[i] = 1e-50
                state_solid.set(self.reaktoro.aq_comp[i], ze[i], 'mol')
            result_solid = self.reaktoro.solver_kin.solve(state_solid)
            a_Na_solid = float(ChemicalProps(state_solid).speciesActivity('Na+'))
            a_Cl_solid = float(ChemicalProps(state_solid).speciesActivity('Cl-'))
            K = a_Na_solid * a_Cl_solid
            # print(K)
            cp_solid: ChemicalProps = ChemicalProps(state_solid)
            gas_props: ChemicalPropsPhaseConstRef = cp_solid.phaseProps(0)
            liq_props: ChemicalPropsPhaseConstRef = cp_solid.phaseProps(1)
            solid_props: ChemicalPropsPhaseConstRef = cp_solid.phaseProps(2)
            density_solid = float(solid_props.density())
            density_tot = float(cp_solid.density())
            n_tot = cp_solid.amount()
            nu = [float(gas_props.amount()/n_tot), float(liq_props.amount()/n_tot), float(solid_props.amount()/n_tot)]
            sat_s = (nu[-1]/density_solid)/density_tot
            # # print(sat_s)
            # exit()
            A_f = 1
            E_a = 1
            R = 8.314
            k = A_f * np.exp(-E_a / (R * 320))
            k = 0.05
            A = 0.8 * float(sat_s) * float(density_solid)/np.sum(density_tot)  # s_s = 0....
            #A = 1
            rate = A * k * (1 - Q / K)
            if abs(rate) < 1e-30:
                rate = 0
            # print(rate)
            # print(Q)
            # print(K)
            # exit()
        else:
            rate = 0
        return [0, 0, -rate, -rate, rate]


# class kinetic_basic():
#     def __init__(self, equi_prod, kin_rate_cte, ne, combined_ions=True):
#         self.equi_prod = equi_prod
#         self.kin_rate_cte = kin_rate_cte
#         self.kinetic_rate = np.zeros(ne)
#         self.combined_ions = combined_ions
#
#     def evaluate(self, x, nu_sol):
#         if self.combined_ions:
#             ion_prod = (x[1][1] / 2) ** 2
#             self.kinetic_rate[1] = - self.kin_rate_cte * (1 - ion_prod / self.equi_prod) * nu_sol
#             self.kinetic_rate[-1] = - 0.5 * self.kinetic_rate[1]
#         else:
#             ion_prod = x[1][1] * x[1][2]
#             self.kinetic_rate[1] = - self.kin_rate_cte * (1 - ion_prod / self.equi_prod) * nu_sol
#             self.kinetic_rate[2] = - self.kin_rate_cte * (1 - ion_prod / self.equi_prod) * nu_sol
#             self.kinetic_rate[-1] = - self.kinetic_rate[1]
#         return self.kinetic_rate

# class kinetic_basic():
#     def __init__(self, equi_prod, kin_rate_cte, ne, combined_ions=True):
#         self.kinetic_rate = np.zeros(ne)
#
#     def evaluate(self, z_c, T, P):
#         editor = ChemicalEditor(Database('supcrt98.xml'))              # Database that Reaktoro uses
#         editor.addAqueousPhase("H2O(l) CO2(aq) Ca++ CO3-- CaCO3(aq)")  # Aqueous phase with elem
#         editor.addGaseousPhase('H2O(g) CO2(g)')
#         editor.addMineralPhase('Calcite')
#
#         editor.addMineralReaction("Calcite") \
#             .setEquation("Ca++ + CO3-- = Calcite") \
#             .addMechanism("logk = -5.81 mol/(m2*s); Ea = 23.5 kJ/mol") \
#             .setSpecificSurfaceArea(10, "cm2/g")
#
#         system = ChemicalSystem(editor)        # Set the system
#         reactions = ReactionSystem(editor)     # Set reaction system for kinetics
#
#         kinetic_solver = KineticSolver(reactions)
#
#         partition = Partition(system)  # Partition the system into equilibrium and kinetic reactions
#         partition.setKineticSpecies(["Calcite"])  # Set which species needs to be defined kinetically
#         kinetic_solver.setPartition(partition)
#
#         problem = EquilibriumProblem(system)
#         problem.setPartition(partition)
#
#         problem.setTemperature(T, 'kelvin')
#         problem.setPressure(P, 'pascal')
#
#         # Add the components to the problem
#         problem.add('H2O', z_c[0], 'mol')
#         problem.add('CO2', z_c[1], 'mol')
#         problem.add('Ca++', z_c[2], 'mol')
#         problem.add('CO3--', z_c[3], 'mol')
#         problem.add('Calcite', z_c[4], 'mol')
#
#         state = ChemicalState(system)
#         state.scaleVolume(1.0, "m3")
#
#         properties = state.properties()
#         self.kinetic_rate = ReactionSystem(editor).rates(properties).val
#         if np.isnan(self.kinetic_rate):
#             self.kinetic_rate = 0
#         else:
#             print('kinetic_rate', self.kinetic_rate)
#         return [0, 0, -self.kinetic_rate, -self.kinetic_rate]
