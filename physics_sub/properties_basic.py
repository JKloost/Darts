import numpy as np
from reaktoro import *
#from src.cubic_main import *
#from src.Binary_Interactions import *
#from src.flash_funcs import *


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
from numba import jit
@jit(nopython=True)
def RR_func(zc, k, eps):

    a = 1 / (1 - np.max(k)) + eps
    b = 1 / (1 - np.min(k)) - eps

    max_iter = 200  # use enough iterations for V to converge
    for i in range(1, max_iter):
        V = 0.5 * (a + b)
        r = np.sum(zc * (k - 1) / (V * (k - 1) + 1))
        if abs(r) < 1e-12:
            break

        if r > 0:
            a = V
        else:
            b = V

    if i >= max_iter:
        print("Flash warning!!!")

    x = zc / (V * (k - 1) + 1)
    y = k * x

    return (x, y, V)

class Flash:
    def __init__(self, components, ki, min_z=1e-11):
        self.components = components
        self.min_z = min_z
        self.K_values = np.array(ki)

    def evaluate(self, pressure, zc):

        (x, y, V) = RR_func(zc, self.K_values, self.min_z)
        return [y, x], [V, 1-V]


# class Flash:
#     def __init__(self,components):
#         self.components = components
#
#     def evaluate(self, pressure, z_beta):
#         m = Model()
#         m.addingproblem(m.T, pressure, z_beta)
#         amount_elements = m.states[0].elementAmounts()  # element amounts in alphabet order: C, Ca, H, O
#         m.solver.solve(m.states[0], m.T, m.p, amount_elements)
#         L, V, S, x, y, z_c, density = m.output()  # this outputs comp h20, co2, ca, co3, caco3
#         return [y,x], [V, L]

class Variables():
    def __init__(self):
        self.p = 1e7    # in pascal
        self.T = 320    # in kelvin

class Model(Variables):
    def __init__(self):
        super().__init__()
        editor = ChemicalEditor(Database('supcrt98.xml'))              # Database that Reaktoro uses
        editor.addAqueousPhase("H2O(l) CO2(aq) Ca++ CO3-- CaCO3(aq)")  # Aqueous phase with elem
        editor.addGaseousPhase('H2O(g) CO2(g)')
        editor.addMineralPhase('Calcite')
        self.system = ChemicalSystem(editor)       # Set the system
        self.solver = EquilibriumSolver(self.system)    # solves the system
        #self.reactions = ReactionSystem(self.editor)
        #self.reaction = ReactionEquation('Ca++ + CO3-- = CaCO3(aq)')
        # print(self.reaction.numSpecies())
        self.states = []
        # Two version can be coded with the same solution. One uses the EquilibriumProblem function,
        # Other will use the state function. Both need to output a state in order to be solved by EquilibriumSolver

    def addingproblem(self, temp, pres, z_e):
        self.problem = EquilibriumProblem(self.system)
        self.problem.setTemperature(temp, 'kelvin')
        self.problem.setPressure(pres, 'pascal')
        self.problem.add('H2O', z_e[0], 'mol')
        self.problem.add('CO2', z_e[1], 'mol')
        self.problem.add('Ca++', z_e[2], 'mol')
        self.problem.add('CO3--', z_e[3], 'mol')
        self.state = equilibrate(self.problem)  # Equilibrate the problem in order to write to state
        self.states.append(self.state)

    def addingstates(self, temp, pres, z_e):
        self.state = ChemicalState(self.system)
        self.state.setTemperature(temp, 'kelvin')
        self.state.setPressure(pres, 'pascal')
        self.state.setSpeciesAmount('H2O(l)', z_e[0], 'mol')
        self.state.setSpeciesAmount('CO2(aq)', z_e[1], 'mol')
        self.state.setSpeciesAmount('Ca++', z_e[2], 'mol')
        self.state.setSpeciesAmount('CO3--', z_e[3], 'mol')
        self.states.append(self.state)

    def output(self):
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
        z_c, x, y, density = [],[],[], []
        for i in range(n_states):
            if volume_aq[i] < 0:
                print('######################NEGATIVE AQUEOUS VOLUME WARNING##########################################')
            mol_total[i] = mol_total_aq[i] + mol_total_gas[i] + mol_total_solid[i]
            S_w[i] = volume_aq[i] / volume_tot[i]
            S_g[i] = volume_gas[i] / volume_tot[i]
            S_s[i] = volume_solid[i] / volume_tot[i]
            S_w_norm = S_w / (S_g+S_w)
            S_g_norm = S_g / (S_g+S_w)
            #L[i] = (density_aq[i] * S_w[i]) / (density_gas[i] * S_g[i] + density_aq[i] * S_w[i] + density_solid[i] * S_s[i])
            L[i] = (density_aq[i] * S_w_norm[i]) / (density_gas[i] * S_g_norm[i] + density_aq[i] * S_w_norm[i])
            V[i] = (density_gas[i] * S_g[i]) / (density_gas[i] * S_g[i] + density_aq[i] * S_w[i] + density_solid[i] * S_s[i])
            S[i] = (density_solid[i] * S_s[i]) / (density_gas[i] * S_g[i] + density_aq[i] * S_w[i] + density_solid[i] * S_s[i])
            x = np.append(x,[H2O[i]/mol_total_aq[i], CO2[i]/mol_total_aq[i], Ca[i]/mol_total_aq[i], CO3[i]/mol_total_aq[i], CaCO3[i]/mol_total_aq[i]])
            y = np.append(y,[H2O_gas[i]/mol_total_gas[i], CO2_gas[i]/mol_total_gas[i], 0, 0, 0])
            z_c = np.append(z_c,[(H2O[i]+H2O_gas[i])/mol_total[i], (CO2[i]+CO2_gas[i])/mol_total[i], Ca[i]/mol_total[i], CO3[i]/mol_total[i], (CaCO3[i]+Calcite[i])/mol_total[i]])
            density = np.append(density, [density_aq, density_gas, density_solid])
        return L, V, S, x, y, z_c, density


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
            self.kre = 0.50
            self.sr = self.Sgr
            self.sr1 = self.Swc
            self.n = 2
        else:  # water
            self.kre = 0.0
            self.sr = 0
            self.sr1 = 0
            self.n = 2


    def evaluate(self, sat):

        if sat >= 1 - self.sr1:
            kr = self.kre

        elif sat <= self.sr:
            kr = 0

        else:
            # general Brook-Corey
            kr = self.kre * ((sat - self.sr) / (1 - self.Sgr - self.Swc)) ** self.n

        return kr


class kinetic_basic():
    def __init__(self, equi_prod, kin_rate_cte, ne, combined_ions=True):
        self.equi_prod = equi_prod
        self.kin_rate_cte = kin_rate_cte
        self.kinetic_rate = np.zeros(ne)
        self.combined_ions = combined_ions

    def evaluate(self, x, nu_sol):
        if self.combined_ions:
            ion_prod = (x[1][1] / 2) ** 2
            self.kinetic_rate[1] = - self.kin_rate_cte * (1 - ion_prod / self.equi_prod) * nu_sol
            self.kinetic_rate[-1] = - 0.5 * self.kinetic_rate[1]
        else:
            ion_prod = x[1][1] * x[1][2]
            self.kinetic_rate[1] = - self.kin_rate_cte * (1 - ion_prod / self.equi_prod) * nu_sol
            self.kinetic_rate[2] = - self.kin_rate_cte * (1 - ion_prod / self.equi_prod) * nu_sol
            self.kinetic_rate[-1] = - self.kinetic_rate[1]

        return self.kinetic_rate