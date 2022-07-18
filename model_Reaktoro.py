from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import sim_params
import numpy as np
from properties_basic import *
from property_container import *
from reaktoro import *  # reaktoro v2.0.0rc22
from physics_comp_sup import Compositional
import matplotlib.pyplot as plt


class Model(DartsModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.zero = 1e-11
        perm = 100  # / (1 - solid_init) ** trans_exp
        nx = 80
        dx = np.array([0.308641975, 0.617283951, 0.925925926, 1.234567901, 1.543209877, 1.851851852, 2.160493827,
                       2.469135802, 2.777777778, 3.086419753, 3.395061728, 3.703703704, 4.012345679, 4.320987654,
                       4.62962963, 4.938271605, 5.24691358, 5.555555556, 5.864197531, 6.172839506, 6.481481481,
                       6.790123457, 7.098765432, 7.407407407, 7.716049383, 8.024691358, 8.333333333, 8.641975309,
                       8.950617284, 9.259259259, 9.567901235, 9.87654321, 10.18518519, 10.49382716, 10.80246914,
                       11.11111111, 11.41975309, 11.72839506, 12.03703704, 12.34567901, 12.65432099, 12.96296296,
                       13.27160494, 13.58024691, 13.88888889, 14.19753086, 14.50617284, 14.81481481, 15.12345679,
                       15.43209877, 15.74074074, 16.04938272, 16.35802469, 16.66666667, 16.97530864, 17.28395062,
                       17.59259259, 17.90123457, 18.20987654, 18.51851852, 18.82716049, 19.13580247, 19.44444444,
                       19.75308642, 20.0617284, 20.37037037, 20.67901235, 20.98765432, 21.2962963, 21.60493827,
                       21.91358025, 22.22222222, 22.5308642, 22.83950617, 23.14814815, 23.45679012, 23.7654321,
                       24.07407407, 24.38271605, 24.69135802])  # totals 1000
        dy = 100
        dz = 100
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=dx, dy=dy, dz=dz, permx=perm, permy=perm,
                                         permz=perm/10, poro=0.2, depth=2000)

        # """well location"""
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1, j=1, k=1, multi_segment=False)

        self.reservoir.add_well("P1")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=nx, j=1, k=1, multi_segment=False)

        """Physical properties"""
        # Create property containers:
        components_name = ['H2O', 'CO2', 'Na+', 'Cl-', 'Halite']
        elements_name = ['H2O', 'CO2', 'Na+', 'Cl-']
        # E_mat = np.array([[1, 0, 0, 0, 0, 0, 0],
        #                   [0, 1, 0, 0, 0, 0, 1],
        #                   [0, 0, 1, 0, 0, 1, 0],
        #                   [0, 0, 0, 1, 0, 0, 0],
        #                   [0, 0, 0, 0, 1, 1, 2]])
        E_mat = np.array([[1, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 1],
                          [0, 0, 0, 1, 1]])
        # aqueous_phase = ['H2O(aq)', 'CO2(aq)', 'Ca+2', 'CO3-2', 'Na+', 'Cl-']
        # gas_phase = ['H2O(g)', 'CO2(g)']
        # solid_phase = ['Calcite', 'Halite']

        self.thermal = 0
        Mw = [18.015, 44.01, 22.99, 35.45, 58.44]
        # Mw = [18.015, 44.01, 22.99, 35.45, 58.44]
        self.reaktoro = Reaktoro()  # Initialise Reaktoro
        solid_density = np.zeros(1)
        solid_density[0] = 2000  # fill in density for amount of solids present

        self.property_container = model_properties(phases_name=['gas', 'wat', 'sol'],
                                                   components_name=components_name, elements_name=elements_name,
                                                   reaktoro=self.reaktoro, E_mat=E_mat, diff_coef=1e-9, rock_comp=1e-7,
                                                   Mw=Mw, min_z=self.zero / 10, solid_dens=solid_density)
        # self.components = self.property_container.components_name
        # self.elements = self.property_container.elements_name
        # self.phases = self.property_container.phases_name

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

        """ Activate physics """
        self.physics = Compositional(self.property_container, self.timer, n_points=101, min_p=1, max_p=1000,
                                     min_z=self.zero / 10, max_z=1 - self.zero / 10, cache=0)

        #                  H2O,                     CO2,    Ca++,       CO3--,      Na+, Cl-
        # H2O = 1 kg
        H2O = 55.5  # mol
        Na = 4.5  # 3.44 # mol/kgW
        Cl = 4.5  # mol/kgW
        # Ca = 0.8
        # CO3 = 0.8
        ze = [H2O-4*self.zero, 0, Na, Cl]
        ze = [float(i) / sum(ze) for i in ze]
        self.ini_stream = ze[:-1]
        # self.ini_stream = [0.8 - 2 * self.zero, self.zero, 0.08, 0.08, 0.02]
        self.inj_stream = [self.zero, 1 - 6 * self.zero, self.zero]
        # self.inj_stream = self.ini_stream
        ne = self.property_container.nc + self.thermal
        equi_prod = ze[2]**2
        self.property_container.kinetic_rate_ev = kinetic_basic(equi_prod, 1e-0, ne)

        self.params.first_ts = 1e-2
        self.params.max_ts = 10
        self.params.mult_ts = 2

        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-6
        self.params.max_i_newton = 10
        self.params.max_i_linear = 50
        self.params.newton_type = sim_params.newton_local_chop
        # self.params.newton_params[0] = 0.2

        self.timer.node["initialization"].stop()

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        """ initialize conditions for all scenarios"""
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, 392.517, self.ini_stream)

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                # w.control = self.physics.new_rate_inj(0.2, self.inj_stream, 0)
                w.control = self.physics.new_bhp_inj(392.517+100, self.inj_stream)
                w.constraint = self.physics.new_rate_inj(500, self.inj_stream, 0)
                # well constraint max 500m3/day
            else:
                w.control = self.physics.new_bhp_prod(392.517-10)

    def set_op_list(self):
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks
        self.op_num[n_res:] = 1
        self.op_list = [self.physics.acc_flux_itor, self.physics.acc_flux_w_itor]

    def properties(self, state):
        (sat, x, rho, rho_m, mu, kr, ph) = self.property_container.evaluate(state)
        return sat[0]

    def flash_properties(self, ze, T, P):
        nu, x, zc, density, kinetic_rate = Flash_Reaktoro(ze, T, P, self.reaktoro)
        return nu, x, zc, density, kinetic_rate

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
        z_caco3 = 1 - (
                    Xn[1:self.reservoir.nb * nc:nc] + Xn[2:self.reservoir.nb * nc:nc] + Xn[3:self.reservoir.nb * nc:nc])

        z_co2 = Xn[1:self.reservoir.nb * nc:nc] / (1 - z_caco3)
        z_inert = Xn[2:self.reservoir.nb * nc:nc] / (1 - z_caco3)
        z_h2o = Xn[3:self.reservoir.nb * nc:nc] / (1 - z_caco3)

        for ii in range(self.reservoir.nb):
            x_list = Xn[ii * nc:(ii + 1) * nc]
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
            print(
                '//Gridblock\t gas_sat\t pressure\t C_m\t poro\t co2_liq\t co2_vap\t h2o_liq\t h2o_vap\t '
                'ca_plus_co3_liq\t liq_dens\t vap_dens\t solid_dens\t liq_mole_dens\t vap_mole_dens\t solid_mole_dens'
                '\t rel_perm_liq\t rel_perm_gas\t visc_liq\t visc_gas',
                file=f)
            print(
                '//[-]\t [-]\t [bar]\t [kmole/m3]\t [-]\t [-]\t [-]\t [-]\t [-]\t [-]\t [kg/m3]\t [kg/m3]\t [kg/m3]\t '
                '[kmole/m3]\t [kmole/m3]\t [kmole/m3]\t [-]\t [-]\t [cP]\t [cP]',
                file=f)
            for ii in range(self.reservoir.nb):
                print(
                    '{:d}\t {:6.5f}\t {:7.5f}\t {:7.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t '
                    '{:8.5f}\t {:8.5f}\t {:8.5f}\t {:7.5f}\t {:7.5f}\t {:7.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t '
                    '{:6.5f}'.format(
                        ii, Sg[ii], P[ii], Ss[ii] * density_m[ii, 2], 1 - Ss[ii], X[ii, 0, 0], X[ii, 0, 1], X[ii, 2, 0],
                        X[ii, 2, 1], X[ii, 1, 0],
                        density[ii, 0], density[ii, 1], density[ii, 2], density_m[ii, 0], density_m[ii, 1],
                        density_m[ii, 2],
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
    def __init__(self, phases_name, components_name, elements_name, reaktoro, E_mat, Mw, min_z=1e-12,
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

    def run_flash(self, pressure, ze):
        # Make every value that is the min_z equal to 0, as Reaktoro can work with 0, but not transport
        ze = comp_extension(ze, self.min_z)
        self.nu, self.x, zc, density, kinetic_rate = Flash_Reaktoro(ze, 350, pressure, self.reaktoro)
        zc = comp_correction(zc, self.min_z)
        # Solid phase always needs to be present
        ph = list(range(len(self.nu)))  # ph = range(number of total phases)
        min_ph = 0.01  # min value to be considered inside the phase
        for i in range(len(self.nu)):
            if density[i] < 0 and min_ph < 0.1:
                min_ph_new = self.nu[i]
                min_ph = max(min_ph, min_ph_new)
            if density[1] > 1500 and min_ph < 0.1:
                min_ph_new = self.nu[1]
                min_ph = max(min_ph, min_ph_new)
        if self.nu[0] <= min_ph:  # if vapor phase is less than min_z, it does not exist
            del ph[0]  # remove entry of vap
            density[0] = 0
        elif self.nu[1] <= min_ph:  # if liq phase is less than min_z, it does not exist
            del ph[1]
            density[1] = 0
        # solid phase always present
        for i in range(len(self.nu)):
            if i > 1 and self.nu[i] < self.min_z:
                self.nu[i] = self.min_z
        # for i in range(len(self.nu)):
        #     if density[i] < 0 or density[1] > 2000:
        #         print('Partial molar problems, likely, density is below 0 or above 2000 for aqueous phase')
        #         print('ze', ze)
        #         print('zc', zc)
        #         print('nu', self.nu)
        #         print(density)
        return ph, zc, density


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


def Flash_Reaktoro(z_e, T, P, reaktoro):
    if z_e[2] != z_e[3]:
        ze_new = (z_e[2]+z_e[3])/2
        z_e[2] = ze_new
        z_e[3] = ze_new
    reaktoro.addingproblem(T, P, z_e)
    nu, x, z_c, density, kinetic_rate = reaktoro.output()  # z_c order is determined by user, check if its the same order as E_mat
    return nu, x, z_c, density, kinetic_rate


class Reaktoro:
    def __init__(self):
        db = SupcrtDatabase("supcrtbl")
        '''Hardcode'''
        self.gas_comp = StringList(["H2O(g)", "CO2(g)"])
        self.aq_comp = StringList(['H2O(aq)', 'CO2(aq)', 'Na+', 'Cl-'])
        self.sol_comp = ['Halite']
        gas = GaseousPhase(self.gas_comp)
        aq = AqueousPhase(self.aq_comp)
        for i in range(len(self.sol_comp)):
            globals()['sol%s' % i] = MineralPhase(self.sol_comp[i])
        self.system = ChemicalSystem(db, gas, aq, sol0)
        self.solver = EquilibriumSolver(self.system)
        self.cp = type(object)
        # self.specs.pH()
        # self.specs.charge()
        # self.specs.openTo("Na+")

    def addingproblem(self, temp, pres, z_e):
        state = ChemicalState(self.system)
        state.temperature(temp, 'kelvin')
        state.pressure(pres, 'bar')
        for i in range(self.aq_comp.size()):
            state.set(self.aq_comp[i], z_e[i], 'mol')
        # conditions.charge(0)
        result = self.solver.solve(state)
        if not result.optima.succeeded:
            print('Reaktoro did not find solution')
        self.cp = ChemicalProps(state)

    def output(self):
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
        solid = self.cp.speciesAmount('Halite')
        Na = self.cp.speciesAmount('Na+')
        Cl = self.cp.speciesAmount('Cl-')

        total_mol = self.cp.amount()
        total_mol_sol = sol_props.amount()

        mol_frac_gas = gas_props.speciesMoleFractions()
        mol_frac_aq = liq_props.speciesMoleFractions()

        '''Hardcode'''
        mol_frac_gas = [float(mol_frac_gas[0]), float(mol_frac_gas[1]), 0, 0, 0]
        mol_frac_aq = [float(mol_frac_aq[0]), float(mol_frac_aq[1]), float(mol_frac_aq[2]), float(mol_frac_aq[3]),
                       0]
        mol_frac_sol = [0, 0, 0, 0, float(solid / total_mol_sol)]
        assert len(mol_frac_gas) == len(mol_frac_aq) and len(mol_frac_aq) == len(mol_frac_sol), \
            'mol frac should be same length'

        # Partial molar volume equation: V_tot = total_mol * sum(molar_frac*partial mole volume)
        # partial_mol_vol_aq = np.zeros(len(mol_frac_aq))
        # for i in range(len(mol_frac_aq)):
        #     partial_mol_vol_aq[i] = float(liq_props.speciesStandardVolumes()[i])
        # volume_aq = total_mol_aq * np.sum(np.multiply(mol_frac_aq, partial_mol_vol_aq))

        volume_gas = gas_props.volume()
        volume_aq = liq_props.volume()
        volume_solid = sol_props.volume()
        volume_tot = self.cp.volume()

        density_gas = gas_props.density()
        density_aq = liq_props.density()
        density_solid = sol_props.density()

        S_g = volume_gas / volume_tot
        S_w = volume_aq / volume_tot
        S_s = volume_solid / volume_tot

        V = (density_gas * S_g) / (density_gas * S_g + density_aq * S_w + density_solid * S_s)
        L = (density_aq * S_w) / (density_gas * S_g + density_aq * S_w + density_solid * S_s)
        S = (density_solid * S_s) / (density_gas * S_g + density_aq * S_w + density_solid * S_s)
        nu = [float(V), float(L), float(S)]
        x = [mol_frac_gas, mol_frac_aq, mol_frac_sol]

        '''Hardcode'''
        z_c = [float(H2O / total_mol), float(CO2 / total_mol),
               float(Na / total_mol), float(Cl / total_mol),
               float(solid / total_mol)]

        density = [float(density_gas), float(density_aq), float(density_solid)]
        activity = self.cp.speciesActivity('Na+') * self.cp.speciesActivity('Cl-')
        H2O_litre = liq_props.speciesStandardVolumes()[0]*H2O_aq * 1000  # V=\hat{V}*n  m3 -> litre
        equilibriumconstant = 1 / (Na/H2O_litre + Cl/H2O_litre)
        A = (1-0.2)*S
        kinetic_rate = A*1*(1-(float(activity)/equilibriumconstant))
        return nu, x, z_c, density, kinetic_rate