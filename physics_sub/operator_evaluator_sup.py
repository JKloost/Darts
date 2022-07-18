import numpy as np
from darts.engines import *
from darts.physics import *

import os.path as osp

physics_name = osp.splitext(osp.basename(__file__))[0]

# Define our own operator evaluator class
class ReservoirOperators(operator_set_evaluator_iface):
    def __init__(self, property_container, thermal=0):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.min_z = property_container.min_z
        self.property = property_container
        self.thermal = thermal
        #self.E_mat =
        self.E_mat = np.array([[1, 0, 0, 0, 0],      # elimination matrix, to transform comp to elem
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 1],
              [0, 0, 0, 1, 1]])
        # self.E_mat = np.array([[1, 0, 0, 0, 0, 0, 0, 0],      # elimination matrix, to transform comp to elem
        #       [0, 1, 0, 0, 0, 0, 0, 0],
        #       [0, 0, 1, 0, 0, 0, 1, 0],
        #       [0, 0, 0, 1, 0, 0, 1, 0],
        #       [0, 0, 0, 0, 1, 0, 0, 1],
        #       [0, 0, 0, 0, 0, 1, 0, 1]])
        # self.E_mat = np.array([[1,0,0],
        #                        [0,1,0],
        #                        [0,0,1]])
        # self.E_mat = np.array([[1, 0, 0, 0, 0, 0, 0],      # elimination matrix, to transform comp to elem
        #       [0, 1, 0, 0, 0, 0, 0],
        #       [0, 0, 1, 0, 0, 1, 0],
        #       [0, 0, 0, 1, 0, 0, 1],
        #       [0, 0, 0, 0, 1, 0, 1]])
        # self.E_mat = np.array([[1, 0, 0, 0, 0, 0],
        #                        [0, 1, 0, 0, 0, 0],
        #                        [0, 0, 1, 0, 0, 0],
        #                        [0, 0, 0, 1, 0, 0],
        #                        [0, 0, 0, 0, 1, 0],
        #                        [0, 0, 0, 0, 0, 1]])
        # self.E_mat = np.array([[1, 0, 0, 0, 0, 0, 0],
        #                        [0, 1, 0, 0, 0, 0, 1],
        #                        [0, 0, 1, 0, 0, 1, 0],
        #                        [0, 0, 0, 1, 0, 0, 0],
        #                        [0, 0, 0, 0, 1, 1, 1]])
        # self.E_mat = np.array([[1, 0, 0, 0, 0],
        #                        [0, 1, 0, 0, 1],
        #                        [0, 0, 1, 0, 0],
        #                        [0, 0, 0, 1, 1]])
        # self.E_mat = np.array([[1, 0, 0, 0],
        #                        [0, 1, 0, 1],
        #                        [0, 0, 1, 1]])


    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        nc = self.property.nc  # number components
        ne = self.property.n_e  # number of elements
        nph = self.property.nph  # number of phases

        # nm = self.property.nm  # number of minerals
        # nc_fl = nc - nm  # number of fluids (aq + gas)
        neq = ne + self.thermal  # number of equations

        # Total needs to be total of element based, as this will be the size of values
        #       al + bt        + gm + dlt + chi     + rock_temp por    + gr/cap  + por
        total = neq + neq * nph + nph + neq + neq * nph + 3 + 2 * nph + 1  # Element based

        for i in range(total):
            values[i] = 0
        # values = np.zeros(total)
        #  some arrays will be reused in thermal
        (self.sat, self.x, rho, self.rho_m, self.mu, self.kr, self.pc, self.ph, zc, kinetic_rate) = self.property.evaluate(state)
        # norm = False
        # for i in range(len(zc)):
        #     if zc[i] < 1e-11:
        #         zc[i] = 0
        #         norm = True
        # if norm == True:
        #     zc = [float(q) / sum(zc) for q in zc]
        self.compr = (1 + self.property.rock_comp * (pressure - self.property.p_ref))  # compressible rock
        ze = np.append(vec_state_as_np[1:ne], 1 - np.sum(vec_state_as_np[1:ne]))

        # ze = np.zeros(self.E_mat.shape[0])
        #for i in range(self.E_mat.shape[0]):
        #    ze[i] = np.divide(np.sum(np.multiply(self.E_mat[i], zc)), np.sum(np.multiply(self.E_mat, zc)))  # ze e_i z - Ez
        density_tot = np.sum(self.sat * self.rho_m)
        #density_tot_c = np.zeros(nph)
        #for i in range(nph):
        #    density_tot_c[i] = self.sat[i]*self.rho_m[i]
        # zc = np.append(vec_state_as_np[1:nc], 1 - np.sum(vec_state_as_np[1:nc]))
        # zc = vec_state_as_np[1:]    # We receive the components from reaktoro
        phi = 1
        """ CONSTRUCT OPERATORS HERE """  # need to do matrix vector multiplication

        """ Alpha operator represents accumulation term """
        alpha = np.zeros(nc)
        beta = np.zeros(nc)
        chi = np.zeros(nph*nc)

        density_tot_e = np.zeros(nph)
        for j in range(nph):
            for i in range(ne):
                density_tot_e[j] = np.sum((self.sat[j] * self.rho_m[j]) * np.sum(np.multiply(self.E_mat, self.x[j])))
        # for i in range(nc):
        #     alpha[i] = self.compr * zc[i] * density_tot
        # for i in range(self.E_mat.shape[0]):
        #     values[i] = np.sum(np.multiply(self.E_mat[i], alpha[i]))
        #     print(np.sum(np.multiply(self.E_mat[i], alpha[i])))
        for i in range(ne):
            values[i] = self.compr * ze[i] * sum(density_tot_e)  # z_e uncorrected
        # print(zc)
        # print(ze)
        # print(sum(density_tot))
        # exit()
        # print('alpha', alpha)
        """ and alpha for mineral components """
        # for i in range(nm):
        #     values[i + nc_fl] = self.property.solid_dens[i] * zc[i + nc_fl]

        """ Beta operator represents flux term: """  # Here we can keep nc_fl
        for j in self.ph:
            # print('beta',beta)
            shift = neq + neq * j   # e.g. ph = [0,2], shift is multiplied by 0 and 2
            # print('betashift',shift)
            beta = np.zeros(nc)
            for i in range(nc):
                beta[i] = self.x[j][i] * self.rho_m[j] * self.kr[j] / self.mu[j]
                # values[shift + i] = self.x[j][i] * self.rho_m[j] * self.kr[j] / self.mu[j]
            for i in range(self.E_mat.shape[0]):
                values[shift+i] = np.sum(np.multiply(self.E_mat[i], beta[i]))

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        shift = neq + neq * nph
        # print('gammashift',shift)
        for j in self.ph:
            gamma = self.compr * self.sat[j]
            values[shift + j] = self.compr * self.kr[j]
            # print('gamma', gamma)

        """ Chi operator for diffusion """
        shift += nph
        # print('chishift',shift)
        for i in range(nc):
            for j in self.ph:
                chi[i*nph+j] = self.property.diff_coef * self.x[j][i] * self.rho_m[j]
                # print('chi', chi)
                # values[shift + i * nph + j] = self.property.diff_coef * self.x[j][i] * self.rho_m[j]
        for i in range(self.E_mat.shape[0]):
            for j in self.ph:
                values[shift+i*nph+j] = np.sum(np.multiply(self.E_mat[i], chi[i*nph+j]))


        """ Delta operator for reaction """
        shift += nph * neq
        # print('deltashift',shift)
        if self.property.kinetic_rate_ev:
            # kinetic_rate = self.property.kinetic_rate_ev.evaluate(self.x, zc[4:])
            # kinetic_rate = self.property.kinetic_rate_ev.evaluate(self,zc)
            # kinetic_rate = [0,0,0,0]
            for i in range(neq):
                # values[shift + i] = kinetic_rate[i]
                values[shift+i] = 0

        """ Gravity and Capillarity operators """
        shift += neq
        # print('gravshift',shift)
        # E3-> gravity
        for i in self.ph:
            values[shift + 3 + i] = rho[i]  # 3 = thermal operators

        # E4-> capillarity
        for i in self.ph:
            values[shift + 3 + nph + i] = self.pc[i]
        # E5_> porosity
        values[shift + 3 + 2 * nph] = phi

        # print('reservoir', state, values)
        # exit()
        return 0

class WellOperators(operator_set_evaluator_iface):
    def __init__(self, property_container, thermal=0):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.nc = property_container.nc
        self.nph = property_container.nph
        self.min_z = property_container.min_z
        self.property = property_container
        self.thermal = thermal
        self.E_mat = np.array([[1, 0, 0, 0, 0],  # elimination matrix, to transform comp to elem
                               [0, 1, 0, 0, 0],
                               [0, 0, 1, 0, 1],
                               [0, 0, 0, 1, 1]])
        # self.E_mat = np.array([[1, 0, 0, 0, 0, 0, 0, 0],  # elimination matrix, to transform comp to elem
        #                        [0, 1, 0, 0, 0, 0, 0, 0],
        #                        [0, 0, 1, 0, 0, 0, 1, 0],
        #                        [0, 0, 0, 1, 0, 0, 1, 0],
        #                        [0, 0, 0, 0, 1, 0, 0, 1],
        #                        [0, 0, 0, 0, 0, 1, 0, 1]])
        # self.E_mat = np.array([[1, 0, 0],
        #                        [0, 1, 0],
        #                        [0, 0, 1]])
        # self.E_mat = np.array([[1, 0, 0, 0, 0, 0],
        #                        [0, 1, 0, 0, 0, 0],
        #                        [0, 0, 1, 0, 0, 0],
        #                        [0, 0, 0, 1, 0, 0],
        #                        [0, 0, 0, 0, 1, 0],
        #                        [0, 0, 0, 0, 0, 1]])
        # self.E_mat = np.array([[1, 0, 0, 0, 0, 0, 0],
        #                        [0, 1, 0, 0, 0, 0, 1],
        #                        [0, 0, 1, 0, 0, 1, 0],
        #                        [0, 0, 0, 1, 0, 0, 0],
        #                        [0, 0, 0, 0, 1, 1, 1]])
        # self.E_mat = np.array([[1, 0, 0, 0],
        #                        [0, 1, 0, 1],
        #                        [0, 0, 1, 1]])
        # self.E_mat = np.array([[1, 0, 0, 0, 0],
        #                        [0, 1, 0, 0, 1],
        #                        [0, 0, 1, 0, 0],
        #                        [0, 0, 0, 1, 1]])

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        nc = self.property.nc
        ne = self.property.n_e
        nph = self.property.nph
        #nm = self.property.nm
        #nc_fl = nc - nm
        neq = ne + self.thermal

        #       al + bt        + gm + dlt + chi     + rock_temp por    + gr/cap  + por
        total = neq + neq * nph + nph + neq + neq * nph + 3 + 2 * nph + 1

        for i in range(total):
            values[i] = 0

        (sat, x, rho, rho_m, mu, kr, pc, ph, zc, kinetic_rate) = self.property.evaluate(state)
        # norm = False
        # for i in range(len(zc)):
        #     if zc[i] < 1e-12:
        #         zc[i] = 1e-12
        #         norm = True
        # if norm == True:
        #     zc = [float(q) / sum(zc) for q in zc]
        self.compr = (1 + self.property.rock_comp * (pressure - self.property.p_ref))  # compressible rock
        ze = np.append(vec_state_as_np[1:ne], 1 - np.sum(vec_state_as_np[1:ne]))

        density_tot = np.sum(sat * rho_m)
        density_tot_e = np.zeros(nph)
        for j in range(nph):
            for i in range(ne):
                density_tot_e[j] = np.sum((sat[j] * rho_m[j]) * np.sum(np.multiply(self.E_mat, x[j])))
        # zc = np.append(vec_state_as_np[1:nc], 1 - np.sum(vec_state_as_np[1:nc]))
        phi = 1
        """ CONSTRUCT OPERATORS HERE """  # need to do matrix vector multiplication

        """ Alpha operator represents accumulation term """
        alpha = np.zeros(nc)
        beta = np.zeros(nc)
        chi = np.zeros(nph * nc)

        # for i in range(nc):
        #     # values[i] = self.compr * density_tot * zc[i]
        #     alpha[i] = self.compr * density_tot * zc[i]
        # for i in range(self.E_mat.shape[0]):
        #     values[i] = np.sum(np.multiply(self.E_mat[i], alpha[i]))
        for i in range(ne):
            values[i] = self.compr * ze[i] * sum(density_tot_e)  # z_e uncorrected
        # print('alpha', alpha)
        """ and alpha for mineral components """
        # for i in range(nm):
        #    values[i + nc_fl] = self.property.solid_dens[i] * zc[i + nc_fl]

        """ Beta operator represents flux term: """  # Here we can keep nc_fl
        for j in ph:
            # print('beta',beta)
            shift = neq + neq * j  # e.g. ph = [0,2], shift is multiplied by 0 and 2
            # print('betashift',shift)
            for i in range(nc):
                beta[i] = x[j][i] * rho_m[j] * kr[j] / mu[j]
                # values[shift + i] = self.x[j][i] * self.rho_m[j] * self.kr[j] / self.mu[j]
            for i in range(self.E_mat.shape[0]):
                values[shift + i] = np.sum(np.multiply(self.E_mat[i], beta[i]))

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        shift = neq + neq * nph

        """ Chi operator for diffusion """
        shift += nph

        """ Delta operator for reaction """
        shift += nph * neq
        # print('deltashift',shift)
        if self.property.kinetic_rate_ev:
            # kinetic_rate = self.property.kinetic_rate_ev.evaluate(self.x, zc)
            # kinetic_rate = self.property.kinetic_reaktoro.evaluate(
            # kinetic_rate = [0,0,0,0]
            for i in range(neq):
                # values[shift + i] = kinetic_rate[i]
                values[shift + i] = 0

        """ Gravity and Capillarity operators """
        shift += neq
        # print('gravshift',shift)
        # E3-> gravity
        for i in ph:
            values[shift + 3 + i] = rho[i]

        # E4-> capillarity
        #for i in ph:
        #    values[shift + 3 + nph + i] = pc[i]
        # E5_> porosity
        values[shift + 3 + 2 * nph] = phi

        # print('well',state, values)
        # print(ph)
        # exit()
        return 0

        # """ CONSTRUCT OPERATORS HERE """
        #
        # """ Alpha operator represents accumulation term """
        # for i in range(nc_fl):
        #     values[i] = self.compr * density_tot * zc[i]
        #
        # """ and alpha for mineral components """
        # for i in range(nm):
        #     values[i + nc_fl] = self.property.solid_dens[i] * zc[i + nc_fl]
        #
        # """ Beta operator represents flux term: """
        # for j in ph:
        #     shift = ne + ne * j
        #     for i in range(nc):
        #         values[shift + i] = x[j][i] * rho_m[j] * sat[j] / mu[j]
        #
        # """ Gamma operator for diffusion (same for thermal and isothermal) """
        # shift = ne + ne * nph
        #
        # """ Chi operator for diffusion """
        # shift += nph
        #
        # """ Delta operator for reaction """
        # shift += nph * ne
        # if self.property.kinetic_rate_ev:
        #     kinetic_rate = self.property.kinetic_rate_ev.evaluate(x, zc[nc_fl:])
        #     for i in range(ne):
        #         values[shift + i] = kinetic_rate[i]
        #
        # """ Gravity and Capillarity operators """
        # shift += ne
        # # E3-> gravity
        # for i in range(nph):
        #     values[shift + 3 + i] = rho[i]
        #
        # # E5_> porosity
        # values[shift + 3 + 2 * nph] = phi
        #
        # #print(state, values)
        # return 0

class RateOperators(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.nc = property_container.nc
        self.ne = property_container.n_e
        self.nph = property_container.nph
        self.min_z = property_container.min_z
        self.property = property_container
        self.flux = np.zeros(self.nc)

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        for i in range(self.nph):
            values[i] = 0

        (sat, x, rho, rho_m, mu, kr, pc, ph, zc, kinetic_rate) = self.property.evaluate(state)


        self.flux[:] = 0
        # step-1
        for j in ph:
            for i in range(self.ne):
                self.flux[i] += rho_m[j] * kr[j] * x[j][i] / mu[j]
        # step-2
        flux_sum = np.sum(self.flux)

        #(sat_sc, rho_m_sc) = self.property.evaluate_at_cond(1, self.flux/flux_sum)
        sat_sc = sat
        rho_m_sc = rho_m

        # step-3
        total_density = np.sum(sat_sc * rho_m_sc)
        # step-4
        for j in ph:
            values[j] = sat_sc[j] * flux_sum / total_density

        # print(state, values)
        return 0


# Define our own operator evaluator class
class ReservoirThermalOperators(ReservoirOperators):
    def __init__(self, property_container, thermal=1):
        super().__init__(property_container, thermal=thermal)  # Initialize base-class

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        super().evaluate(state, values)

        vec_state_as_np = np.asarray(state)
        pressure = state[0]
        temperature = vec_state_as_np[-1]

        # (enthalpy, rock_energy) = self.property.evaluate_thermal(state)
        (enthalpy, cond, rock_energy) = self.property.evaluate_thermal(state)

        nc = self.property.nc
        ne = self.property.n_e
        nph = self.property.nph
        neq = ne + self.thermal

        i = nc  # use this numeration for energy operators
        """ Alpha operator represents accumulation term: """
        for m in self.ph:
            values[i] += self.compr * self.sat[m] * self.rho_m[m] * enthalpy[m]  # fluid enthalpy (kJ/m3)
        values[i] -= self.compr * 100 * pressure

        """ Beta operator represents flux term: """
        for j in self.ph:
            shift = neq + neq * j
            values[shift + i] = enthalpy[j] * self.rho_m[j] * self.kr[j] / self.mu[j]

        """ Chi operator for temperature in conduction, gamma operators are skipped """
        shift = neq + neq * nph + nph
        for j in range(nph):
            # values[shift + nc * nph + j] = temperature
            values[shift + neq * j + ne] = temperature * cond[j]

        """ Delta operator for reaction """
        shift += nph * neq
        values[shift + i] = 0

        """ Additional energy operators """
        shift += ne
        # E1-> rock internal energy
        values[shift] = rock_energy / self.compr  # kJ/m3
        # E2-> rock temperature
        values[shift + 1] = temperature
        # E3-> rock conduction
        values[shift + 2] = 1 / self.compr  # kJ/m3

        # print(state, values)

        return 0


class DefaultPropertyEvaluator(operator_set_evaluator_iface):
    def __init__(self, property_container, thermal=0):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.min_z = property_container.min_z
        self.property = property_container
        self.thermal = thermal
        self.n_ops = self.property.nph

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:

        nph = self.property.nph

        #  some arrays will be reused in thermal
        (self.sat, self.x, rho, self.rho_m, self.mu, self.kr, self.pc, self.ph, zc, kinetic_rate) = self.property.evaluate(state)

        for i in range(nph):
            values[i] = self.sat[i]

        return