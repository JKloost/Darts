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
        self.E_mat = property_container.E_mat


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

        nm = self.property.nm  # number of minerals
        nc_fl = nc - nm  # number of fluids (aq + gas)
        neq = ne + self.thermal  # number of equations
        log_flag = self.property.log_flag

        # Total needs to be total of element based, as this will be the size of values
        #       al + bt        + gm + dlt + chi     + rock_temp por    + gr/cap  + por
        total = neq + neq * nph + nph + neq + neq * nph + 3 + 2 * nph + 1  # Element based

        for i in range(total):
            values[i] = 0
        #  some arrays will be reused in thermal
        (self.sat, self.x, rho, self.rho_m, self.mu, self.kr, self.pc, self.ph) = self.property.evaluate(state)

        self.compr = (1 + self.property.rock_comp * (pressure - self.property.p_ref))  # compressible rock
        if log_flag == 1:
            ze = np.append(np.exp(vec_state_as_np[1:ne]), 1 - np.sum(np.exp(vec_state_as_np[1:ne])))
        else:
            ze = np.append(vec_state_as_np[1:ne], 1 - np.sum(vec_state_as_np[1:ne]))
        rho_t = np.sum(np.multiply(rho, self.sat))

        phi = 1 # - (self.sat[-1] * rho[-1])/rho_t  # * (self.x[-1])  # - rho_t * self.sat[-1]

        """ CONSTRUCT OPERATORS HERE """  # need to do matrix vector multiplication

        """ Alpha operator represents accumulation term """
        chi = np.zeros(nph*nc)
        density_tot_e = np.zeros(nph)
        for j in range(nph):
            for i in range(ne):
                density_tot_e[j] = np.sum((self.sat[j] * self.rho_m[j]) * np.sum(np.multiply(self.E_mat, self.x[j])))
        for i in range(ne):
            values[i] = self.compr * ze[i] * sum(density_tot_e) * phi  # z_e uncorrected

        """ Beta operator represents flux term: """  # Here we can keep nc_fl
        for j in self.ph:
            shift = neq + neq * j   # e.g. ph = [0,2], shift is multiplied by 0 and 2
            beta = np.zeros(nc)
            for i in range(nc_fl):
                beta[i] = self.x[j][i] * self.rho_m[j] * self.kr[j] / self.mu[j]
                # values[shift + i] = self.x[j][i] * self.rho_m[j] * self.kr[j] / self.mu[j]
            for i in range(self.E_mat.shape[0]):
                values[shift+i] = np.sum(np.multiply(self.E_mat[i], beta[i]))
            # exit()

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        shift = neq + neq * nph
        for j in self.ph:
            values[shift + j] = self.compr * self.kr[j]

        """ Chi operator for diffusion """
        shift += nph
        for i in range(nc):
            for j in self.ph:
                chi[i*nph+j] = self.property.diff_coef * self.x[j][i] * self.rho_m[j]
                # values[shift + i * nph + j] = self.property.diff_coef * self.x[j][i] * self.rho_m[j]
        for i in range(self.E_mat.shape[0]):
            for j in self.ph:
                values[shift+i*nph+j] = np.sum(np.multiply(self.E_mat[i], chi[i*nph+j]))

        """ Delta operator for reaction """
        shift += nph * neq
        # if self.property.kinetic_rate_ev:
        kinetic_rate = [0, 0, 0, 0, -self.property.kin_rate, -self.property.kin_rate, 0, self.property.kin_rate]
            # kinetic_rate = self.property.kinetic_rate_ev.evaluate(self.x, zc[4:])
            # kinetic_rate = self.property.kinetic_rate_ev.evaluate(self,zc)
            #kinetic_rate = [0,0,0,0]
            # kinetic_rate = self.property.kinetic_rate
        for i in range(neq):
            values[shift + i] = np.sum(np.multiply(self.E_mat[i], kinetic_rate[i]))

            # print(kinetic_rate)
                # values[shift+i] = 0

        """ Gravity and Capillarity operators """
        shift += neq
        # E3-> gravity
        for i in self.ph:
            values[shift + 3 + i] = rho[i]  # 3 = thermal operators

        # E4-> capillarity
        for i in self.ph:
            values[shift + 3 + nph + i] = self.pc[i]
        # E5_> porosity
        values[shift + 3 + 2 * nph] = phi

        # print(pressure, ze)
        # print(self.x[0])
        # print('.')

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
        self.E_mat = property_container.E_mat

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
        nm = self.property.nm
        nc_fl = nc - nm
        neq = ne + self.thermal


        #       al + bt        + gm + dlt + chi     + rock_temp por    + gr/cap  + por
        total = neq + neq * nph + nph + neq + neq * nph + 3 + 2 * nph + 1

        for i in range(total):
            values[i] = 0

        (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property.evaluate(state)

        self.compr = (1 + self.property.rock_comp * (pressure - self.property.p_ref))  # compressible rock
        log_flag = self.property.log_flag
        if log_flag == 1:
            ze = np.append(np.exp(vec_state_as_np[1:ne]), 1 - np.sum(np.exp(vec_state_as_np[1:ne])))
        else:
            ze = np.append(vec_state_as_np[1:ne], 1 - np.sum(vec_state_as_np[1:ne]))

        density_tot = np.sum(sat * rho_m)
        density_tot_e = np.zeros(nph)
        for j in range(nph):
            for i in range(ne):
                density_tot_e[j] = np.sum((sat[j] * rho_m[j]) * np.sum(np.multiply(self.E_mat, x[j])))
        # zc = np.append(vec_state_as_np[1:nc], 1 - np.sum(vec_state_as_np[1:nc]))
        rho_t = np.sum(np.multiply(rho, sat))
        phi = 1 # - rho_t * sat[-1]
        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term """
        beta = np.zeros(nc)

        for i in range(ne):
            values[i] = self.compr * ze[i] * sum(density_tot_e)  # z_e uncorrected

        """ Beta operator represents flux term: """
        for j in ph:
            shift = neq + neq * j
            for i in range(nc_fl):
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
        if self.property.kinetic_rate_ev:
            # kinetic_rate = self.property.kinetic_rate_ev.evaluate(self.x, zc)
            # kinetic_rate = self.property.kinetic_reaktoro.evaluate(
            kinetic_rate = [0, 0, 0, 0]
            # kinetic_rate = self.property.kinetic_rate
            for i in range(neq):
                values[shift + i] = 0
                # values[shift + i] = 0

        """ Gravity and Capillarity operators """
        shift += neq
        # E3-> gravity
        for i in ph:
            values[shift + 3 + i] = rho[i]

        # E4-> capillarity
        #for i in ph:
        #    values[shift + 3 + nph + i] = pc[i]

        # E5_> porosity
        values[shift + 3 + 2 * nph] = phi

        return 0

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

        (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property.evaluate(state)

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