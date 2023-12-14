from typing import Union, Sequence
from math import exp, floor, log
import numpy as np


class SolarCell_new:
    def __init__(self,
                 cell_num: int = 36,
                 mesh_points_num: int = 1000,

                 Voc: float = 22.0,
                 Isc: float = 3.45,

                 Pmpp: float = 55,
                 Impp: float = 3.15,
                 Vmpp: float = 17.4,

                 experimental_V: Sequence[int] = (),
                 experimental_I: Sequence[Union[float, int]] = (),

                 Ku: Union[float, int] = 1.0,
                 Ki: Union[float, int] = 1.0,
                 Kp: Union[float, int] = 1.0,

                 T: Union[float, int] = 25,
                 Tstc: Union[float, int] = 25,

                 G: Union[float, int] = 1000,
                 Gstc: Union[float, int] = 1000,

                 custom_thermal_voltage_func=None,
                 custom_photovoltaic_current_func=None,
                 custom_backward_current_func=None,

                 thermal_coefficient_type="relative",
                 is_experimental_data_provided=False,

                 a1=1.0, a2=1.2,

                 I0=None, Ipv=None,
                 start_slope=None,
                 end_slope=None,
                 Vt=None,

                 fixed_point_method_tolerance=10 * 10 ** -12,
                 brute_force_steps=50,
                 mode="overall",
                 approx_range_minimization=None
                 ):

        if thermal_coefficient_type in ("relative", "absolute"):
            self.thermal_coefficient_type = thermal_coefficient_type
        else:
            raise ValueError("Unknown type of 'thermal_coefficient_type' parameter")

        # Custom functions
        self.custom_thermal_voltage_func = custom_thermal_voltage_func
        self.custom_photovoltaic_current_func = custom_photovoltaic_current_func
        self.custom_backward_current_func = custom_backward_current_func

        self.cell_num = cell_num

        # Constants
        self.k = 1.3806503 * 10 ** -23
        self.q = 1.60217646 * 10 ** -19

        # STC parameters
        self.Gstc = Gstc
        self.Tstc = Tstc + 273

        # Environmental conditions
        self.G = G
        self.T = T + 273

        # Experimental data
        self.e_current = experimental_I
        self.e_voltage = experimental_V

        # Solar cell parameters
        self.Voc = Voc
        self.Isc = Isc
        self.Pmpp = Pmpp
        self.Impp = Impp
        self.Vmpp = Vmpp

        self.Ku = Ku
        self.Ki = Ki
        self.Kp = Kp
        self.dT = self.T - self.Tstc

        # Model parameters
        self.I0 = I0
        self.Ipv = Ipv
        self.start_slope = start_slope
        self.end_slope = end_slope
        self.Rs = None
        self.Rp = None
        self.Vt = self.find_thermal_voltage() if Vt is None else Vt

        # Impurity coefficients
        self.a1 = a1
        self.a2 = a2

        """if self.Vt is None:
            self.Vt = self.find_thermal_voltage()
        self.Ipv_estimate = self.find_photovoltaic_current_estimate()
        self.I0_estimate = self.find_backward_current_estimate()
        """

        # Algorithm's variables
        self.is_experimental_data_provided = is_experimental_data_provided
        self.fixed_point_method_tolerance = fixed_point_method_tolerance
        self.mesh_points_num = mesh_points_num
        self.brute_force_steps = brute_force_steps
        self.brute_force_range = 0.3
        self.mode = mode
        self.eta = 0.2
        self.approx_range_minimization = floor(
            len(self.e_voltage) / 10) if approx_range_minimization is None else approx_range_minimization

        # Approximation
        self.I_0 = self.e_current[0]
        self.V_0 = self.e_voltage[-1]

        if is_experimental_data_provided:
            if 0 in self.e_voltage:
                self.voltage = self.e_voltage
            else:
                self.voltage = [0] + self.e_voltage
        else:
            self.create_voltage_mesh()

        self.start_fit_points_number = floor(len(self.voltage) * 0.22)
        self.end_fit_points_number = floor(len(self.voltage) * 0.085)

        self.area1_2 = int(0.55 * len(self.voltage))
        self.area2_3 = int(0.9 * len(self.voltage))

        if self.start_slope is None:
            self.start_slope = self.find_slope("start")
        if self.end_slope is None:
            self.end_slope = self.find_slope("end")

        self.beta = self.find_beta()

        # Algorithm parameters
        self.current = None
        self.approx_error = None
        if self.Ipv is None:
            self.Ipv = self.find_photovoltaic_current()

        if self.I0 is None:
            self.Rs, self.Rp, self.I0 = self.find_parameters()

    def res_enum(self, weight_arr=None):
        self.current, self.Rs, self.Rp, self.approx_error = self.find_best_fit(weight_arr)
        mape = self.find_mape(self.e_current, self.current) # weight_arr)
        print(f"\nRMSE = {self.approx_error:.5f}")
        print(f"MAPE = {mape:.5f}")
        print(f"Prod = {mape * self.approx_error:.10f}")

    def alpha_enum(self, weight_arr=None, use_one_diode_model=True, recalculate=False):
        self.current, *_, self.approx_error = self.find_best_fit_1(weight_arr, use_one_diode_model, recalculate)
        mape = self.find_mape(self.e_current, self.current)
        print(f"\nRMSE = {self.approx_error:.5f}")
        print(f"MAPE = {mape:.5f}")
        print(f"Prod = {mape * self.approx_error:.10f}")

    def recalc(self, Rs, Rp, G, voltage=None):
        self.G = G
        self.Ipv = self.find_photovoltaic_current()
        self.current = self.fixed_point_method(Rs, Rp, voltage)
        return self.current

    def create_voltage_mesh(self) -> Sequence[Union[float, int]]:
        mesh_center_index = self.mesh_points_num // 2
        return np.concatenate(
            (
                np.linspace(0, self.Vmpp, mesh_center_index, endpoint=False),
                np.linspace(self.Vmpp, self.Voc, self.mesh_points_num - mesh_center_index)
            )
        )

    def find_thermal_voltage(self) -> Union[float, int]:
        if self.custom_thermal_voltage_func is None:
            return self.cell_num * self.k * self.T / self.q
        else:
            return self.custom_thermal_voltage_func()

    def find_slope(self, slope_point='start') -> Union[float, int]:
        if slope_point == 'start':
            k, b = np.polyfit(self.voltage[:self.start_fit_points_number:],
                              self.e_current[:self.start_fit_points_number:], 1)
            return k
        if slope_point == 'end':
            a, b, c = np.polyfit(self.voltage[-self.end_fit_points_number::],
                                 self.e_current[-self.end_fit_points_number::], 2)
            return a * self.V_0 * 2 + b

    def find_beta(self):
        self.beta = (exp(self.V_0 / self.Vt / self.a1) +
                     exp(self.V_0 / self.Vt / self.a2)) / \
                    (exp(self.V_0 / self.Vt / self.a1) +
                     exp(self.V_0 / self.Vt / self.a2) - 2)
        return self.beta

    def find_photovoltaic_current(self) -> Union[float, int]:
        self.Ipv = (self.end_slope * self.start_slope * self.I_0 * self.Vt) / (
                (self.start_slope * self.Vt - self.beta * self.I_0 -
                 self.beta * self.V_0 * self.start_slope) *
                (self.start_slope - self.end_slope)) - \
                 self.end_slope * self.I_0 / (self.start_slope - self.end_slope) * self.G / self.Gstc
        return self.Ipv

    def find_parameters(self):
        Isc_g = self.I_0 * self.G / self.Gstc
        self.Rs = (Isc_g - self.Ipv) / self.Ipv / self.start_slope
        self.Rp = -Isc_g / self.start_slope / self.Ipv
        self.I0 = (self.Ipv - self.V_0 / self.Rp) / \
                  (exp(self.V_0 / self.Vt / self.a1) + exp(self.V_0 / self.Vt / self.a2) - 2)
        return self.Rs, self.Rp, self.I0

    def fixed_point_method(self, Rs, Rp, voltage=None):
        current = [self.Ipv - Rs * self.Ipv / Rp]
        if voltage is None:
            voltage = self.voltage[1::] if 0 in self.voltage else self.voltage
        else:
            voltage = voltage[1:]

        lIb = log(self.I0)

        for V in voltage:
            prev_current = current[-1]
            while True:
                arg = (V + prev_current * Rs) / self.Vt
                new_current = self.Ipv - (V + prev_current * Rs) / Rp - exp(arg / self.a1 + lIb) - \
                              exp(arg / self.a2 + lIb) + 2 * self.I0
                prev_current = prev_current * (1 - self.eta) + self.eta * new_current

                if abs(prev_current - new_current) < self.fixed_point_method_tolerance:
                    break
            current.append(prev_current)
        return current

    def fixed_point_method_1(self, a1, a2, voltage=None):
        current = [self.Ipv - self.Rs * self.Ipv / self.Rp]
        if voltage is None:
            voltage = self.voltage[1::] if 0 in self.voltage else self.voltage
        else:
            voltage = voltage[1:]

        lIb = log(self.I0)

        for V in voltage:
            prev_current = current[-1]
            while True:
                arg = (V + prev_current * self.Rs) / self.Vt
                new_current = self.Ipv - (V + prev_current * self.Rs) / self.Rp - \
                              exp(arg / a1 + lIb) - \
                              exp(arg / a2 + lIb) + 2 * self.I0
                prev_current = prev_current * (1 - self.eta) + self.eta * new_current

                if abs(prev_current - new_current) < self.fixed_point_method_tolerance:
                    break
            current.append(prev_current)
        return current

    def find_mape(self, arr1, arr2, weight_arr=None):
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)

        if arr1.shape != arr2.shape:
            raise ValueError("Shape mismatch between input arrays.")

        weight_arr = np.ones_like(arr1) if weight_arr is None else weight_arr
        if arr1.shape != weight_arr.shape:
            raise ValueError("Shape mismatch between input arrays and weight array.")

        error = 0
        for exper_val, approx_val, weight in zip(arr1, arr2, weight_arr):
            if exper_val == 0:
                continue
            error += weight * abs((exper_val - approx_val) / exper_val)

        mare = error / np.sum(weight_arr)
        return mare

    def find_rmse(self, arr1, arr2, weight_arr=None):
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)

        if arr1.shape != arr2.shape:
            raise ValueError("Shape mismatch between input arrays.")

        weight_arr = np.ones_like(arr1) if weight_arr is None else weight_arr
        if arr1.shape != weight_arr.shape:
            raise ValueError("Shape mismatch between input arrays and weight array.")

        wmse = weight_arr * (arr1 - arr2) ** 2
        wmse = np.sum(wmse) / np.sum(weight_arr)
        return np.sqrt(wmse)


    @staticmethod
    def progress_bar(idx, iter_num):
        idx += 1
        print('\rCompleted {}% |{:<20}|'.format(int(idx / iter_num * 100),
                                                '█' * int(idx / iter_num * 20)), end='')

    def find_best_fit(self, weight_arr=None):
        c = self.brute_force_range
        best_fit_error = 100
        best_fit_current = None
        best_fit_rs = None
        best_fit_rp = None

        for idx, Rs in enumerate(np.linspace(self.Rs * (1 - c), self.Rs * (1 + c),
                                             self.brute_force_steps)):
            self.progress_bar(idx, self.brute_force_steps)
            for Rp in np.linspace(self.Rp * (1 - c), self.Rp * (1 + c), self.brute_force_steps):
                current = self.fixed_point_method(Rs, Rp)

                rmse = self.find_rmse(self.e_current, current, weight_arr)
                mape = self.find_mape(self.e_current, current)
                # approx_error = rmse * mape
                approx_error = rmse

                if approx_error < best_fit_error:
                    best_fit_error = approx_error
                    best_fit_current = current
                    best_fit_rs = Rs
                    best_fit_rp = Rp

        self.approx_error = best_fit_error
        self.current = best_fit_current
        self.Rs = best_fit_rs
        self.Rp = best_fit_rp
        return best_fit_current, best_fit_rs, best_fit_rp, best_fit_error

    def find_best_fit_1(self, weight_arr=None, use_one_diode_model=True, recalc_on_each_step=False):
        best_fit_error = 100
        best_fit_current = None
        best_fit_params = None

        betas = []
        photocur = []
        rs = []
        rp = []
        backward_cur = []

        for idx, a1 in enumerate(np.linspace(1, 2, self.brute_force_steps, endpoint=True)):
            self.progress_bar(idx, self.brute_force_steps)

            if not use_one_diode_model:
                for a2 in np.linspace(1, 2, self.brute_force_steps, endpoint=True):
                    self.a1, self.a2 = a1, a2

                    if recalc_on_each_step:
                        self.find_beta()
                        self.find_photovoltaic_current()
                        self.find_parameters()

                    betas.append(self.beta)
                    photocur.append(self.Ipv)
                    rs.append(self.Rs)
                    rp.append(self.Rp)
                    backward_cur.append(self.I0)

                    current = self.fixed_point_method_1(a1, a2)
                    rmse = self.find_rmse(self.e_current, current, weight_arr)
                    mape = self.find_mape(self.e_current, current)
                    # approx_error = rmse * mape
                    approx_error = rmse

                    if approx_error < best_fit_error:
                        best_fit_error = approx_error
                        best_fit_current = current
                        best_fit_params = (self.a1, self.a2, self.Rs, self.Rp, self.Ipv, self.I0)
            else:
                current = self.fixed_point_method_1(a1, 10 ** 10)
                approx_error = self.find_rmse(self.e_current, current, weight_arr)

                if approx_error < best_fit_error:
                    best_fit_error = approx_error
                    best_fit_current = current
                    best_fit_params = (self.a1, 10 ** 10, self.Rs, self.Rp, self.Ipv, self.I0)

        self.approx_error = best_fit_error
        self.current = best_fit_current
        self.a1 = best_fit_params[0]
        self.a2 = best_fit_params[1]
        self.Rs = best_fit_params[2]
        self.Rp = best_fit_params[3]
        self.Ipv = best_fit_params[4]
        self.I0 = best_fit_params[5]
        return best_fit_current, *best_fit_params, best_fit_error

    def find_best_fit_2(self, weight_arr=None):
        best_fit_error = 100
        best_fit_current = None
        best_fit_params = None

        Rs_vals = np.linspace(0.8 * self.Rs, 1.2 * self.Rs, self.brute_force_steps, endpoint=True)
        Rp_vals = np.linspace(0.8 * self.Rp, 1.2 * self.Rp, self.brute_force_steps, endpoint=True)

        for idx, a1 in enumerate(np.linspace(1, 2, self.brute_force_steps, endpoint=True)):
            self.progress_bar(idx, self.brute_force_steps)
            self.a1 = a1
            for a2 in np.linspace(1, 2, self.brute_force_steps, endpoint=True):
                self.a2 = a2
                for Rs in Rs_vals:
                    for Rp in Rp_vals:
                        current = self.fixed_point_method(Rs, Rp)
                        rmse = self.find_rmse(self.e_current, current, weight_arr)
                        mape = self.find_mape(self.e_current, current)
                        approx_error = rmse * mape

                        if approx_error < best_fit_error:
                            best_fit_error = approx_error
                            best_fit_current = current
                            best_fit_params = (self.a1, self.a2, self.Rs, self.Rp, self.Ipv, self.I0)

        return best_fit_current, *best_fit_params, best_fit_error

    @staticmethod
    def find_power(voltage, current):
        return [V * I for V, I in zip(voltage, current)]

    @staticmethod
    def find_max_power_index(power):
        max_power_index = 0
        max_power = 0
        for index, power in enumerate(power):
            if power > max_power:
                max_power_index = index
                max_power = power
        return max_power_index, max_power

    def __str__(self):
        return (f'Rs = {self.Rs}' + f'\nRp = {self.Rp}' + f'\na1 = {self.a1}' +
                f'\na2 = {self.a2}' + f'\nIpv = {self.Ipv}' + f'\nI0 = {self.I0}'
                f'\nSlope1 = {self.start_slope}' + f'\nSlope2 = {self.end_slope}')
