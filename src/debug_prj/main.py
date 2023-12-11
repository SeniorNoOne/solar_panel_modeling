import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Sequence
from math import exp, floor, log


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

        self.beta = (exp(self.V_0 / self.Vt / self.a1) +
                     exp(self.V_0 / self.Vt / self.a2)) / \
                    (exp(self.V_0 / self.Vt / self.a1) +
                     exp(self.V_0 / self.Vt / a2) - 2)

        # Algorithm parameters
        self.current = None
        self.approx_error = None
        if self.Ipv is None:
            self.Ipv = self.find_photovoltaic_current()

        if self.I0 is None:
            self.Rs, self.Rp, self.I0 = self.find_parameters()

    def res_enum(self, weight_arr=None):
        self.current, self.Rs, self.Rp, self.approx_error = self.find_best_fit(weight_arr)
        print("\nError = ", self.approx_error)

    def alpha_enum(self, weight_arr=None, use_one_diode_model=True):
        self.current, self.a1, self.a2, self.approx_error = self.find_best_fit_1(weight_arr,
                                                                                 use_one_diode_model)
        print("\nError = ", self.approx_error)

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

    def find_photovoltaic_current(self) -> Union[float, int]:
        return (self.end_slope * self.start_slope * self.I_0 * self.Vt) / (
                (self.start_slope * self.Vt - self.beta * self.I_0 -
                 self.beta * self.V_0 * self.start_slope) *
                (self.start_slope - self.end_slope)) - \
            self.end_slope * self.I_0 / (self.start_slope - self.end_slope) * self.G / self.Gstc

    def find_parameters(self):
        Isc_g = self.I_0 * self.G / self.Gstc
        Rs = (Isc_g - self.Ipv) / self.Ipv / self.start_slope
        Rp = -Isc_g / self.start_slope / self.Ipv
        I0 = (self.Ipv - self.V_0 / Rp) / \
             (exp(self.V_0 / self.Vt / self.a1) + exp(self.V_0 / self.Vt / self.a2) - 2)
        return Rs, Rp, I0

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

    def find_mape(self, experimental_data, approximation):
        error = 0

        if len(experimental_data) == len(approximation):
            if self.mode == 'overall':
                for exper_val, approx_val in zip(experimental_data, approximation):
                    if exper_val == 0:
                        continue
                    error += abs(exper_val - approx_val) / abs(exper_val) * 100
                return error / len(approximation)

            else:
                for exper_val, approx_val in zip(experimental_data[self.area1_2:self.area2_3],
                                                 approximation[self.area1_2:self.area2_3]):
                    if exper_val == 0:
                        continue
                    error += abs(exper_val - approx_val) / abs(exper_val) * 100
                return error / self.approx_range_minimization / 2
        else:
            raise ValueError("Different arrays length")

    def find_rmse(self, experimental_data, approximation, weight=None):
        if self.mode == "overall":
            if len(experimental_data) == len(approximation):
                error = []
                for exper_val, approx_val in zip(experimental_data, approximation):
                    if exper_val == 0:
                        continue
                    error.append((exper_val - approx_val) ** 2)
                if weight is not None:
                    error = weight[1:] * np.array(error)
                    error = np.sum(error) / np.sum(weight)
                else:
                    error = np.sum(error)
                return np.sqrt(error / len(approximation))
            else:
                print(len(approximation))
                print(len(experimental_data))
                raise ValueError("Different arrays length")

        else:
            if len(experimental_data) == len(approximation):
                error = 0
                for exper_val, approx_val in zip(experimental_data[self.area1_2:self.area2_3],
                                                 approximation[self.area1_2:self.area2_3]):
                    if exper_val == 0:
                        continue
                    error += abs(exper_val - approx_val) / abs(exper_val) * 100
                return error / self.approx_range_minimization / 2
            else:
                raise ValueError("Different arrays length")

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
                approx_error = self.find_rmse(self.e_current, current, weight_arr)

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

    def find_best_fit_1(self, weight_arr=None, use_one_diode_model=True):
        best_fit_error = 100
        best_fit_current = None
        best_fit_a1 = None
        best_fit_a2 = None

        for idx, a1 in enumerate(np.linspace(1, 2, self.brute_force_steps, endpoint=True)):
            self.progress_bar(idx, self.brute_force_steps)

            if not use_one_diode_model:
                for a2 in np.linspace(1, 2, self.brute_force_steps, endpoint=True):
                    current = self.fixed_point_method_1(a1, a2)
                    approx_error = self.find_rmse(self.e_current, current, weight_arr)

                    if approx_error < best_fit_error:
                        best_fit_error = approx_error
                        best_fit_current = current
                        best_fit_a1 = a1
                        best_fit_a2 = a2
            else:
                current = self.fixed_point_method_1(a1, 10 ** 10)
                approx_error = self.find_rmse(self.e_current, current, weight_arr)

                if approx_error < best_fit_error:
                    best_fit_error = approx_error
                    best_fit_current = current
                    best_fit_a1 = a1
                    best_fit_a2 = 10 ** 10

        self.approx_error = best_fit_error
        self.current = best_fit_current
        self.a1 = best_fit_a1
        self.a2 = best_fit_a2
        return best_fit_current, best_fit_a1, best_fit_a2, best_fit_error

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
                f'\na2 = {self.a2}' + f'\nIpv = {self.Ipv}' + f'\nI0 = {self.I0}')


def get_data(file):
    x_coord, y_coord = [], []
    for line in file:
        buff = list(map(float, line.strip(" ").split()))
        x_coord.append(buff[0])
        y_coord.append(buff[1])
    return x_coord, y_coord


def get_data_1(file):
    x_coord, y_coord = [], []
    for line in file:
        buff = list(map(float, line.strip(" ").split(", ")))
        x_coord.append(buff[0])
        y_coord.append(buff[1])
    return x_coord, y_coord


def normalize(arr, zero_padding=False, offset_coef=0.001):
    min_val = np.min(arr)
    max_val = np.max(arr)

    if zero_padding:
        new_arr = (arr - min_val) / (max_val - min_val)
    else:
        offset = offset_coef * abs(min_val)
        new_arr = (arr - min_val + offset) / (max_val - min_val + offset)
    return new_arr


def main():
    r_file = open("C:\\Users\\Martyniuk Vadym\\Desktop\\solar_panel_modeling\\input_files\\" +
                  "KC200GT_new.txt")
    e_voltage, e_current = get_data(r_file)
    e_voltage = np.array(e_voltage)
    r_file.close()

    r_file = open("C:\\Users\\Martyniuk Vadym\\Desktop\\600.txt")
    voltage_800, current_800 = get_data_1(r_file)
    voltage_800 = np.array(voltage_800)
    r_file.close()

    solar_cell_num = 54
    G = 600

    KC200GT_1000 = SolarCell_new(cell_num=54, Voc=32.9, Isc=8.21, a1=1.0, a2=1.2,
                                 thermal_coefficient_type='absolute', G=1000,
                                 experimental_V=e_voltage,
                                 experimental_I=e_current,
                                 is_experimental_data_provided=True,
                                 mode='overall')

    """KC200GT_800 = SolarCell_new(cell_num=54, Voc=32.9, Isc=8.21, a1=1.0, a2=1.2,
                                thermal_coefficient_type='absolute', G=1000,
                                experimental_V=voltage_800,
                                experimental_I=current_800,
                                is_experimental_data_provided=True,
                                mode='overall')"""

    print(KC200GT_1000.start_slope, KC200GT_1000.end_slope)

    power_1000 = np.array(KC200GT_1000.find_power(KC200GT_1000.e_voltage, KC200GT_1000.e_current))
    weight = normalize(power_1000) ** 2
    max_power_index, _ = KC200GT_1000.find_max_power_index(power_1000)
    # weight[max_power_index:] = [1] * (len(weight) - max_power_index)
    # weight[max_power_index:] = np.linspace(1, 0.5, len(weight) - max_power_index)
    # KC200GT_1000.res_enum(weight)
    KC200GT_1000.Rs = 0.26289609693141797
    KC200GT_1000.Rp = 120.47888324978973
    KC200GT_1000.alpha_enum(weight, use_one_diode_model=False)

    #plt.plot(voltage_800, current_800)
    plt.plot(KC200GT_1000.voltage, KC200GT_1000.current)
    plt.plot(KC200GT_1000.e_voltage, KC200GT_1000.e_current)
    plt.plot(KC200GT_1000.e_voltage, weight)

    # current_800_new = KC200GT_1000.recalc(KC200GT_1000.Rs * 1000 / 600, KC200GT_1000.Rp * 1000 / 600, G, voltage_800)
    # current_800_new = KC200GT_1000.recalc(KC200GT_1000.Rs, KC200GT_1000.Rp, G, voltage_800)

    #plt.plot(voltage_800, current_800_new)
    plt.grid()
    plt.show()

    # print(KC200GT_1000.find_rmse(current_800, current_800_new))
    print(KC200GT_1000)

    """
    KC200GT_1000.run(weight)
    power_1000 = KC200GT_1000.find_power(KC200GT_1000.e_voltage, KC200GT_1000.e_current)
    current_800_new = KC200GT_1000.recalc(G)"""

    """plt.plot(KC200GT_1000.e_voltage, KC200GT_1000.e_current)
    plt.plot(KC200GT_1000.voltage, KC200GT_1000.current)
    # plt.plot(KC200GT_1000.voltage, power_1000)
    plt.plot(KC200GT_1000.voltage, weight)
    plt.plot(voltage_800, current_800)
    plt.plot(KC200GT_1000.voltage, current_800_new)
    plt.grid()
    plt.show()"""


def plot_slope_first():
    with open("C:\\Users\\Martyniuk Vadym\\Desktop\\solar_panel_modeling\\input_files\\" +
              "KC200GT_new.txt") as r_file:
        e_voltage, e_current = get_data(r_file)
        e_voltage = np.array(e_voltage)

    area_len = floor(len(e_voltage) * 0.22)
    k, b = np.polyfit(e_voltage[:area_len], e_current[:area_len], 1)

    x_line = np.array(e_voltage[:area_len])
    y_line = k * x_line + np.full_like(x_line, b)

    print(k, b)

    plt.plot(e_voltage[:area_len], e_current[:area_len], marker='.', linestyle='-')
    plt.plot(x_line, y_line, linestyle='--')
    plt.grid()
    plt.xlabel('V, B')
    plt.ylabel('I, A')
    plt.show()


def plot_slope_second():
    with open("C:\\Users\\Martyniuk Vadym\\Desktop\\solar_panel_modeling\\input_files\\" +
              "KC200GT_new.txt") as r_file:
        e_voltage, e_current = get_data(r_file)
        e_voltage = np.array(e_voltage)

    area_len = -floor(len(e_voltage) * 0.085)
    area_len_ext = 5 * area_len

    x = e_voltage[area_len:]
    y = e_current[area_len:]
    V_0 = e_voltage[-1]

    a_1, a_2 = np.polyfit(x, y, 1)
    b_1, b_2, b_3 = np.polyfit(x, y, 2)

    x_line = np.array(e_voltage[area_len_ext:])
    y_line_1 = a_1 * x_line + a_2
    y_line_2 = b_1 * x_line * x_line + b_2 * x_line + b_3

    print(a_1, a_2)
    print(b_1, b_2, b_3)
    print(2 * b_1 * V_0 + b_2)

    plt.plot(e_voltage[area_len_ext:], e_current[area_len_ext:], marker='.', linestyle='-')
    plt.plot(x_line, y_line_1, linestyle='--')
    plt.plot(x_line, y_line_2, linestyle='-.')
    plt.axvline(e_voltage[area_len], linestyle='--', color='black')
    plt.grid()
    plt.xlabel('V, B')
    plt.ylabel('I, A')
    plt.show()


def plot_power_curve_window():
    with open("C:\\Users\\Martyniuk Vadym\\Desktop\\solar_panel_modeling\\input_files\\" +
              "KC200GT_new.txt") as r_file:
        e_voltage, e_current = get_data(r_file)
        e_voltage = np.array(e_voltage)

    sp = SolarCell_new(cell_num=54, Voc=32.9, Isc=8.21, a1=1.0, a2=1.2,
                       thermal_coefficient_type='absolute', G=1000,
                       experimental_V=e_voltage,
                       experimental_I=e_current,
                       is_experimental_data_provided=True,
                       mode='overall')

    power_window = normalize(np.array(sp.find_power(sp.e_voltage, sp.e_current)))
    max_power_index, _ = sp.find_max_power_index(power_window)
    std = np.std(e_voltage)
    window = np.exp(-(e_voltage - e_voltage[max_power_index]) ** 2 / 2 / std)

    # plt.plot(e_voltage, e_current, marker='.', linestyle='-')
    plt.plot(e_voltage, window, linestyle='--')
    plt.plot(e_voltage, power_window, linestyle='-')
    plt.plot(e_voltage, power_window ** 2, linestyle='--')
    plt.plot(e_voltage, power_window ** 3, linestyle='-.')
    plt.plot(e_voltage, power_window ** 4, linestyle=':')
    plt.grid()
    plt.xlabel('V, B')
    plt.ylabel('$w_{norm}$')
    plt.show()


def plot_high_order_power_curve():
    with open("C:\\Users\\Martyniuk Vadym\\Desktop\\solar_panel_modeling\\input_files\\" +
              "KC200GT_new.txt") as r_file:
        e_voltage, e_current = get_data(r_file)
        e_voltage = np.array(e_voltage)

    sp = SolarCell_new(cell_num=54, Voc=32.9, Isc=8.21, a1=1.0, a2=10 ** 10,
                       thermal_coefficient_type='absolute', G=1000,
                       experimental_V=e_voltage,
                       experimental_I=e_current,
                       is_experimental_data_provided=True,
                       mode='overall')

    first_idx = 0
    second_idx = 0

    target_val = e_voltage[-1] * 0.65
    for idx, val in enumerate(e_voltage):
        if target_val <= val:
            first_idx = idx
            break

    target_val = e_voltage[-1] * 0.95
    for idx, val in enumerate(e_voltage):
        if target_val <= val:
            second_idx = idx
            break

    power_window = normalize(np.array(sp.find_power(sp.e_voltage, sp.e_current)))
    max_power_index, _ = sp.find_max_power_index(power_window)
    power_window_order_4 = power_window ** 4
    power_window_order_4[max_power_index:] = power_window[max_power_index:] ** 0.9
    sp.alpha_enum(power_window_order_4, use_one_diode_model=False)

    plt.plot(e_voltage[first_idx:], e_current[first_idx:], linestyle='-')
    plt.plot(sp.voltage[first_idx:], sp.current[first_idx:], linestyle='--')
    plt.plot(e_voltage[first_idx:], power_window_order_4[first_idx:], linestyle='-.')
    plt.axvline(e_voltage[second_idx], linestyle='--', color='black')

    plt.grid()
    plt.xlabel('V, B')
    plt.ylabel('I, A')
    plt.show()


def plot_gaussian_window():
    with open("C:\\Users\\Martyniuk Vadym\\Desktop\\solar_panel_modeling\\input_files\\" +
              "KC200GT_new.txt") as r_file:
        e_voltage, e_current = get_data(r_file)
        e_voltage = np.array(e_voltage)

    sp = SolarCell_new(cell_num=54, Voc=32.9, Isc=8.21, a1=1.0, a2=10 ** 10,
                       thermal_coefficient_type='absolute', G=1000,
                       experimental_V=e_voltage,
                       experimental_I=e_current,
                       is_experimental_data_provided=True,
                       mode='overall')

    first_idx = 0
    second_idx = 0

    target_val = e_voltage[-1] * 0.65
    for idx, val in enumerate(e_voltage):
        if target_val <= val:
            first_idx = idx
            break

    target_val = e_voltage[-1] * 0.95
    for idx, val in enumerate(e_voltage):
        if target_val <= val:
            second_idx = idx
            break

    power_window = normalize(np.array(sp.find_power(sp.e_voltage, sp.e_current)))
    max_power_index, _ = sp.find_max_power_index(power_window)

    Vmpp = e_voltage[max_power_index]
    eta = 0.1
    window = np.exp(-0.5 * ((e_voltage - Vmpp) / eta / Vmpp / 2) ** 2)

    sp.alpha_enum(window, use_one_diode_model=False)

    plt.plot(e_voltage[first_idx:], e_current[first_idx:], linestyle='-')
    plt.plot(sp.voltage[first_idx:], sp.current[first_idx:], linestyle='--')
    plt.plot(e_voltage[first_idx:], window[first_idx:], linestyle='-.')
    plt.axvline(e_voltage[second_idx], linestyle='--', color='black')

    plt.grid()
    plt.xlabel('V, B')
    plt.ylabel('I, A')
    plt.show()


if __name__ == '__main__':
    # main()
    # plot_slope_first()
    # plot_slope_second()
    # plot_power_curve_window()
    # plot_high_order_power_curve()
    plot_gaussian_window()
