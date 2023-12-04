import numpy as np
from typing import Union, Sequence
from math import exp, floor, log


class SolarCell:
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
            raise ValueError("  ")
            
        self.I0 = I0
        self.Ipv = Ipv
        self.start_slope = start_slope
        self.end_slope = end_slope
        self.Vt = Vt
        
        # Impurity coefficients
        self.a1 = a1
        self.a2 = a2

        # Custom functions
        self.custom_thermal_voltage_func = custom_thermal_voltage_func
        self.custom_photovoltaic_current_func = custom_photovoltaic_current_func
        self.custom_backward_current_func = custom_backward_current_func

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
        self.e_I = experimental_I
        self.e_V = experimental_V

        # Solar cell parameters
        self.cell_num = cell_num
        self.Voc = Voc
        self.Isc = Isc
        self.Pmpp = Pmpp
        self.Impp = Impp
        self.Vmpp = Vmpp

        self.Ku = Ku
        self.Ki = Ki
        self.Kp = Kp
        self.dT = self.T - self.Tstc

        if self.Vt is None:
            self.Vt = self.find_thermal_voltage()
        self.Ipv_estimate = self.find_photovoltaic_current_estimate()
        self.I0_estimate = self.find_backward_current_estimate()

        # Algorithm's variables
        self.is_experimental_data_provided = is_experimental_data_provided
        self.fixed_point_method_tolerance = fixed_point_method_tolerance
        self.mesh_points_num = mesh_points_num
        self.brute_force_steps = brute_force_steps
        self.mode = mode
        self.approx_range_minimization = floor(len(self.e_V) / 10) if approx_range_minimization is None else approx_range_minimization

        # Approximation
        self.I_0 = self.e_I[0]
        self.V_0 = self.e_V[-1]

        if is_experimental_data_provided:
            if 0 in self.e_V:
                self.V_mesh = self.e_V
            else:
                self.V_mesh = [0] + self.e_V
        else:
            self.create_voltage_mesh()

        self.start_fit_points_number = floor(len(self.V_mesh) * 0.22) 
        self.end_fit_points_number = floor(len(self.V_mesh) * 0.085)
        
        
        self.area1_2 = int(0.55 * len(self.V_mesh))
        self.area2_3 = int(0.9 * len(self.V_mesh))
        
        
        if self.start_slope is None:
            self.start_slope = self.find_slope("start")
        if self.end_slope is None:
            self.end_slope = self.find_slope("end")

        self.beta = (exp(self.V_0 / self.Vt / self.a1) +
                     exp(self.V_0 / self.Vt / self.a2)) / \
                    (exp(self.V_0 / self.Vt / self.a1) +
                     exp(self.V_0 / self.Vt / a2) - 2)

        if self.Ipv is None:
            self.Ipv = self.find_photovoltaic_current()
            
        if self.I0 is None:
            self.Rs_estimate, self.Rp_estimate, self.I0 = self.find_parameters()

        """
        while abs(self.Ipv_estimate - self.Ipv) / self.Ipv > 0.001:
            self.end_fit_points_number = self.end_fit_points_number - 1
            self.start_slope = self.find_slope("start")
            self.end_slope = self.find_slope("end")
            self.Ipv = self.find_photovoltaic_current()
        """
        

        #self.current = self.fixed_point_method()
        #self.approximation_error = self.find_approximation_error()
        pass  
    
    def run(self):
        self.current, self.Rs, self.Rp, self.approx_error = self.find_best_fit()
        #self.current = self.fixed_point_method(self.Rs_estimate, self.Rp_estimate)
        print("\nRs = ", self.Rs, "\nRp = ", self.Rp, "\nRS Error = ", self.approx_error)
        
        
    def recalc(self, G):
        _, Rs, Rp, approx_error = self.find_best_fit()
        print("\nRs = ", Rs, "\nRp = ", Rp)
        
        voltage = self.V_mesh * G / self.Gstc
        current = self.fixed_point_method_rv(Rs, Rp, voltage)
        return voltage, current
        
        
    def create_voltage_mesh(self) -> Sequence[Union[float, int]]:
        mesh_center_index = int(self.mesh_points_num / 2)
        return list(np.concatenate(
            (np.linspace(0, self.Vmpp, mesh_center_index, endpoint=False),
             np.linspace(self.Vmpp, self.Voc, self.mesh_points_num -
                         mesh_center_index))))

    def find_thermal_voltage(self) -> Union[float, int]:
        if self.custom_thermal_voltage_func is None:
            return self.cell_num * self.k * self.T / self.q
        else:
            return self.custom_thermal_voltage_func()

    def find_slope(self, slope_point) -> Union[float, int]:
        if slope_point == "start":
            k, b = np.polyfit(self.V_mesh[:self.start_fit_points_number:],
                              self.e_I[:self.start_fit_points_number:], 1)
            return k
        if slope_point == "end":
            a, b, c = np.polyfit(self.V_mesh[-self.end_fit_points_number::],
                                 self.e_I[-self.end_fit_points_number::], 2)
            return a * self.V_0 * 2 + b

    def find_photovoltaic_current(self) -> Union[float, int]:
        return (self.end_slope * self.start_slope * self.I_0 * self.Vt) / (
                (self.start_slope * self.Vt - self.beta * self.I_0 -
                 self.beta * self.V_0 * self.start_slope) *
                (self.start_slope - self.end_slope)) - \
               self.end_slope * self.I_0 / (self.start_slope - self.end_slope) * self.G / 1000

    def find_parameters(self):
        Rs = (self.I_0 * self.G / self.Gstc - self.Ipv) / self.Ipv / self.start_slope
        Rp = -self.I_0 * self.G / self.Gstc / self.start_slope / self.Ipv
        I0 = (self.Ipv - self.V_0 / Rp) / \
             (exp(self.V_0 / self.Vt) + exp(self.V_0 / self.Vt / self.a2) - 2)
        return Rs, Rp, I0

    def find_photovoltaic_current_estimate(self) -> Union[float, int]:
        if self.custom_photovoltaic_current_func is None:
            if self.thermal_coefficient_type == "absolute":
                return (self.Isc + self.dT * self.Ki) * \
                       self.G / self.Gstc
            elif self.thermal_coefficient_type == "relative":
                return self.Isc * (1 + self.dT * self.Ki) * \
                       self.G / self.Gstc
        else:
            return self.custom_thermal_voltage_func()

    def find_backward_current_estimate(self) -> Union[float, int]:
        if self.custom_backward_current_func is None:
            if self.thermal_coefficient_type == "absolute":
                return self.Ipv_estimate / (np.exp((self.Voc + self.dT *
                                                    self.Ku) / self.Vt) - 1)
            elif self.thermal_coefficient_type == "relative":
                return self.Ipv_estimate / (np.exp(
                    self.Voc * (1 + self.dT * self.Ku) / self.Vt) - 1)
        else:
            return self.custom_backward_current_func()

    def fixed_point_method(self, Rs, Rp):
        current = [self.Ipv - Rs * self.Ipv / Rp]
        #l_cur = []
        lIb = log(self.I0)
        V_mesh = self.V_mesh[1::] if 0 in self.V_mesh else self.V_mesh
        for V in V_mesh:
            last_current_val = current[-1]
            while True:
                arg = (V + last_current_val * Rs) / self.Vt
                """func_val = self.Ipv - (
                            V + last_current_val * Rs) / Rp - self.I0 * (
                                       exp(arg / self.a1) + exp(
                                   arg / self.a2) - 2)"""

                func_val = self.Ipv - (V + last_current_val * Rs) / Rp - self.I0 * exp(
                    arg + lIb) - exp(arg / self.a2 + lIb) + 2 * self.I0
                

                last_current_val = last_current_val * (1 - 0.2) + 0.2 * func_val
                #buffer = (exp(arg + lIb) + exp(arg / self.a2 + lIb) + 2 * self.I0) 
                if abs(last_current_val - func_val) < \
                        self.fixed_point_method_tolerance:
                    break
            current.append(last_current_val)
            #l_cur.append(buffer / last_current_val)
        return current
    
    def fixed_point_method_rv(self, Rs, Rp, V_mesh):
        current = [self.Ipv - Rs * self.Ipv / Rp]
        lIb = log(self.I0)
        
        for V in V_mesh:
            last_current_val = current[-1]
            while True:
                arg = (V + last_current_val * Rs) / self.Vt
        
                func_val = self.Ipv - (V + last_current_val * Rs) / Rp - exp(arg + lIb) - exp(arg / self.a2 + lIb) + 2 * self.I0
                

                last_current_val = last_current_val * (1 - 0.2) + 0.2 * func_val
                
                if abs(last_current_val - func_val) < self.fixed_point_method_tolerance:
                    break
                    
            current.append(last_current_val)
        return current

    def find_approximation_error(self, experimental_data, approximation):
        if self.mode == "overall":
            if len(experimental_data) == len(approximation):
                error = 0
                for exper_val, approx_val in zip(experimental_data, approximation):
                    if exper_val == 0:
                        continue
                    error += abs(exper_val - approx_val) / abs(exper_val) * 100
                return error / len(approximation)
            else:
                raise ValueError("Different arrays length")

        else:
            if len(experimental_data) == len(approximation):
                error = 0
                for exper_val, approx_val in zip(experimental_data[self.area1_2:self.area2_3], approximation[self.area1_2:self.area2_3]):
                    if exper_val == 0:
                        continue
                    error += abs(exper_val - approx_val) / abs(exper_val) * 100
                return error / self.approx_range_minimization / 2
            else:
                raise ValueError("Different arrays length")
                
                
    def RMSE(self, experimental_data, approximation):
        if self.mode == "overall":
            if len(experimental_data) == len(approximation):
                error = 0
                for exper_val, approx_val in zip(experimental_data, approximation):
                    if exper_val == 0:
                        continue
                    error += (exper_val - approx_val) ** 2 
                return (error / len(approximation)) ** 0.5
            else:
                print(len(approximation))
                print(len(experimental_data))
                raise ValueError("Different arrays length")

        else:
            if len(experimental_data) == len(approximation):
                error = 0
                for exper_val, approx_val in zip(experimental_data[self.area1_2:self.area2_3], approximation[self.area1_2:self.area2_3]):
                    if exper_val == 0:
                        continue
                    error += abs(exper_val - approx_val) / abs(exper_val) * 100
                return error / self.approx_range_minimization / 2
            else:
                raise ValueError("Different arrays length")


    def find_best_fit(self):
        best_fit_error = 100
        best_fit_res = None
        best_fit_current = None
        
        iter_num = 50
        
        for index, Rs in enumerate(np.linspace(self.Rs_estimate * 0.7, self.Rs_estimate * 1.3, iter_num)):
            print('\rCompleted {}% |{:<20}|'.format(int(index / iter_num * 100), '█' * int(index / iter_num * 20)), end="")
            for Rp in np.linspace(self.Rp_estimate * 0.7, self.Rp_estimate * 1.3, iter_num):
                current_buff = self.fixed_point_method(Rs, Rp)
                power_buff = self.find_power(current_buff)
                max_power_index_buff, _ = self.find_max_power_index(power_buff)
                error_buff = self.RMSE(self.e_I, current_buff)
                if error_buff < best_fit_error:
                    best_fit_error = error_buff
                    best_fit_res = (Rs, Rp)
                    best_fit_current = current_buff
        return (best_fit_current, *best_fit_res, best_fit_error)

    def find_best_fit_2(self):
        best_fit_error = 100
        best_fit_res = None
        best_fit_current = None
        for index, Rs in enumerate(np.linspace(self.start_slope * 0.9, self.start_slope * 1.1, 100)):
            print(index)
            for index, Rp in np.linspace(self.end_slope * 0.9, self.end_slope * 1.1, 100):
                Rs, Rp, self.I0 = self.find_parameters()
                current_buff = self.fixed_point_method(Rs, Rp)
                max_power_index, max_power = self.find_max_power_index(current_buff)
                error_buff = self.find_approximation_error(self.e_I, current_buff[1::], max_power_index)
                if error_buff < best_fit_error:
                    best_fit_error = error_buff
                    best_fit_res = (Rs, Rp)
                    best_fit_current = current_buff
        return (best_fit_current, *best_fit_res, best_fit_error)

    def find_power(self, current):
        return [V * I for V, I in zip(self.V_mesh, current)]

    def find_max_power_index(self, power):
        max_power_index = 0
        max_power = 0
        for index, power in enumerate(power):
            if power > max_power:
                max_power_index = index
                max_power = power
        return max_power_index, max_power
