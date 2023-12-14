import matplotlib.pyplot as plt
import numpy as np
from math import floor
from os import getcwd

from SolarCell import SolarCell_new

CURRENT_DIR = getcwd()
INP_FILES_DIR = CURRENT_DIR + "/input_files/"
KC200GT_DIR = INP_FILES_DIR + "KC200GT_new.txt"


def get_data(filename):
    x_coord, y_coord = [], []
    with open(filename) as file:
        for line in file:
            buff = list(map(float, line.strip(" ").split()))
            x_coord.append(buff[0])
            y_coord.append(buff[1])
    return np.array(x_coord), np.array(y_coord)


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


def get_area_edge_val(arr, fraction):
    target = arr[-1] * fraction

    idx = 0
    for idx, val in enumerate(arr):
        if target <= val:
            break

    return idx - 1


def main():
    r_file = "C:\\Users\\Martyniuk Vadym\\Desktop\\600.txt"
    voltage_800, current_800 = get_data_1(r_file)
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

    # plt.plot(voltage_800, current_800)
    plt.plot(KC200GT_1000.voltage, KC200GT_1000.current)
    plt.plot(KC200GT_1000.e_voltage, KC200GT_1000.e_current)
    plt.plot(KC200GT_1000.e_voltage, weight)

    # current_800_new = KC200GT_1000.recalc(KC200GT_1000.Rs * 1000 / 600,
    # KC200GT_1000.Rp * 1000 / 600, G, voltage_800)
    # current_800_new = KC200GT_1000.recalc(KC200GT_1000.Rs, KC200GT_1000.Rp, G, voltage_800)

    # plt.plot(voltage_800, current_800_new)
    plt.grid()
    plt.show()

    # print(KC200GT_1000.find_rmse(current_800, current_800_new))
    print(KC200GT_1000)

    """
    KC200GT_1000.run(weight)
    power_1000 = KC200GT_1000.find_power(KC200GT_1000.e_voltage, KC200GT_1000.e_current)
    current_800_new = KC200GT_1000.recalc(G)
    """

    """
    plt.plot(KC200GT_1000.e_voltage, KC200GT_1000.e_current)
    plt.plot(KC200GT_1000.voltage, KC200GT_1000.current)
    # plt.plot(KC200GT_1000.voltage, power_1000)
    plt.plot(KC200GT_1000.voltage, weight)
    plt.plot(voltage_800, current_800)
    plt.plot(KC200GT_1000.voltage, current_800_new)
    plt.grid()
    plt.show()
    """


def plot_slope_first():
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
    sp = SolarCell_new(cell_num=54, Voc=32.9, Isc=8.21, a1=1.0, a2=1.2,
                       thermal_coefficient_type='absolute', G=1000,
                       experimental_V=e_voltage,
                       experimental_I=e_current,
                       is_experimental_data_provided=True,
                       mode='overall')

    power_window = normalize(np.array(sp.find_power(sp.e_voltage, sp.e_current)))
    max_power_index, _ = sp.find_max_power_index(power_window)
    power_window_order_4 = power_window ** 4
    power_window_order_4[max_power_index:] = power_window[max_power_index:]
    sp.alpha_enum(power_window_order_4, use_one_diode_model=False)
    # sp.res_enum(power_window_order_4)

    print(sp)

    plt.plot(e_voltage[first_idx:], e_current[first_idx:], linestyle='-')
    plt.plot(sp.voltage[first_idx:], sp.current[first_idx:], linestyle='--')
    plt.plot(e_voltage[first_idx:], power_window_order_4[first_idx:], linestyle='-.')
    plt.axvline(e_voltage[second_idx], linestyle='--', color='black')

    plt.grid()
    plt.xlabel('V, B')
    plt.ylabel('I, A')
    plt.show()


def plot_gaussian_window():
    sp = SolarCell_new(cell_num=54, Voc=32.9, Isc=8.21, a1=1.0, a2=1.2,
                       thermal_coefficient_type='absolute', G=1000,
                       experimental_V=e_voltage,
                       experimental_I=e_current,
                       is_experimental_data_provided=True,
                       mode='overall')
    # sp.a2 = 10 ** 10

    power_window = normalize(np.array(sp.find_power(sp.e_voltage, sp.e_current)))
    max_power_index, _ = sp.find_max_power_index(power_window)

    Vmpp = e_voltage[max_power_index]
    eta = 0.1
    window = np.exp(-0.5 * ((e_voltage - Vmpp) / eta / Vmpp / 2) ** 2)

    sp.alpha_enum(window, use_one_diode_model=False, recalculate=False)
    # sp.res_enum(window)

    print(sp)

    plt.plot(e_voltage[first_idx:], e_current[first_idx:], linestyle='-')
    plt.plot(sp.voltage[first_idx:], sp.current[first_idx:], linestyle='--')
    plt.plot(e_voltage[first_idx:], window[first_idx:], linestyle='-.')
    plt.axvline(e_voltage[second_idx], linestyle='--', color='black')

    plt.grid()
    plt.xlabel('V, B')
    plt.ylabel('I, A')
    plt.show()


def plot_error_bar_graph():
    error_names = ("$w^{1}$", "$w^{4}$", "$w^{4}_{norm}$", "$w_{g}$")

    # Alpha enum
    bin_vals = {
        '$wRMSE$': (0.09199, 0.04377, 0.10426, 0.10119),
        '$MAPE$': (0.01859, 0.08241, 0.01859, 0.01859),
        '$P$': [i * 15 for i in (0.0017105123, 0.0036067037, 0.0019386657, 0.0018815802)],
    }

    # Res enum
    """bin_vals = {
        '$wRMSE$': (0.01816, 0.02007, 0.02038, 0.01977),
        '$MAPE$': (0.00578, 0.00619, 0.00581, 0.00582),
        '$P$': [i * 100 for i in (0.0001049787, 0.0001242272, 0.0001183182, 0.0001150385)],
    }"""

    x = np.arange(len(error_names))
    width = 0.25
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in bin_vals.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding=3, fmt='%.3f', label_type='center')
        multiplier += 1

    ax.set_axisbelow(True)
    ax.grid(axis='y')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Значення помилки')
    ax.set_ylabel('Тип вагової функції')
    ax.set_xticks(x + width, error_names)
    ax.legend(loc='upper right', ncols=3)

    ax.set_ylim(0, 0.125) # Alpha enum
    # ax.set_ylim(0, 0.025) # Res enum

    plt.show()


def plot_different_alpha_enum_approaches():
    sp = SolarCell_new(cell_num=54, Voc=32.9, Isc=8.21, a1=1.0, a2=1.2,
                       thermal_coefficient_type='absolute', G=1000,
                       experimental_V=e_voltage,
                       experimental_I=e_current,
                       is_experimental_data_provided=True,
                       mode='overall')

    power_window = normalize(np.array(sp.find_power(sp.e_voltage, sp.e_current)))
    max_power_index, _ = sp.find_max_power_index(power_window)

    Vmpp = e_voltage[max_power_index]
    eta = 0.1
    window = np.exp(-0.5 * ((e_voltage - Vmpp) / eta / Vmpp / 2) ** 2)

    sp.alpha_enum(window, use_one_diode_model=False)
    current_1 = sp.current

    sp.alpha_enum(window, use_one_diode_model=False, recalculate=True)
    current_2 = sp.current

    print(current_1[-1])
    print(current_2[-1])

    # area 2
    plt.plot(e_voltage[first_idx:second_idx], e_current[first_idx:second_idx], linestyle='-')
    plt.plot(sp.voltage[first_idx:second_idx], current_2[first_idx:second_idx], linestyle='--')
    plt.plot(sp.voltage[first_idx:second_idx], current_1[first_idx:second_idx], linestyle=':',
             color='k')
    plt.grid()
    plt.xlabel('V, B')
    plt.ylabel('I, A')
    plt.show()

    # area 3
    plt.plot(e_voltage[second_idx:], e_current[second_idx:], linestyle='-')
    plt.plot(sp.voltage[second_idx:], current_1[second_idx:], linestyle='--')
    plt.plot(sp.voltage[second_idx:], current_2[second_idx:], linestyle=':', color='k')
    plt.grid()
    plt.xlabel('V, B')
    plt.ylabel('I, A')
    plt.show()


def plot_res_enum_different_windows():
    sp = SolarCell_new(cell_num=54, Voc=32.9, Isc=8.21, a1=1.0, a2=1.2,
                       thermal_coefficient_type='absolute', G=1000,
                       experimental_V=e_voltage,
                       experimental_I=e_current,
                       is_experimental_data_provided=True,
                       mode='overall')

    power_window = normalize(np.array(sp.find_power(sp.e_voltage, sp.e_current)))
    max_power_index, _ = sp.find_max_power_index(power_window)

    Vmpp = e_voltage[max_power_index]
    eta = 0.1

    power_window_order_4 = power_window ** 1
    sp.res_enum(power_window_order_4)
    current_1 = sp.current

    power_window_order_4 = power_window ** 4
    sp.res_enum(power_window_order_4)
    current_2 = sp.current

    power_window_order_4[max_power_index:] = power_window[max_power_index:]
    sp.res_enum(power_window_order_4)
    current_3 = sp.current

    window = np.exp(-0.5 * ((e_voltage - Vmpp) / eta / Vmpp / 2) ** 2)
    sp.res_enum(window)
    current_4 = sp.current

    plt.plot(e_voltage[first_idx:second_idx], e_current[first_idx:second_idx], linestyle='-')
    plt.plot(sp.voltage[first_idx:second_idx], current_1[first_idx:second_idx], linestyle='--')
    plt.plot(sp.voltage[first_idx:second_idx], current_2[first_idx:second_idx], linestyle='-.')
    plt.plot(sp.voltage[first_idx:second_idx], current_3[first_idx:second_idx], linestyle=':')
    plt.plot(sp.voltage[first_idx:second_idx], current_4[first_idx:second_idx], linestyle='--')
    plt.axvline(e_voltage[second_idx], linestyle='--', color='black')

    plt.grid()
    plt.xlabel('V, B')
    plt.ylabel('I, A')
    plt.show()


def get_first_method_current():
    sp = SolarCell_new(cell_num=54, Voc=32.9, Isc=8.21,
                       Pmpp=200, Impp=7.61, Vmpp=26.3,
                       Ku=-0.123, Ki=3.18 / 1000, a1=1.0, a2=1.2,
                       I0=0.4218 * 10 ** (-9), Ipv=8.21,
                       thermal_coefficient_type='absolute', G=1000,
                       experimental_V=e_voltage,
                       experimental_I=e_current,
                       is_experimental_data_provided=True,
                       mode='overall')

    Rs = 0.320
    Rp = 160.5

    current = sp.fixed_point_method(Rs, Rp)
    original_len = len(current)
    current = [i for i in current if i >= 0]
    return current, original_len - len(current)


def get_second_method_current(Ipv=8.223, I0=2.152e-9, Rs=0.308, Rp=193.049):
    sp = SolarCell_new(cell_num=54, Voc=32.9, Isc=8.21,
                       Pmpp=200, Impp=7.61, Vmpp=26.3,
                       Ku=-0.123, Ki=3.18 / 1000, a1=1.076, a2=1.2,
                       I0=I0, Ipv=Ipv,
                       thermal_coefficient_type='absolute', G=1000,
                       experimental_V=e_voltage,
                       experimental_I=e_current,
                       is_experimental_data_provided=True,
                       mode='overall')

    current = sp.fixed_point_method(Rs, Rp)
    original_len = len(current)
    current = [i for i in current if i >= 0]
    return current, original_len - len(current)


if __name__ == '__main__':
    e_voltage, e_current = get_data(KC200GT_DIR)
    first_idx = get_area_edge_val(e_voltage, 0.65)
    second_idx = get_area_edge_val(e_voltage, 0.95)

    # main()
    # plot_slope_first()
    # plot_slope_second()
    # plot_power_curve_window()
    # plot_high_order_power_curve()
    # plot_gaussian_window()
    # plot_error_bar_graph()
    # plot_different_alpha_enum_approaches()
    plot_res_enum_different_windows()
