import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
from math import floor
from os import getcwd, listdir

from SolarPanel import SolarPanel

CURRENT_DIR = getcwd()
INP_FILES_DIR = CURRENT_DIR + "/input_files/"
OUTPUT_FILES_DIR = CURRENT_DIR + "/output_files/"
KC200GT_DIR = INP_FILES_DIR + "KC200GT_new.txt"

A_ENUM_1D_FN = 'alpha_enum_one_diode_no_recalc.txt'
A_ENUM_2D_FN = 'alpha_enum_two_diode_no_recalc.txt'
R_ENUM_1D_FN = 'res_enum_one_diode.txt'
R_ENUM_2D_FN = 'res_enum_two_diode.txt'
CUR_FN = [A_ENUM_1D_FN, A_ENUM_2D_FN, R_ENUM_1D_FN, R_ENUM_2D_FN]

LINESTYLES = ['-', '--', '-.', ':', '--']

SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

FIGSIZE_WIDE = (9, 4.8)
FIGSIZE_TALL = (6.4, 6)

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  #


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
        if target < val:
            break

    return idx - 1


def get_max_power_idx(voltage_arr, current_arr):
    max_power_idx = -1
    max_power_val = -1

    for idx, (v, i) in enumerate(zip(voltage_arr, current_arr)):
        if (power := i * v) > max_power_val:
            max_power_idx = idx
            max_power_val = power

    return max_power_idx


def main():
    r_file = "C:\\Users\\Martyniuk Vadym\\Desktop\\600.txt"
    voltage_800, current_800 = get_data_1(r_file)
    solar_cell_num = 54
    G = 600

    KC200GT_1000 = SolarPanel(cell_num=54, Voc=32.9, Isc=8.21, a1=1.0, a2=1.2,
                              thermal_coefficient_type='absolute', G=1000,
                              experimental_V=e_voltage,
                              experimental_I=e_current,
                              is_experimental_data_provided=True,
                              mode='overall')

    """KC200GT_800 = SolarPanel(cell_num=54, Voc=32.9, Isc=8.21, a1=1.0, a2=1.2,
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


def plot_iv_curve_split():
    area_1_idx = get_area_edge_val(e_voltage, 0.55)
    area_2_idx = get_area_edge_val(e_voltage, 0.9)

    max_power_idx = 0
    max_power = -1
    for idx, (v, i) in enumerate(zip(e_voltage, e_current)):
        if (power := v * i) > max_power:
            max_power = power
            max_power_idx = idx

    fig, ax = plt.subplots(1, 1)

    ax.plot(e_voltage, e_current)
    ax.scatter(e_voltage[0], e_current[0], s=30, c='r', zorder=2)
    ax.scatter(e_voltage[-1], e_current[-1], s=30, c='r', zorder=2)
    ax.scatter(e_voltage[max_power_idx], e_current[max_power_idx], s=30, c='r', zorder=2)
    ax.axvline(e_voltage[area_1_idx], linestyle='--', color='black')
    ax.axvline(e_voltage[area_2_idx], linestyle='--', color='black')
    ax.grid()
    ax.set_xlabel('V, B')
    ax.set_ylabel('I, A')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.text(10, 4.8, 'I', fontsize=BIGGER_SIZE)
    ax.text(24, 4.8, 'II', fontsize=BIGGER_SIZE)
    ax.text(32, 4.8, 'III', fontsize=BIGGER_SIZE)
    ax.text(0.5, 7.5, '$I_{sc}$', fontsize=BIGGER_SIZE)
    ax.text(30.5, 0.2, '$V_{oc}$', fontsize=BIGGER_SIZE)
    ax.text(20, 6.8, '$(V_{mpp}, I_{mpp})$', fontsize=BIGGER_SIZE)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    plt.tight_layout()
    plt.show()


def plot_slope_first():
    area_len = floor(len(e_voltage) * 0.22)
    k, b = np.polyfit(e_voltage[:area_len], e_current[:area_len], 1)

    x_line = np.array(e_voltage[:area_len])
    y_line = k * x_line + np.full_like(x_line, b)

    x1, x2 = 1.0, 4.5
    y1, y2 = 8.15, 8.18
    fig, ax = plt.subplots(1, 1)

    ax.plot(e_voltage[:area_len], e_current[:area_len], marker='.', linestyle='-')
    ax.plot(x_line, y_line, linestyle='--')
    ax.set_ylim(top=8.21)
    ax.set_xlim(left=0)
    ax.grid()
    ax.set_xlabel('V, B')
    ax.set_ylabel('I, A')
    plt.tight_layout()

    axins = zoomed_inset_axes(ax, 2, loc=1)
    axins.plot(e_voltage[:area_len], e_current[:area_len], marker='.', linestyle='-')
    axins.plot(x_line, y_line, linestyle='--')
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid()
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec='0.5')

    """axins = zoomed_inset_axes(ax, 1.7, loc=3)
    axins.plot(e_voltage[:area_len], e_current[:area_len], marker='.', linestyle='-')
    axins.plot(x_line, y_line, linestyle='--')
    axins.set_xlim(8, 12)
    axins.set_ylim(8.09, 8.12)
    axins.grid()
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=1, loc2=4, fc='none', ec='0.5')"""

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
    plt.ylim(bottom=0)
    plt.axvline(e_voltage[area_len], linestyle='--', color='black')
    plt.grid()
    plt.xlabel('V, B')
    plt.ylabel('I, A')
    plt.tight_layout()
    plt.show()


def plot_power_curve_window():
    sp = SolarPanel(cell_num=54, Voc=32.9, Isc=8.21, a1=1.0, a2=1.2,
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
    # plt.plot(e_voltage, window, linestyle='--')
    plt.plot(e_voltage, power_window, linestyle='-')
    plt.plot(e_voltage, power_window ** 2, linestyle='--')
    plt.plot(e_voltage, power_window ** 3, linestyle='-.')
    plt.plot(e_voltage, power_window ** 4, linestyle=':')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid()
    plt.xlabel('V, B')
    plt.ylabel('$w^{k}$')
    plt.tight_layout()
    plt.show()


def plot_high_order_power_curve():
    sp = SolarPanel(cell_num=54, Voc=32.9, Isc=8.21, a1=1.0, a2=1.2,
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

    plt.ylim(bottom=0)
    plt.xlim(left=e_voltage[first_idx])
    plt.grid()
    plt.xlabel('V, B')
    plt.ylabel('I, A')
    plt.tight_layout()
    plt.show()


def plot_gaussian_window(use_alpha_enum=True, use_one_diode_model=False, recalculate=False):
    sp = SolarPanel(cell_num=54, Voc=32.9, Isc=8.21, a1=1.0, a2=1.2,
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

    if use_alpha_enum:
        sp.alpha_enum(window, use_one_diode_model=use_one_diode_model, recalculate=recalculate)
    else:
        if use_one_diode_model:
            sp.a2 = 10 ** 10
        sp.res_enum(window)

    print(sp.current[-1])

    plt.plot(e_voltage[first_idx:], e_current[first_idx:], linestyle='-')
    plt.plot(sp.voltage[first_idx:], sp.current[first_idx:], linestyle='--')
    plt.plot(e_voltage[first_idx:], window[first_idx:], linestyle='-.')
    plt.axvline(e_voltage[second_idx], linestyle='--', color='black')

    plt.ylim(bottom=0)
    plt.xlim(left=e_voltage[first_idx])
    plt.grid()
    plt.xlabel('V, B')
    plt.ylabel('I, A')
    plt.tight_layout()
    plt.show()

    return sp.current


def plot_error_bar_graph():
    error_names = ("$w^{1}$", "$w^{4}$", "$w^{4}_{norm}$", "$w_{g}$")

    # Alpha enum
    """bin_vals = {
        '$wRMSE$': (0.09199, 0.04377, 0.10426, 0.10119),
        '$MAPE$': (0.01859, 0.08241, 0.01859, 0.01859),
        '$P$': [i * 15 for i in (0.0017105123, 0.0036067037, 0.0019386657, 0.0018815802)],
    }"""

    # Res enum
    bin_vals = {
        '$wRMSE$': (0.01816, 0.02007, 0.02038, 0.01977),
        '$MAPE$': (0.00578, 0.00619, 0.00581, 0.00582),
        '$P$': [i * 100 for i in (0.0001049787, 0.0001242272, 0.0001183182, 0.0001150385)],
    }

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
    ax.set_xlabel('Тип вагової функції')
    ax.set_xticks(x + width, error_names)
    ax.legend(loc='upper right', ncols=3)

    # ax.set_ylim(0, 0.125) # Alpha enum
    ax.set_ylim(0, 0.025) # Res enum
    plt.show()


def plot_different_alpha_enum_approaches():
    sp = SolarPanel(cell_num=54, Voc=32.9, Isc=8.21, a1=1.0, a2=1.2,
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
    print(sp)

    sp.alpha_enum(window, use_one_diode_model=False, recalculate=True)
    current_2 = sp.current
    print(sp)

    mape1 = find_mape(e_current, current_1)
    mape2 = find_mape(e_current, current_2)
    print(np.sum(mape1) / len(mape1))
    print(np.sum(mape2) / len(mape2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # area 2
    print(sp.voltage[first_idx:second_idx])

    ax1.plot(e_voltage[first_idx:second_idx], e_current[first_idx:second_idx], linestyle='-')
    ax1.plot(sp.voltage[first_idx:second_idx], current_2[first_idx:second_idx], linestyle='--')
    ax1.plot(sp.voltage[first_idx:second_idx], current_1[first_idx:second_idx], linestyle=':',
             color='k')
    ax1.grid()
    ax1.set_xlabel('V, B \na')
    ax1.set_ylabel('I, A')
    ax1.set_ylim(bottom=min(e_current[second_idx], current_1[second_idx], current_2[second_idx]))
    ax1.set_xlim(left=e_voltage[first_idx], right=e_voltage[second_idx])

    # area 3
    ax2.plot(e_voltage[second_idx:], e_current[second_idx:], linestyle='-')
    ax2.plot(sp.voltage[second_idx:], current_1[second_idx:], linestyle='--')
    ax2.plot(sp.voltage[second_idx:], current_2[second_idx:], linestyle=':', color='k')
    # ax2.plot(sp.voltage[second_idx:], mape1[second_idx:])
    # ax2.plot(sp.voltage[second_idx:], mape2[second_idx:])
    ax2.grid()
    ax2.set_xlabel('V, B \nб')
    ax2.set_ylabel('I, A')
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(left=e_voltage[second_idx])

    plt.tight_layout()
    plt.show()


def plot_res_enum_different_windows():
    sp = SolarPanel(cell_num=54, Voc=32.9, Isc=8.21, a1=1.0, a2=1.2,
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
    sp = SolarPanel(cell_num=54, Voc=32.9, Isc=8.21,
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

    sp.Rs = Rs
    sp.Rp = Rp

    # sp.I0 = sp.find_backward_current()
    current_raw = sp.fixed_point_method(Rs, Rp)
    original_len = len(current_raw)
    current = [i for i in current_raw if i >= 0]
    current.append(current_raw[len(current)])
    return current, original_len - len(current)


def get_second_method_current(Ipv=8.223, I0=2.152e-9, Rs=0.308, Rp=193.049):
    sp = SolarPanel(cell_num=54, Voc=32.9, Isc=8.21,
                    Pmpp=200, Impp=7.61, Vmpp=26.3,
                    Ku=-0.123, Ki=3.18 / 1000, a1=1.076, a2=10 ** 10,
                    I0=I0, Ipv=Ipv,
                    thermal_coefficient_type='absolute', G=1000,
                    experimental_V=e_voltage,
                    experimental_I=e_current,
                    is_experimental_data_provided=True,
                    mode='overall')
    sp.Rs = Rs
    sp.Rp = Rp
    sp.I0 = sp.find_backward_current()

    print(sp)

    current_raw = sp.fixed_point_method(Rs, Rp)
    original_len = len(current_raw)
    current = [i for i in current_raw if i >= 0]
    # current.append(current_raw[len(current)])
    return current, original_len - len(current)


def proposed_model_current():
    current_arr = []
    for idx, filename in enumerate(CUR_FN):
        if filename in listdir(OUTPUT_FILES_DIR):
            print(f'Found {filename} current. Loading')
            current_arr.append(read_file(filename))
        else:
            match idx:
                case 0:
                    current_arr.append(plot_gaussian_window(use_alpha_enum=True,
                                                            use_one_diode_model=True))
                    write_file(A_ENUM_1D_FN, current_arr[-1])
                case 1:
                    current_arr.append(plot_gaussian_window(use_alpha_enum=True,
                                                            use_one_diode_model=False))
                    write_file(A_ENUM_2D_FN, current_arr[-1])
                case 2:
                    current_arr.append(plot_gaussian_window(use_alpha_enum=False,
                                                            use_one_diode_model=True))
                    write_file(R_ENUM_1D_FN, current_arr[-1])
                case 3:
                    current_arr.append(plot_gaussian_window(use_alpha_enum=False,
                                                            use_one_diode_model=False))
                    write_file(R_ENUM_2D_FN, current_arr[-1])
    return current_arr


def plot_proposed_model_current_area1():
    current_arr = proposed_model_current()
    alpha_cur = current_arr[0]
    res_cur = current_arr[2]

    y_ticks = np.arange(8, 8.205, 0.05)
    zoom_area_left = 0
    zoom_area_right = get_area_edge_val(e_voltage, 5 / e_voltage[-1])

    fig, ax = plt.subplots(1, 1)

    ax.plot(e_voltage[:first_idx], e_current[:first_idx])
    ax.plot(e_voltage[:first_idx], alpha_cur[:first_idx], linestyle='--')
    ax.plot(e_voltage[:first_idx], res_cur[:first_idx], linestyle=':', color='k')
    ax.scatter(e_voltage[0], e_current[0], s=30, c='r', zorder=2)
    ax.grid()
    ax.set_xlabel('V, B')
    ax.set_ylabel('I, A')
    ax.set_yticks(y_ticks)
    ax.set_ylim(bottom=min(e_current[first_idx], alpha_cur[first_idx], res_cur[first_idx]))
    ax.set_xlim(left=0, right=e_voltage[first_idx - 1])
    plt.tight_layout()

    # axin
    axins = zoomed_inset_axes(ax, 2.2, loc=3)
    axins.plot(e_voltage[zoom_area_left:zoom_area_right + 1],
               e_current[zoom_area_left:zoom_area_right + 1])
    axins.plot(e_voltage[zoom_area_left:zoom_area_right + 1],
               alpha_cur[zoom_area_left:zoom_area_right + 1], linestyle='--')
    axins.plot(e_voltage[zoom_area_left:zoom_area_right + 1],
               res_cur[zoom_area_left:zoom_area_right + 1], linestyle=':', color='k')
    axins.scatter(e_voltage[0], e_current[0], s=30, c='r', zorder=2)
    axins.set_xlim(e_voltage[zoom_area_left], e_voltage[zoom_area_right])
    axins.set_ylim(e_current[zoom_area_right], e_current[zoom_area_left])
    axins.grid()
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=1, fc='none', ec='0.5')
    plt.show()


def plot_proposed_model_current_area2():
    current_arr = proposed_model_current()
    alpha_cur = current_arr[0]
    res_cur = current_arr[3]

    zoom_area_left = get_area_edge_val(e_voltage, (e_voltage[max_power_idx] - 2) / e_voltage[-1])
    zoom_area_right = get_area_edge_val(e_voltage, (e_voltage[max_power_idx] + 2) / e_voltage[-1])

    fig, ax = plt.subplots(1, 1)

    ax.plot(e_voltage[first_idx:second_idx], e_current[first_idx:second_idx])
    ax.plot(e_voltage[first_idx:second_idx], alpha_cur[first_idx:second_idx], linestyle='--')
    ax.plot(e_voltage[first_idx:second_idx], res_cur[first_idx:second_idx], linestyle=':',
            color='k')
    ax.scatter(e_voltage[max_power_idx], e_current[max_power_idx], s=30, c='r', zorder=2)
    ax.grid()
    ax.set_xlabel('V, B')
    ax.set_ylabel('I, A')
    ax.set_ylim(
        bottom=min(e_current[second_idx - 1], alpha_cur[second_idx - 1], res_cur[second_idx - 1]))
    ax.set_xlim(left=e_voltage[first_idx], right=e_voltage[second_idx])
    plt.tight_layout()

    # axin
    axins = zoomed_inset_axes(ax, 1.75, loc=3)
    axins.plot(e_voltage, e_current)
    axins.plot(e_voltage, alpha_cur, linestyle='--')
    axins.plot(e_voltage, res_cur, linestyle=':', color='k')
    axins.scatter(e_voltage[max_power_idx], e_current[max_power_idx], s=30, c='r', zorder=2)
    axins.set_xlim(e_voltage[zoom_area_left], e_voltage[zoom_area_right])
    axins.set_ylim(e_current[zoom_area_right], e_current[zoom_area_left])
    axins.grid()
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='0.5')
    plt.show()


def plot_proposed_model_current_area3():
    current_arr = proposed_model_current()
    alpha_cur = current_arr[0]
    res_cur = current_arr[3]

    zoom_area_left = get_area_edge_val(e_voltage, (e_voltage[-1] - 0.5) / e_voltage[-1])
    zoom_area_right = get_area_edge_val(e_voltage, 1) + 1

    fig, ax = plt.subplots(1, 1)

    ax.plot(e_voltage[second_idx:], e_current[second_idx:], linestyle='-')
    ax.plot(e_voltage[second_idx:], alpha_cur[second_idx:], linestyle='--')
    ax.plot(e_voltage[second_idx:], res_cur[second_idx:], linestyle=':', color='k')
    ax.scatter(e_voltage[-1], e_current[-1], s=30, c='r', zorder=2)
    ax.grid()
    ax.set_xlabel('V, B')
    ax.set_ylabel('I, A')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=e_voltage[second_idx])
    plt.tight_layout()

    # axin
    axins = zoomed_inset_axes(ax, 1.4, loc=3)
    axins.plot(e_voltage, e_current)
    axins.plot(e_voltage, alpha_cur, linestyle='--')
    axins.plot(e_voltage, res_cur, linestyle=':', color='k')
    axins.scatter(e_voltage[-1], e_current[-1], s=30, c='r', zorder=2)
    axins.scatter(e_voltage[max_power_idx], e_current[max_power_idx], s=30, c='r', zorder=2)
    axins.set_xlim(e_voltage[zoom_area_left], e_voltage[zoom_area_right])
    axins.set_ylim(e_current[zoom_area_right], e_current[zoom_area_left])
    axins.grid()
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec='0.5')
    plt.show()


def plot_proposed_model_current_total():
    area_1_idx = get_area_edge_val(e_voltage, 0.65)
    area_2_idx = get_area_edge_val(e_voltage, 0.92)

    current_arr = proposed_model_current()
    alpha_cur = current_arr[0]
    res_cur = current_arr[3]

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_TALL)

    ax.plot(e_voltage, e_current, linestyle='-')
    ax.plot(e_voltage, alpha_cur, linestyle='--')
    ax.plot(e_voltage, res_cur, linestyle=':', color='k')
    ax.axvline(e_voltage[area_1_idx], linestyle='--', color='black')
    ax.axvline(e_voltage[area_2_idx], linestyle='--', color='black')
    ax.scatter(e_voltage[max_power_idx], e_current[max_power_idx], s=30, c='r', zorder=2)
    ax.grid()
    ax.set_xlabel('V, B')
    ax.set_ylabel('I, A')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.tight_layout()

    # area 2
    zoom_area_left = get_area_edge_val(e_voltage, (e_voltage[max_power_idx] - 2) / e_voltage[-1]) - 1
    zoom_area_right = get_area_edge_val(e_voltage, (e_voltage[max_power_idx] + 2) / e_voltage[-1])

    axins = zoomed_inset_axes(ax, 5, loc=3)
    axins.plot(e_voltage, e_current)
    axins.plot(e_voltage, alpha_cur, linestyle='--')
    axins.plot(e_voltage, res_cur, linestyle=':', color='k')
    axins.scatter(e_voltage[max_power_idx], e_current[max_power_idx], s=30, c='r', zorder=2)
    axins.set_xlim(e_voltage[zoom_area_left], e_voltage[zoom_area_right])
    axins.set_ylim(e_current[zoom_area_right], e_current[zoom_area_left])
    axins.grid()
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.5')

    plt.show()


def compare_methods():
    c_1, shift_1 = get_first_method_current()
    c_21, shift_21 = get_second_method_current()
    c_22, shift_22 = get_second_method_current(8.21, 2.195e-9, 0.284, 157.853)

    current_arr = proposed_model_current()
    res_enum = current_arr[0]

    plt.plot(e_voltage, e_current, linestyle='-')
    plt.plot(e_voltage[:-shift_1], c_1, linestyle='--')
    plt.plot(e_voltage, c_21, linestyle='--')
    plt.plot(e_voltage, c_22, linestyle='--')
    # plt.plot(e_voltage[:-shift_21][second_idx:], c_21[second_idx:], linestyle='-.')
    # plt.plot(e_voltage[:-shift_22][second_idx:], c_22[second_idx:], linestyle=':')
    # plt.plot(e_voltage[second_idx:], current_1[second_idx:])

    #for c in current_arr:
        # plt.plot(e_voltage[second_idx:], c[second_idx:])

    plt.scatter([32.9], [0])
    plt.axvline(e_voltage[second_idx], linestyle='--', color='black')

    plt.grid()
    plt.xlabel('V, B')
    plt.ylabel('I, A')
    plt.tight_layout()
    plt.show()


def find_mape(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    error = []
    for exper_val, approx_val, in zip(arr1, arr2):
        if exper_val == 0:
            error.append(0)
        else:
            error.append(abs((exper_val - approx_val) / exper_val))
    return error


def find_rmse(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    mse = 0
    for i, j in zip(arr1, arr2):
        mse += (i - j) ** 2
    return np.sqrt(mse / len(arr1))


def read_matlab_txt():
    with open(INP_FILES_DIR + "MyFile.txt") as file:
        next(file)

        v_vals = []
        i_vals = []
        for line in file:
            v, i = [float(i) for i in line.split(',')]

            if i <= 0:
                break

            v_vals.append(v)
            i_vals.append(i)

    with open(INP_FILES_DIR + "1000.txt") as file:
        v_vals_n = []
        i_vals_n = []
        for line in file:
            v, i = [float(i) for i in line.split(', ')]

            if i <= 0:
                break

            v_vals_n.append(v)
            i_vals_n.append(i)

    c1, shift1 = get_first_method_current()

    error = find_mape(c1, e_current)

    # plt.plot(v_vals, i_vals)
    plt.plot(e_voltage[:-shift1], c1)
    plt.plot(e_voltage, e_current, ':', color='k')
    plt.plot(e_voltage[:-6], error)
    # plt.plot(v_vals_n, i_vals_n, '--')
    plt.grid()
    plt.show()


def write_file(filename, arr):
    with open(OUTPUT_FILES_DIR + filename, 'w') as file:
        for val in arr:
            file.write(f'{val}\n')
    print(f'Done writing {filename}')


def read_file(filename):
    with open(OUTPUT_FILES_DIR + filename) as file:
        output_data = []
        for line in file.readlines():
            output_data.append(float(line))
    return output_data


if __name__ == '__main__':
    e_voltage, e_current = get_data(KC200GT_DIR)
    first_idx = get_area_edge_val(e_voltage, 0.65)
    second_idx = get_area_edge_val(e_voltage, 0.95)
    max_power_idx = get_max_power_idx(e_voltage, e_current)

    # main()
    # plot_iv_curve_split()
    # plot_slope_first()
    # plot_slope_second()
    # plot_power_curve_window()
    # plot_high_order_power_curve()
    # plot_gaussian_window()
    # plot_error_bar_graph()
    # plot_different_alpha_enum_approaches()
    # plot_res_enum_different_windows()
    # plot_proposed_model_current_area1()
    # plot_proposed_model_current_area2()
    # plot_proposed_model_current_area3()
    # plot_proposed_model_current_total()
    # proposed_model_current()
    # compare_methods()

    # read_matlab_txt()
