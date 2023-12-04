def replace_str(initial_str: str, str_to_replace: str,
                replace_with: str) -> str:
    while str_to_replace in initial_str:
        initial_str = initial_str.replace(str_to_replace, replace_with)
    return initial_str


def get_data(file):
    x_coord, y_coord = [], []
    for line in file:
        buff = list(map(float, line.strip(" ").split()))
        x_coord.append(buff[0])
        y_coord.append(buff[1])
    return x_coord, y_coord


def write_in_file(file, *iterables):
    for x_, y_ in zip(*iterables):
        file.write(f"{x_:.20f}\t\t{y_:.20f}\n")
    file.write(f"\n\n")


def read_file(filename, sep=',', outp_type=float):
    with open(filename) as file:
        lines = file.readlines()
        res = []
        for line in lines:
            res += [outp_type(i) for i in line.split(sep) if line]
            
    return res


def write_file(filename, data, sep=' '):
    with open(filename, 'w') as file:
        for line in data:
            str_line = str(line) + '\n' # sep.join(line) + '\n'
            file.write(str_line)


def rmse(arr1, arr2):
    diff = (arr1 - arr2) ** 2
    mse = np.sum(diff) / arr1.size
    rmse = np.sqrt(mse)
    return rmse


def minimize(init_cur, eta_min=0.5, eta_max=3, steps=500):
    min_error = 10 ** 10
    
    for eta in np.linspace(eta_min, eta_max, steps):
        approx = init_cur * np.exp(v / v_t / eta - 1)
        error = rmse(i, approx)
        errors.append(error)
    
        if error < min_error:
            min_error = error
            min_eta = eta
        
    return min_eta, min_error


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1,
                       length=100, fill='█', print_end='\r'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()
