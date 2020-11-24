import time
import os
import csv
import subprocess
import re
import shutil
import logging
import math


class SolveEquationError(RuntimeError):
    def __init__(self, *args):
        self.args = args


DATA_DIR = './data/'
if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR)

logging.basicConfig(filename=DATA_DIR + 'fd.log', level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

MPI_n = 2

deformed_ffd_box = 'deform_ffd.cfg'
state_cfg = 'turbulent_rht_cht.cfg'

state_cfg = 'turbulent_rht_cht.cfg'
flow_cfg = 'config_flow_rht.cfg'
solid_cfg = 'config_solid_cht.cfg'

adj_cfg = 'turbulent_rht_cht_adjoint.cfg'
state_sol_file = 'turbulent_rht_cht.csv'
orig_flow_mesh = 'mesh_flow_ffd.su2'
deformed_flow_mesh = 'mesh_flow_ffd_deform.su2'

orig_ffd_box = 'config_ffd.cfg'
grad_file= 'of_grad.dat'

def compile_ffd(config):
    LOGGER.info(f"Compile FFD box using {config}")
    subprocess.run(['SU2_DEF', f'{config}'], check=True, stdout=subprocess.DEVNULL)


def solve_state(config):
    LOGGER.info("Solve State Equation")
    tic = time.perf_counter()
    try:
        with open('output_primal.txt', 'w') as outfile:
            subprocess.run(['mpirun', '-n', f'{MPI_n}', 'SU2_CFD', f'{config}'],
                    check=True, stdout=outfile, stderr=outfile)
    except subprocess.CalledProcessError:
        LOGGER.warn("Could not solve State Equation, see output_primal.txt")
        raise SolveEquationError

    toc = time.perf_counter()
    time_seconds = toc - tic
    time_minutes = time_seconds/60.
    LOGGER.info(f"Done in {time_seconds} seconds or {time_minutes:.2f} minutes")
    os.remove('output_primal.txt')
    return


def solve_adj_state(config):
    LOGGER.info("Solve Adjoint Equation")
    tic = time.perf_counter()
    try:
        with open('output_adjoint.txt', 'w') as outfile:
            subprocess.run(['mpirun', '-n', f'{MPI_n}', 'SU2_CFD_AD', f'{config}'],
                    check=True, stdout=outfile, stderr=outfile)
    except subprocess.CalledProcessError:
        LOGGER.warn("Could not solve Adjoint Equation, see output_adjoint.txt")
        raise SolveEquationError

    toc = time.perf_counter()
    time_seconds = toc - tic
    time_minutes = time_seconds/60.
    LOGGER.info(f"Done in {time_seconds} seconds or {time_minutes:.2f} minutes")
    os.remove('output_adjoint.txt')
    return


def project_sensitivities(config):
    LOGGER.info("Project Sensitivities")
    subprocess.run(['SU2_DOT_AD', f'{config}'], check=True, stdout=subprocess.DEVNULL)


def extract_value(sol_file, index):
    tail = subprocess.Popen(['tail', '-n', '1', f'{sol_file}'], stdout=subprocess.PIPE)
    value_string = subprocess.check_output(['awk', '-F', ',', f'{{ print ${index} }}'], stdin=tail.stdout)
    tail.wait()
    return float(value_string)


def rename_state_files():
    os.rename('restart_flow_rht_0.dat', 'solution_flow_rht_0.dat')
    os.rename('restart_solid_cht_1.dat', 'solution_solid_cht_1.dat')


def rename_adj_files():
    os.rename('restart_adj_flow_rht_totheat_0.dat', 'solution_adj_flow_rht_totheat_0.dat')
    os.rename('restart_adj_solid_cht_totheat_1.dat', 'solution_adj_solid_cht_totheat_1.dat')


def get_ffd_deformation(values, name):
    LOGGER.info(f"Generate {name}")
    sample_file = 'sample_ffd_deform.cfg'
    assert len(values) == 24
    with open(sample_file, 'r') as f:
        lines = f.readlines()
    with open(name, 'w') as f:
        for line in lines:
            if re.match('^DV_VALUE', line):
                f.write('DV_VALUE = ' + ', '.join([str(val) for val in values]) + '\n')
            else:
                f.write(line)


def change_mesh(config, meshfile):
    LOGGER.debug(f"Change Mesh in {config} to {meshfile}")
    with open(config, 'r') as f:
        lines = f.readlines()

    with open(config, 'w') as f:
        for line in lines:
            if re.match('^MESH_FILENAME', line):
                f.write(f'MESH_FILENAME = {meshfile}\n')
            else:
                f.write(line)


def read_sensitivities(dat_file):
    with open(dat_file, 'r') as f:
        lines = f.readlines()
    values = [float(s.strip('\n')) for s in lines[1:]]
    return values


def central_difference_verification():
    LOGGER.info("\n ========= Start Central Difference Approx =========\n")
    state_cfg = 'turbulent_rht_cht.cfg'
    flow_cfg = 'config_flow_rht.cfg'
    solid_cfg = 'config_solid_cht.cfg'

    adj_cfg = 'turbulent_rht_cht_adjoint.cfg'
    state_sol_file = 'turbulent_rht_cht.csv'
    orig_flow_mesh = 'mesh_flow_ffd.su2'
    deformed_flow_mesh = 'mesh_flow_ffd_deform.su2'

    orig_ffd_box = 'config_ffd.cfg'
    deformed_ffd_box = 'deform_ffd.cfg'

    results_file = DATA_DIR + 'results_central_dif.csv'


    LOGGER.info("Calculate Adjoint sensitivities")
    change_mesh(flow_cfg, orig_flow_mesh)
    compile_ffd(orig_ffd_box)
    solve_state(state_cfg)
    rename_state_files()
    solve_adj_state(adj_cfg)
    rename_adj_files()
    project_sensitivities(adj_cfg)

    adj_sensitivities = read_sensitivities('of_grad.dat')
    change_mesh(flow_cfg, deformed_flow_mesh)

    fieldnames = ['index', 'h', 'F_xph', 'F_xmh', 'central_dif', 'adj_grad']

    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()

    h = 1e-6
    for h in [1e-4, 1e-5,]:  # 1e-6, 1e-7]:
        LOGGER.info(f"Start verification for h={h}")
        for i, adj_grad in enumerate(adj_sensitivities[:2]):
            LOGGER.info(f"Point {i}")
            deformation = 24*[0]
            deformation[i] = h
            get_ffd_deformation(deformation, deformed_ffd_box)
            compile_ffd(deformed_ffd_box)
            solve_state(state_cfg)
            F_xph = extract_value(state_sol_file, 11)

            deformation[i] = -h
            get_ffd_deformation(deformation, deformed_ffd_box)
            compile_ffd(deformed_ffd_box)
            solve_state(state_cfg)
            F_xmh = extract_value(state_sol_file, 11)

            central_difference = (F_xph - F_xmh)/(2*h)

            data = {'index': i, 'h': h, 'F_xph': F_xph,
                'F_xmh': F_xmh, 'central_dif': central_difference,
                'adj_grad': adj_grad}

            with open(results_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames)
                writer.writerow(data)


def store_important_files(index):
    shutil.copy('./volume_flow_rht_0.vtu', DATA_DIR + f'volume_flow_{index}.vtu')
    shutil.copy('./volume_solid_cht_1.vtu', DATA_DIR + f'volume_solid_{index}.vtu')
    shutil.copy('of_grad.dat', DATA_DIR + f'of_grad_{index}.csv')
    shutil.copy('surface_sens_0.vtu', DATA_DIR + f'surface_sens_line_{index}.vtu')
    shutil.copy('surface_sens_1.vtu', DATA_DIR + f'surface_sens_nothing_{index}.vtu')
    shutil.copy('volume_sens_0.vtu', DATA_DIR + f'surface_sens_flow_{index}.vtu')
    shutil.copy('volume_sens_1.vtu', DATA_DIR + f'surface_sens_solid_{index}.vtu')
    # shutil.copy('ffd_boxes_0.vtk', f'opt_iterations/ffd_boxes_upper_{index}.vtk')
    # shutil.copy('ffd_boxes_1.vtk', f'opt_iterations/ffd_boxes_lower_{index}.vtk')


def store_functional_value(index, value, functional_file):
    with open(functional_file, 'a') as f:
        f.write(f"{index}, {value}\n")


def armijo(prev_deformation, sensitivities, max_iterations, J_i):
    norm = math.sqrt(sum([s**2 for s in sensitivities]))
    initial_stepsize = 1./norm*1./2**4
    stepsize = 2*initial_stepsize
    gamma = 1e-4
    armijo_successful = False
    failed_because_mesh_quality = False
    J_ip1 = None
    for j in range(max_iterations):
        stepsize *= 1./2

        deformation = [defo + stepsize*sens for defo,sens in zip(prev_deformation, sensitivities)]
        get_ffd_deformation(deformation, deformed_ffd_box)
        compile_ffd(deformed_ffd_box)

        try:
            solve_state(state_cfg)
        except SolveEquationError:
            number = j+1
            LOGGER.warn(f"Could not solve State eq in Amijo, {number}/{max_iterations} failed")
            continue

        J_ip1 = extract_value(state_sol_file, 11)

        directional_derivative = 1
        tol = gamma*directional_derivative*stepsize

        dif = J_ip1 - J_i
        LOGGER.debug(f'new fvalue - old fvalue = {dif:.4f} < {tol:.4f} = tol?')

        if dif < tol:
            LOGGER.debug("Armijo finished after {} steps".format(j+1))  # wg Start bei 0
            LOGGER.debug("Check Adjoint")

            rename_state_files()
            try:
                solve_adj_state(adj_cfg)
            except SolveEquationError:
                LOGGER.warn("Armijo step accepted but adjoint not solvable, return to amijo")
            else:
                armijo_successful = True
                break
        else:
            number = j+1
            LOGGER.debug(f"Try again :(, {number}/{max_iterations} failed")

    return armijo_successful, J_ip1, deformation


def gradient_descent():

    LOGGER.info("\n ========= starting gradient descent ========= \n")

    functional_data_file = DATA_DIR + 'functional_evolution.csv'
    if os.path.isfile(functional_data_file):
        os.remove(functional_data_file)
    with open(functional_data_file, 'w') as f:
        f.write("i, fvalue\n")

    opt_steps = 3
    max_armijo_it = 5

    # change_mesh(flow_cfg, orig_flow_mesh)
    # compile_ffd(orig_ffd_box)
    # solve_state(state_cfg)
    # J_0 = extract_value(state_sol_file, 11)
    # J_0 = extract_value(state_sol_file, 11)
    # store_functional_value(0, J_0, functional_data_file)
    # rename_state_files()
    # solve_adj_state(adj_cfg)
    # rename_adj_files()
    # project_sensitivities(adj_cfg)
    J_0 = 113

    prev_deformation = [0 for i in range(24)]

    ref_sensitivities = read_sensitivities(grad_file)
    sensitivities = [s - d for d,s in zip(prev_deformation, ref_sensitivities)]
    norm = math.sqrt(sum([s**2 for s in sensitivities]))
    LOGGER.info(f"L2 norm sensitivities: {norm}")

    change_mesh(flow_cfg, deformed_flow_mesh)
    J_i = J_0
    for i in range(1, opt_steps):

        success, J_i, prev_deformation = armijo(prev_deformation, sensitivities, max_armijo_it, J_i)
        if not success:
            LOGGER.error("Amijo failed, exit")
            return
        store_functional_value(i, J_i, functional_data_file)
        store_important_files(index=i)

        ref_sensitivities = read_sensitivities(grad_file)
        sensitivities = [s - d for d,s in zip(prev_deformation, ref_sensitivities)]
        norm = math.sqrt(sum([s**2 for s in sensitivities]))
        LOGGER.info(f"L2 norm sensitivities: {norm}")

if __name__ == '__main__':
    # central_difference_verification()
    gradient_descent()

