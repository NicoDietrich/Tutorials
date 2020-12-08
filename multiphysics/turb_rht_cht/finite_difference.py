# from multiprocessing import Pool
import datetime
import time
import os
import csv
import subprocess
import re
import shutil
import logging
import math
import argparse
import glob


class SolveEquationError(RuntimeError):
    def __init__(self, *args):
        self.args = args


class ArmijoError(RuntimeError):
    def __init__(self, *args):
        self.args = args


DATA_DIR = './data/'
if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR)

FAIL_DIR = DATA_DIR + 'fail/'
if not os.path.isdir(FAIL_DIR):
    os.makedirs(FAIL_DIR)

logging.basicConfig(
        filename=DATA_DIR + 'fd.log',
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s:%(message)s',
        datefmt='%Y.%m.%d_%H:%M:%S'
)

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-n', dest='n', type=int, default=2,
                    help='Argument that is passed as mpirun -n, default is 2')
args = parser.parse_args()

MPI_n = args.n

LOGGER.info("")
LOGGER.info("")
LOGGER.info(f"Use mpi_run -n {MPI_n} to solve equations")

# Files are Gloabl
deformed_ffd_config = 'manual_deformed_ffd.cfg'
state_cfg = 'turbulent_rht_cht.cfg'
state_cfg = 'turbulent_rht_cht.cfg'
flow_cfg = 'config_flow_rht.cfg'
solid_cfg = 'config_solid_cht.cfg'
adj_cfg = 'turbulent_rht_cht_adjoint.cfg'
state_sol_file = 'turbulent_rht_cht.csv'
orig_flow_mesh = 'mesh_flow_ffd.su2'
deformed_flow_mesh = 'mesh_flow_ffd_deform.su2'
orig_ffd_box = 'config_ffd.cfg'
grad_file = 'of_grad.dat'


def compile_ffd(config):
    LOGGER.info(f"Compile FFD box using {config}")
    subprocess.run(['SU2_DEF', f'{config}'], check=True, stdout=subprocess.DEVNULL)


def solve_state(config, mpi_n, outfile_path='output.txt'):
    LOGGER.info(f"Solve State Equation using {config} and mpirun -n {mpi_n}")
    tic = time.perf_counter()
    try:
        with open(outfile_path, 'w') as outfile:
            subprocess.run(['mpirun', '-n', f'{mpi_n}', 'SU2_CFD', f'{config}'],
                           check=True, stdout=outfile, stderr=outfile)
    except subprocess.CalledProcessError:
        toc = time.perf_counter()
        time_seconds = toc - tic
        time_minutes = time_seconds/60.
        LOGGER.warn(f"Could not solve State Equation after {time_minutes:.2f} minutes")
        raise SolveEquationError

    toc = time.perf_counter()
    time_seconds = toc - tic
    time_minutes = time_seconds/60.
    LOGGER.info(f"Done in {time_minutes:.2f} minutes")
    os.remove(outfile_path)
    return


def solve_adj_state(config):
    rename_state_files()
    LOGGER.info("Solve Adjoint Equation")
    tic = time.perf_counter()
    try:
        with open('output.txt', 'w') as outfile:
            subprocess.run(['mpirun', '-n', f'{MPI_n}', 'SU2_CFD_AD', f'{config}'],
                           check=True, stdout=outfile, stderr=outfile)
    except subprocess.CalledProcessError:
        toc = time.perf_counter()
        time_seconds = toc - tic
        time_minutes = time_seconds/60.
        LOGGER.warn(f"Could not solve Adjoint Equation after {time_minutes:.2f} minutes")
        raise SolveEquationError

    toc = time.perf_counter()
    time_seconds = toc - tic
    time_minutes = time_seconds/60.
    LOGGER.info(f"Done in {time_minutes:.2f} minutes")
    os.remove('output.txt')
    return


def project_sensitivities(config):
    rename_adj_files()
    LOGGER.info("Project Sensitivities")
    subprocess.run(['SU2_DOT_AD', f'{config}'], check=True, stdout=subprocess.DEVNULL)


def extract_value(sol_file, index):
    tail = subprocess.Popen(['tail', '-n', '1', f'{sol_file}'], stdout=subprocess.PIPE)
    value_string = subprocess.check_output(
        ['awk', '-F', ',', f'{{ print ${index} }}'],
        stdin=tail.stdout
    )
    tail.wait()
    return float(value_string)


def rename_state_files():
    os.rename('restart_flow_rht_0.dat', 'solution_flow_rht_0.dat')
    os.rename('restart_solid_cht_1.dat', 'solution_solid_cht_1.dat')


def rename_adj_files():
    os.rename('restart_adj_flow_rht_totheat_0.dat', 'solution_adj_flow_rht_totheat_0.dat')
    os.rename('restart_adj_solid_cht_totheat_1.dat', 'solution_adj_solid_cht_totheat_1.dat')


def write_ffd_deformation(values, name):
    LOGGER.info(f"Generate {name}")
    assert len(values) == 24
    with open(name, 'r') as f:
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


def change_config_list(state_cfg, flow_cfg, solid_cfg):
    LOGGER.debug(f"Change List in {state_cfg} to ({flow_cfg},{solid_cfg})")
    with open(state_cfg, 'r') as f:
        lines = f.readlines()
    with open(state_cfg, 'w') as f:
        for line in lines:
            if re.match('^CONFIG_LIST', line):
                f.write(f'CONFIG_LIST = ({flow_cfg}, {solid_cfg})\n')
            else:
                f.write(line)


def change_output_files(config, name, index):
    LOGGER.debug(f"Change output files in {config} to {name} with index {index}")
    with open(config, 'r') as f:
        lines = f.readlines()
    with open(config, 'w') as f:
        for line in lines:
            if re.match('^SOLUTION_FILENAME', line):
                f.write(f'SOLUTION_FILENAME = solution_{name}_{index}\n')
            elif re.match('^RESTART_FILENAME', line):
                f.write(f'RESTART_FILENAME = restart_{name}_{index}\n')
            elif re.match('^VOLUME_FILENAME', line):
                f.write(f'VOLUME_FILENAME = volume_{name}_{index}\n')
            elif re.match('^CONV_FILENAME', line):
                f.write(f'CONV_FILENAME = history_{name}_{index}\n')
            else:
                f.write(line)


def change_out_mesh(config, meshfile):
    LOGGER.debug(f"Change Output Mesh in {config} to {meshfile}")
    with open(config, 'r') as f:
        lines = f.readlines()

    with open(config, 'w') as f:
        for line in lines:
            if re.match('^MESH_OUT_FILENAME', line):
                f.write(f'MESH_OUT_FILENAME = {meshfile}\n')
            else:
                f.write(line)


def read_sensitivities(dat_file):
    with open(dat_file, 'r') as f:
        lines = f.readlines()
    values = [float(s.strip('\n')) for s in lines[1:]]
    return values


def _central_difference(args):
    h = args[0]
    index = args[1]
    LOGGER.info(f"Index {index} with h={h}")

    tmp_state_cfg = str(index) + '_' + state_cfg
    tmp_flow_cfg = str(index) + '_' + flow_cfg
    tmp_solid_cfg = str(index) + '_' + solid_cfg
    tmp_ffd_cfg = str(index) + '_' + deformed_ffd_config
    tmp_sol_file = str(index) + '_' + state_sol_file

    shutil.copy(state_cfg, tmp_state_cfg)
    shutil.copy(deformed_ffd_config, tmp_ffd_cfg)
    shutil.copy(flow_cfg, tmp_flow_cfg)
    shutil.copy(solid_cfg, tmp_solid_cfg)

    tmp_deformed_mesh = f"{index}_deformed_flow_mesh.su2"
    change_out_mesh(tmp_ffd_cfg, tmp_deformed_mesh)

    change_config_list(tmp_state_cfg, tmp_flow_cfg, tmp_solid_cfg)
    change_mesh(tmp_flow_cfg, tmp_deformed_mesh)
    # change_mesh(tmp_solid_cfg, tmp_deformed_mesh)
    change_output_files(tmp_flow_cfg, 'flow_tmp', index)
    change_output_files(tmp_solid_cfg, 'solid_tmp', index)

    deformation = 24*[0]

    deformation[index] = h
    write_ffd_deformation(deformation, tmp_ffd_cfg)
    change_out_mesh(tmp_ffd_cfg, tmp_deformed_mesh)

    compile_ffd(tmp_ffd_cfg)
    solve_state(tmp_state_cfg, mpi_n=4, outfile_path=f'{index}_output.txt')
    F_xph = extract_value(tmp_sol_file, 11)

    deformation[index] = -h
    write_ffd_deformation(deformation, tmp_ffd_cfg)
    change_out_mesh(tmp_ffd_cfg, tmp_deformed_mesh)
    compile_ffd(tmp_ffd_cfg)
    solve_state(tmp_state_cfg, mpi_n=4, outfile_path=f'{index}_output.txt')
    F_xmh = extract_value(tmp_sol_file, 11)

    os.remove(tmp_state_cfg)
    os.remove(tmp_flow_cfg)
    os.remove(tmp_solid_cfg)
    os.remove(tmp_ffd_cfg)
    os.remove(tmp_sol_file)
    os.remove(tmp_deformed_mesh)

    return (F_xmh, F_xph)


def central_difference_verification():
    LOGGER.info("========= Start Central Difference Approx =========")

    results_file = DATA_DIR + 'gradient_test_perturbed.csv'

    LOGGER.info("Calculate Adjoint sensitivities")
    change_mesh(flow_cfg, orig_flow_mesh)
    compile_ffd(orig_ffd_box)
    solve_state(state_cfg, mpi_n=4)
    solve_adj_state(adj_cfg)
    project_sensitivities(adj_cfg)

    adj_sensitivities = read_sensitivities('of_grad.dat')
    change_mesh(flow_cfg, deformed_flow_mesh)

    fieldnames = ['index', 'h', 'F_xph', 'F_xmh', 'central_dif', 'adj_grad']

    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()

    for h in [1e-5, 1e-6, 1e-7]:
        LOGGER.info(f"Start verification for h={h}")

        H = [h for s in adj_sensitivities]
        res = map(_central_difference, zip(H, range(len(adj_sensitivities))))

        for f in glob.glob('*_tmp_*'):
            os.remove(f)

        for i, (F_xmh, F_xph) in enumerate(res):
            central_difference = (F_xph - F_xmh)/(2*h)
            data = {'index': i, 'h': h, 'F_xph': F_xph,
                    'F_xmh': F_xmh, 'central_dif': central_difference,
                    'adj_grad': adj_sensitivities[i]}
            with open(results_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames)
                writer.writerow(data)


def store_important_files(index):
    shutil.copy('./volume_flow_rht_0.vtu', DATA_DIR + f'volume_flow_{index}.vtu')
    shutil.copy('./volume_solid_cht_1.vtu', DATA_DIR + f'volume_solid_{index}.vtu')
    shutil.copy('surface_sens_0.vtu', DATA_DIR + f'surface_sens_line_{index}.vtu')
    shutil.copy('surface_sens_1.vtu', DATA_DIR + f'surface_sens_nothing_{index}.vtu')
    shutil.copy('volume_sens_0.vtu', DATA_DIR + f'surface_sens_flow_{index}.vtu')
    shutil.copy('volume_sens_1.vtu', DATA_DIR + f'surface_sens_solid_{index}.vtu')
    shutil.copy('of_grad.dat', DATA_DIR + f'of_grad_{index}.csv')


def store_functional_value(index, value, functional_file):
    with open(functional_file, 'a') as f:
        f.write(f"{index}, {value}\n")


def armijo(prev_deformation, sensitivities, max_iterations, J_i):
    gamma = 1e-4

    # in finite dimensions the gradient g is the transposed derivative, d=g^T.
    # Assuming the sensitivites are the gradient, then the directional
    # derivative is g^T*g and therefore:
    LOGGER.debug("Starting Armijo Rule")
    LOGGER.debug(f"prev_deformation: {prev_deformation}")
    LOGGER.debug(f"sensitivities: {sensitivities}")
    directional_derivative = sum([s**2 for s in sensitivities])
    l2_norm = math.sqrt(directional_derivative)
    LOGGER.info(f"l2 norm perturbed sensitivities: {l2_norm}")

    initial_stepsize = 1./l2_norm*1./2**0
    stepsize = 2*initial_stepsize

    for j in range(max_iterations):
        LOGGER.info(f"== Armijo {j} ==")
        stepsize *= 1./2
        LOGGER.info(f"Stepsize = {stepsize} ==")

        deformation = [defo + stepsize*sens for defo, sens in zip(prev_deformation, sensitivities)]
        write_ffd_deformation(deformation, deformed_ffd_config)
        compile_ffd(deformed_ffd_config)
        # write_ffd_deformation(deformation, flow_cfg)

        try:
            solve_state(state_cfg, MPI_n)
        except SolveEquationError:
            LOGGER.warn("Could not solve State eq in Amijo")
            now_str = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
            shutil.move("output.txt", FAIL_DIR + now_str + "_state" + ".txt")
            continue
        J_ip1 = extract_value(state_sol_file, 11)

        # In traditional armijo the functional is minimized
        # and therefore dif = J_ip1 - J_i condidered
        dif = J_i - J_ip1
        tol = gamma*directional_derivative*stepsize
        LOGGER.debug(f'old fvalue - new fvalue = {J_i} - {J_ip1} = {dif:.4f} < {tol:.4f} = tol?')
        if dif < tol:
            LOGGER.debug("Armijo finished after {} steps".format(j+1))  # wg Start bei 0
            LOGGER.debug("Check Adjoint")
            try:
                solve_adj_state(adj_cfg)
            except SolveEquationError:
                LOGGER.warn("Armijo step accepted but adjoint not solvable, return to amijo")
                now_str = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
                shutil.move("output.txt", FAIL_DIR + now_str + "_adjoint" + ".txt")
            else:
                return J_ip1, deformation
    raise ArmijoError


def gradient_descent():

    LOGGER.info("========= starting gradient descent =========")

    functional_data_file = DATA_DIR + 'functional_evolution.csv'
    # if os.path.isfile(functional_data_file):
    #     os.remove(functional_data_file)
    with open(functional_data_file, 'a') as f:
        f.write("i, fvalue\n")

    opt_steps = 8
    max_armijo_it = 10

    compile_ffd(orig_ffd_box)
    prev_deformation = [0.0, 0.05, 0.1, 0.14, 0.16, 0.17,
                        0.175, 0.17, 0.16, 0.14, 0.1, 0.05,
                        0.0, -0.05, -0.1, -0.14, -0.16, -0.17,
                        -0.175, -0.17, -0.16, -0.14, -0.1, -0.05]

    # prev_deformation = [0 for i in range(24)]
    # write_ffd_deformation(prev_deformation, flow_cfg)

    shutil.copy('./sample_ffd_deform.cfg',  deformed_ffd_config)

    write_ffd_deformation(prev_deformation, deformed_ffd_config)
    compile_ffd(deformed_ffd_config)

    change_mesh(flow_cfg, deformed_flow_mesh)
    solve_state(state_cfg, MPI_n)

    J_i = extract_value(state_sol_file, 11)

    solve_adj_state(adj_cfg)

    for i in range(1, opt_steps):

        LOGGER.info(f"===== Opt iteration {i} =====")
        store_functional_value(i-1, J_i, functional_data_file)
        project_sensitivities(adj_cfg)
        store_important_files(index=i-1)
        if i == 1:
            # In the optimization loop the equations are only solved on the deformed
            change_mesh(flow_cfg, deformed_flow_mesh)

        ref_sensitivities = read_sensitivities(grad_file)
        # Sensitivities are defined on reference geometry and "include the previous perturbations"
        # we therefore have to substract them to get the actual gradient information
        # at the deformed domain
        local_sensitivities = [s - d for d, s in zip(prev_deformation, ref_sensitivities)]
        try:
            J_i, prev_deformation = armijo(
                prev_deformation, local_sensitivities, max_armijo_it, J_i
            )
        except ArmijoError:
            LOGGER.error("Amijo failed, exit")
            return

    store_functional_value(opt_steps, J_i, functional_data_file)
    store_important_files(index=opt_steps)


if __name__ == '__main__':
    central_difference_verification()
    # gradient_descent()
    LOGGER.info("Finished")
