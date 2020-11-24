import os
import csv
import subprocess
import re
import shutil
import logging
import math



DATA_DIR = './data/'
logging.basicConfig(filename=DATA_DIR + 'fd.log', level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)
MPI_n = 2

def compile_ffd(config):
    LOGGER.info(f"Compile FFD box using {config}")
    subprocess.run(['SU2_DEF', f'{config}'], check=True, stdout=subprocess.DEVNULL)


def solve_state(config):
    LOGGER.info("Solve State Equation")
    try:
        with open('output_primal.txt', 'w') as outfile:
            subprocess.run(['mpirun', '-n', f'{MPI_n}', 'SU2_CFD', f'{config}'],
                    check=True, stdout=outfile, stderr=outfile)
    except CalledProcessError:
        LOGGER.warn("Could not solve State Equation, see output_primal.txt")
        raise CalledProcessError
        return
    os.remove('output_primal.txt')
    return


def solve_adj_state(config):
    LOGGER.info("Solve Adjoint Equation")
    try:
        with open('output_adjoint.txt', 'w') as outfile:
            subprocess.run(['mpirun', '-n', f'{MPI_n}', 'SU2_CFD_AD', f'{config}'],
                    check=True, stdout=outfile, stderr=outfile)
    except CalledProcessError:
        LOGGER.warn("Could not solve Adjoint Equation, see output_adjoint.txt")
        raise CalledProcessError
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
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)
    shutil.copy('./volume_flow_rht_0.vtu', DATA_DIR + f'volume_flow_{index}.vtu')
    shutil.copy('./volume_solid_cht_1.vtu', DATA_DIR + f'./opt_iterations/volume_solid_{index}.vtu')
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


def armijo_rule(prev_deformation, sensitivities):
    return


def gradient_descent():

    LOGGER.info("\n ========= starting gradient descent ========= \n")
    state_cfg = 'turbulent_rht_cht.cfg'
    flow_cfg = 'config_flow_rht.cfg'
    solid_cfg = 'config_solid_cht.cfg'

    adj_cfg = 'turbulent_rht_cht_adjoint.cfg'
    state_sol_file = 'turbulent_rht_cht.csv'
    orig_flow_mesh = 'mesh_flow_ffd.su2'
    deformed_flow_mesh = 'mesh_flow_ffd_deform.su2'

    orig_ffd_box = 'config_ffd.cfg'
    deformed_ffd_box = 'deform_ffd.cfg'
    grad_file= 'of_grad.dat'
    scale = 5./100

    functional_data_file = DATA_DIR + 'functional_evolution.csv'
    if os.path.isfile(functional_data_file):
        os.remove(functional_data_file)
    with open(functional_data_file, 'w') as f:
        f.write("i, fvalue\n")

    opt_steps = 3

    change_mesh(flow_cfg, orig_flow_mesh)
    compile_ffd(orig_ffd_box)

    prev_deformation = [0 for i in range(24)]

    for i in range(opt_steps):
        LOGGER.info(f"========== {i} ==========")
        if i == 1:
            change_mesh(flow_cfg, deformed_flow_mesh)
        solve_state(state_cfg)
        J_i = extract_value(state_sol_file, 11)
        store_functional_value(i, J_i, functional_data_file)
        rename_state_files()
        solve_adj_state(adj_cfg)
        rename_adj_files()
        project_sensitivities(adj_cfg)

        store_important_files(index=i)

        ref_sensitivities = read_sensitivities(grad_file)
        sensitivities = [s - d for d,s in zip(prev_deformation, ref_sensitivities)]
        norm = math.sqrt(sum([s**2 for s in sensitivities]))
        LOGGER.info(f"L2 norm sensitivities: {norm}")

        deformation = [d + scale/norm*s for d,s in zip(prev_deformation, sensitivities)]
        get_ffd_deformation(deformation, deformed_ffd_box)
        compile_ffd(deformed_ffd_box)

        prev_deformation = deformation

    solve_state(state_cfg)
    J_i = extract_value(state_sol_file, 11)
    store_functional_value(opt_steps, J_final, functional_data_file)


if __name__ == '__main__':
    central_difference_verification()
    # gradient_descent()

