import os
import csv
import subprocess
import re
import shutil
import logging
import numpy as np


logging.basicConfig(level=logging.DEBUG)

LOGGER = logging.getLogger(__name__)

def compile_ffd(config):
    LOGGER.info(f"Compile FFD box using {config}")
    subprocess.run(['SU2_DEF', f'{config}'], check=True, stdout=subprocess.DEVNULL)


def solve_state(config):
    LOGGER.info("Solve State Equation")
    subprocess.run(['mpirun', '-n', '2', 'SU2_CFD', f'{config}'], check=True, stdout=subprocess.DEVNULL)


def solve_adj_state(config):
    LOGGER.info("Solve Adjoint Equation")
    subprocess.run(['mpirun', '-n', '2', 'SU2_CFD_AD', f'{config}'], check=True, stdout=subprocess.DEVNULL)


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
    state_cfg = 'turbulent_rht_cht.cfg'
    flow_cfg = 'config_flow_rht.cfg'
    solid_cfg = 'config_solid_cht.cfg'

    adj_cfg = 'turbulent_rht_cht_adjoint.cfg'
    state_sol_file = 'turbulent_rht_cht.csv'
    orig_flow_mesh = 'mesh_flow_ffd.su2'
    deformed_flow_mesh = 'mesh_flow_ffd_deform.su2'

    orig_ffd_box = 'config_ffd.cfg'
    deformed_ffd_box = 'deform_ffd.cfg'

    results_file = 'results_test.csv'

    h = 1e-6
    if not os.path.isfile('of_grad.dat'):
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

    for i, adj_grad in enumerate(adj_sensitivities[:1]):
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
    if not os.path.isdir('opt_iterations'):
        os.makedirs('opt_iterations')
    shutil.copy('./volume_flow_rht_0.vtu', f'./opt_iterations/volume_flow_{index}.vtu')
    shutil.copy('./volume_solid_cht_1.vtu', f'./opt_iterations/volume_solid_{index}.vtu')
    shutil.copy('of_grad.dat', f'opt_iterations/of_grad_{index}.csv')
    shutil.copy('surface_sens_0.vtu', f'opt_iterations/surface_sens_line_{index}.vtu')
    shutil.copy('surface_sens_1.vtu', f'opt_iterations/surface_sens_nothing_{index}.vtu')
    shutil.copy('volume_sens_0.vtu', f'opt_iterations/surface_sens_flow_{index}.vtu')
    shutil.copy('volume_sens_1.vtu', f'opt_iterations/surface_sens_solid_{index}.vtu')
    shutil.copy('ffd_boxes_0.vtk', f'opt_iterations/ffd_boxes_upper_{index}.vtk')
    shutil.copy('ffd_boxes_1.vtk', f'opt_iterations/ffd_boxes_lower_{index}.vtk')


def store_functional_value(index, value, functional_file):
    with open(functional_file, 'a') as f:
        f.write(f"{index}, {value}\n")

def gradient_descent():
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

    functional_data_file = 'functional_evolution.csv'
    if os.path.isfile(functional_data_file):
        os.remove(functional_data_file)
    with open(functional_data_file, 'w') as f:
        f.write("i, fvalue\n")

    opt_steps = 5

    change_mesh(flow_cfg, orig_flow_mesh)
    compile_ffd(orig_ffd_box)

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

        sensitivities = read_sensitivities(grad_file)
        sensitivities = np.array(sensitivities)
        deformation = scale/np.linalg.norm(sensitivities)*sensitivities
        get_ffd_deformation(deformation, deformed_ffd_box)
        compile_ffd(deformed_ffd_box)

        # deformation[0] = 0
        # deformation[12] = 0
        # deformation = [0.0, 0.05, 0.1, 0.14, 0.16, 0.17, 0.175, 0.17, 0.16,
        #         0.14, 0.1, 0.05, 0.0, -0.05, -0.1, -0.14, -0.16, -0.17, -0.175,
        #         -0.17, -0.16, -0.14, -0.1, -0.05]


if __name__ == '__main__':
    # central_difference_verification()
    gradient_descent()

