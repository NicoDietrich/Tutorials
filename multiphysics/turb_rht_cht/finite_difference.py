import os
import csv
import subprocess
import re

def compile_ffd(config):
    subprocess.run(['SU2_DEF', f'{config}'])


def solve_state(config):
    subprocess.run(['mpirun', '-n', '2', 'SU2_CFD', f'{config}'])


def solve_adj_state(config):
    subprocess.run(['mpirun', '-n', '2', 'SU2_CFD_AD', f'{config}'])


def project_sensitivities(config):
    subprocess.run(['SU2_DOT_AD', f'{config}'])


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

if __name__ == '__main__':
    state_cfg = 'turbulent_rht_cht.cfg'
    flow_cfg = 'config_flow_rht.cfg'
    solid_cfg = 'config_solid_cht.cfg'

    adj_cfg = 'turbulent_rht_cht_adjoint.cfg'
    state_sol_file = 'turbulent_rht_cht.csv'
    orig_flow_mesh = 'mesh_flow_ffd.su2'
    deformed_flow_mesh = 'mesh_flow_ffd_deform.su2'

    results_file = 'results.csv'

    h = 1e-6
    adj_sensitivities = read_sensitivities('of_grad.dat')
    change_mesh(flow_cfg, deformed_flow_mesh)

    fieldnames = ['index', 'h', 'F_xph', 'F_xmh', 'central_dif', 'adj_grad']

    # with open(results_file, 'w', newline='') as f:
    #     writer = csv.DictWriter(f, fieldnames)
    #     writer.writeheader()

    for i, adj_grad in enumerate(adj_sensitivities[2:]):
        i += 2  # first two already done
        deformation = 24*[0]
        deformation[i] = h
        get_ffd_deformation(deformation, "deform_ffd.cfg")
        compile_ffd('deform_ffd.cfg')
        solve_state(state_cfg)
        F_xph = extract_value(state_sol_file, 11)

        deformation[i] = -h
        get_ffd_deformation(deformation, "deform_ffd.cfg")
        compile_ffd('deform_ffd.cfg')
        solve_state(state_cfg)
        F_xmh = extract_value(state_sol_file, 11)

        central_difference = (F_xph - F_xmh)/(2*h)

        data = {'index': i, 'h': h, 'F_xph': F_xph,
            'F_xmh': F_xmh, 'central_dif': central_difference,
            'adj_grad': adj_grad}

        with open(results_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames)
            writer.writerow(data)

