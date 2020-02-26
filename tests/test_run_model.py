import sabs_pkpd

import io
import pytest
import numpy as np

def test_simulate_data():
    sabs_pkpd.constants.s = sabs_pkpd.load_model.load_simulation_from_mmt('./tests/test resources/pints_problem_def_test.mmt')

    # Save the default state as defined in the mmt file
    sabs_pkpd.constants.default_state = sabs_pkpd.constants.s.default_state()

    sabs_pkpd.constants.data_exp = sabs_pkpd.load_data.load_data_file('./tests/test resources/load_data_test.csv')

    fitted_params_annot = ['constants.unknown_cst', 'constants.unknown_cst2', 'comp1.y']
    fitted_params_values = [0.1, 0.1, 0]
    sim_output_param_annot = 'comp1.y'
    exp_cond_param_annot = 'constants.T'

    sabs_pkpd.constants.data_exp.Add_fitting_instructions(fitted_params_annot, exp_cond_param_annot,
                                                          sim_output_param_annot)

    out = sabs_pkpd.run_model.simulate_data(fitted_params_values, sabs_pkpd.constants.s, sabs_pkpd.constants.data_exp)
    diff = np.array(out) - np.array([[0.0, 0.01975, 0.09404, 0.17718, 0.42628, 0.58526, 0.79111, 0.99642],
                                     [0.0, 0.019504, 0.08836, 0.15682, 0.30589, 0.35632, 0.37457, 0.37037]])
    assert np.linalg.norm(diff) < 0.0003


def test_quick_simulate():
    s = sabs_pkpd.load_model.load_simulation_from_mmt('./tests/test resources/pints_problem_def_test.mmt')

    # Save the default state as defined in the mmt file
    sabs_pkpd.constants.default_state = sabs_pkpd.constants.s.default_state()

    time_max = 6
    changed_params_names = ['constants.unknown_cst', 'constants.unknown_cst2']
    changed_params_values = [0.1, 0.1]
    time_samples = [0, 0.01, 0.05, 0.1, 0.3, 0.5, 1, 5]

    test1 = sabs_pkpd.run_model.quick_simulate(s, time_max, 'comp1.y')
    expected_value = np.array([0.0, 0.00691, 0.00643, 0.00585, 0.00559, 0.00550, 0.00547, 0.00546, 0.00546, 0.00545, 0.00545, 0.005454])
    diff = np.array(test1[0])[0:12] - expected_value
    assert np.linalg.norm(diff) < 0.003

    test2 = sabs_pkpd.run_model.quick_simulate(s, time_max, 'comp1.y', time_samples=time_samples, fixed_params_annot=changed_params_names, fixed_params_values=changed_params_values)

    diff = np.array(test2[0]) - np.array([0.0, 0.019504, 0.08836, 0.15683, 0.30589, 0.35623, 0.37456, 0.37037])
    assert np.linalg.norm(diff) < 0.0001
