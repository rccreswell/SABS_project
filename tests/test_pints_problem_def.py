import sabs_pkpd
import io
import pytest
import numpy as np

def test_infer_params():

    # Fix the variables which have to be global
    sabs_pkpd.constants.n = 2
    sabs_pkpd.constants.s = sabs_pkpd.load_model.load_simulation_from_mmt('./tests/test resources/pints_problem_def_test.mmt')

    # Define all the conditions for parameters inference
    initial_point = [0.5, 0.5]
    boundaries_low = [0, 0]
    boundaries_high = [1, 1]

    sabs_pkpd.constants.data_exp = sabs_pkpd.load_data.load_data_file('./tests/test resources/load_data_test.csv')
    sabs_pkpd.constants.data_exp.Add_fitting_instructions(['constants.unknown_cst', 'constants.unknown_cst2'], 'constants.T', 'comp1.y')

    # Fix the random seed so that the parameter inference always returns the same fitted parameters
    np.random.seed(19580)

    inferred_params = sabs_pkpd.pints_problem_def.infer_params(initial_point, sabs_pkpd.constants.data_exp, boundaries_low, boundaries_high)
    diff = inferred_params - np.array([0.1, 0.1])
    print(inferred_params)
    assert np.linalg.norm(diff) < 0.01
