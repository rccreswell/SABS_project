import sabs_pkpd

import io
import pytest
import numpy as np

def test_simulate_data():
    s = sabs_pkpd.run_model.set_myokit_simulation('./tests/test resources/pints_problem_def_test.mmt')

    data = sabs_pkpd.load_data.load_data_file('./tests/test resources/load_data_test.csv')

    fitted_params = ['constants.unknown_cst', 'constants.unknown_cst2']
    params_values = [0.1, 0.1]
    read_out = 'comp1.y'
    exp_param = 'constants.T'
    exp_cond = [20, 37]

    out = sabs_pkpd.run_model.simulate_data(fitted_params, params_values, exp_param, s, read_out, data)
    diff = np.array(out) - np.array([[0.0, 0.01975, 0.09404, 0.17719, 0.42628, 0.58513, 0.79126, 0.99661],
                                     [0.0, 0.019504, 0.08836, 0.15683, 0.30589, 0.35623, 0.37456, 0.37037]])
    assert np.linalg.norm(diff) < 0.0001