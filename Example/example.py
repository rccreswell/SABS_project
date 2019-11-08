import sabs_pkpd
import io
import pytest
import numpy as np


# Fix the variables which have to be global
n = 2
s = sabs_pkpd.run_model.set_myokit_simulation('./tests/test resources/pints_problem_def_test.mmt')

# Define all the conditions for parameters inference
initial_point = [0.5, 0.5]
data_exp = sabs_pkpd.load_data.load_data_file('./tests/test resources/load_data_test.csv')
data_exp.Add_fitting_instructions(['constants.unknown_cst', 'constants.unknown_cst2'], 'constants.T', 'comp1.y')
boundaries_low = [0, 0]
boundaries_high = [1, 1]

infered_params = sabs_pkpd.pints_problem_def.infer_params(initial_point, data_exp, boundaries_low, boundaries_high)