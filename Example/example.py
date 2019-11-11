import sabs_pkpd
import io
import pytest
import numpy as np


# Fix the variables which have to be global. For now, the variables names have to be n and s.
sabs_pkpd.constants.n = 2
sabs_pkpd.constants.s = sabs_pkpd.run_model.set_myokit_simulation('C:/Users/yanral/Documents/Software Development/tests/test resources/pints_problem_def_test.mmt')

# Define all the conditions for parameters inference
initial_point = [0.5, 0.5]
boundaries_low = [0, 0]
boundaries_high = [1, 1]

# Define the data to fit and the fitting instructions
sabs_pkpd.constants.data_exp = sabs_pkpd.load_data.load_data_file('C:/Users/yanral/Documents/Software Development/tests/test resources/load_data_test.csv')
sabs_pkpd.constants.data_exp.Add_fitting_instructions(['constants.unknown_cst', 'constants.unknown_cst2'], 'constants.T', 'comp1.y')

inferred_params = sabs_pkpd.pints_problem_def.infer_params(initial_point, sabs_pkpd.constants.data_exp, boundaries_low, boundaries_high)

# Plot of the results against the data
sabs_pkpd.run_model.plot_model_vs_data(['constants.unknown_cst', 'constants.unknown_cst2'], inferred_params, sabs_pkpd.constants.data_exp, sabs_pkpd.constants.s)