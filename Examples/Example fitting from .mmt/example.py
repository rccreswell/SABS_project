import sabs_pkpd


# Fix the variables which have to be global. For now, the variables names have
# to be n (number of fitted params) and s (the .mmt model loaded for
# simulations).
sabs_pkpd.constants.n = 2
sabs_pkpd.constants.s = sabs_pkpd.run_model.set_myokit_simulation(
    './tests/test resources/pints_problem_def_test.mmt')

# Define the data to fit and the fitting instructions
fit_param_annot = ['constants.unknown_cst', 'constants.unknown_cst2']
exp_cond_annot = 'constants.T'
model_output_annot = 'comp1.y'
sabs_pkpd.constants.data_exp = sabs_pkpd.load_data.load_data_file(
    './tests/test resources/load_data_test.csv')
sabs_pkpd.constants.data_exp.Add_fitting_instructions(
    fit_param_annot, exp_cond_annot, model_output_annot)

# Define all the conditions for parameters inference
initial_point = [0.5, 0.5]
boundaries_low = [0, 0]
boundaries_high = [1, 1]


inferred_params = sabs_pkpd.pints_problem_def.infer_params(
    initial_point,
    sabs_pkpd.constants.data_exp,
    boundaries_low,
    boundaries_high)

# Plot of the results against the data
sabs_pkpd.run_model.plot_model_vs_data(
    ['constants.unknown_cst', 'constants.unknown_cst2'],
    inferred_params,
    sabs_pkpd.constants.data_exp,
    sabs_pkpd.constants.s)
