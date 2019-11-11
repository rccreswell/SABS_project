import sabs_pkpd

n = 2

s = sabs_pkpd.run_model.set_myokit_simulation('./tests/test resources/pints_problem_def_test.mmt')

data_exp = sabs_pkpd.load_data.Data_exp([], [], [], [])