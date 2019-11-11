import sabs_pkpd


n = 2

if type(n) != int:
    raise ValueError('sabs_pkpd.constants.n should be an integer')

s = sabs_pkpd.run_model.set_myokit_simulation('C:/Users/yanral/Documents/Software Development/tests/test resources/pints_problem_def_test.mmt')

data_exp = sabs_pkpd.load_data.Data_exp([], [], [], [])