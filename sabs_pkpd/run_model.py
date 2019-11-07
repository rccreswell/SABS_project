import pints
import myokit
import numpy as np
import load_data

def set_myokit_simulation(model_filename):
    model, prot, script = myokit.load(model_filename)
    return myokit.Simulation(model, prot)


def simulate(self, fitted_params, params_values, data):
    s.reset()

    # Set parameters for simulation
    for i in range(len(fitteds_param)):
        eval('s.set_constant(' + selected_params[i] + ', ' + str(params_values[i]) + ')')

    # Run the model solving for all experiment conditions
    for i in range(len(set(data.exp_nums))):
        # reset timer
        s.set_time(0)

        # set the right experimental conditions
        eval('s.set_constant(' + exp_param + ', ' + str(list(set(data.exp_conds))[i]) + ')')
        s.set_constant('', dose[i])

        # Run the simulation with starting parameters
        a = s.run(data_times[i][-1] + 0.1, log_times=data_times[i][0:5])
        out = list(a['plasma.y_c'])
        output = output + out
        # Convert output in concentration
    output = np.multiply(output, 1 / parameters[1])
    return output


s = set_myokit_simulation('C:/Users/yanral/Documents/Software Development/tests/test resources/pints_problem_def_test.mmt')

fitted_params = ['constants.unknown_cst']
data = load_data.load_data_file('C:/Users/yanral/Documents/Software Development/tests/test resources/load_data_test.csv')
print(data.times)
exp_param = 'constants.T'
exp_cond = [20, 37]