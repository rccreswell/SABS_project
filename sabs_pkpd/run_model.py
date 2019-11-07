import pints
import myokit
import numpy as np

def set_myokit_simulation(model_filename):
    model, prot, script = myokit.load(model_filename)
    return myokit.Simulation(model, prot)


def simulate_data(fitted_params, params_values, read_out: str, data):

    # Set parameters for simulation
    for i in range(0, len(fitted_params)):
        s.set_constant( fitted_params[i], params_values[i])

    # Allocate memory for the output
    output = []

    # Run the model solving for all experiment conditions
    for k in range(0, len(set(data.exp_nums))):
        s.reset()
        # reset timer
        s.set_time(0)

        # set the right experimental conditions
        s.set_constant(exp_param, list(set(data.exp_conds))[k])

        # Run the simulation with starting parameters
        a = s.run(data.times[k][-1]+0.01, log_times=data.times[k])
        # Convert output in concentration
        output.append( list(a[read_out]))
    return output

s = set_myokit_simulation('C:/Users/yanral/Documents/Software Development/tests/test resources/pints_problem_def_test.mmt')

data = load_data.load_data_file('C:/Users/yanral/Documents/Software Development/tests/test resources/load_data_test.csv')

fitted_params = ['constants.unknown_cst', 'constants.unknown_cst2']
params_values = [0.1,0.1]
read_out = 'comp1.y'
exp_param = 'constants.T'
exp_cond = [20, 37]

out = simulate_data(fitted_params, params_values,read_out, data)
diff = np.array(out) - np.array([[0.0, 0.01975, 0.09404, 0.17719, 0.42628, 0.58513, 0.79126, 0.99661], [0.0, 0.019504, 0.08836, 0.15683, 0.30589, 0.35623, 0.37456, 0.37037]])
print(np.linalg.norm(diff))