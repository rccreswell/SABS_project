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
