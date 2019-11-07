import pints
import myokit
import numpy as np

def set_myokit_simulation(model_filename):
    model, prot, script = myokit.load(model_filename)
    return myokit.Simulation(model, prot)


def simulate_data(fitted_params, params_values, exp_param, s, read_out: str, data, pre_run = 0):

    # This function is meant for comparison of simulation conditions to data, or to be called by the PINTS optimisation tool

    # Allocate memory for the output
    output = []

    # Run the model solving for all experiment conditions
    for k in range(0, len(set(data.exp_conds))):
        s.reset()
        # reset timer
        s.set_time(0)

        # Set parameters for simulation
        for i in range(0, len(fitted_params)):
            s.set_constant(fitted_params[i], params_values[i])

        # set the right experimental conditions
        s.set_constant(exp_param, list(set(data.exp_conds))[k])

        # Eventually run a pre-run to reach steady-state
        s.pre(pre_run)

        # Run the simulation with starting parameters
        a = s.run(data.times[k][-1]+0.01, log_times=data.times[k])
        # Convert output in concentration
        output.append( list(a[read_out]))
    return output


def quick_simulate(s, time_max, read_out: str,  varying_param_name = None, varying_param_values = [], fixed_params_names = [], fixed_params_values = [], pre_run = 0, time_samples = []):

    '''This function is for quick simulation of user selected conditions
    '''

    if len(fixed_params_names) != len(fixed_params_values):
        raise ValueError('The parameters clamped for the simulation must have the same length for names and values')

    if len(time_samples) == 0:
        time_samples = np.linspace(0,time_max, 100)

    # Allocate memory for the output
    output = []

    # Run the model solving for all experiment conditions
    # In case the user wants some parameter to vary between simulations
    if len(varying_param_values) > 0:
        for k in range(0, len(varying_param_values)):
            s.reset()

            # Set parameters for simulation
            if len(fixed_params_names) > 0:
                for i in range(0, len(fixed_params_names)):
                    s.set_constant(fixed_params_names[i], fixed_params_values[i])

            # set the right experimental conditions
            s.set_constant(varying_param_name, varying_param_values[k])

            # reset timer
            s.set_time(0)

            # Eventually run a pre-run to reach steady-state
            s.pre(pre_run)

            # Run the simulation with starting parameters
            a = s.run(time_max, log_times=time_samples)
            # Convert output in concentration
            output.append( list(a[read_out]))

    # In case there is no looping over a v varying parameter
    else:
        s.reset()

        # Set parameters for simulation
        if len(fixed_params_names) > 0:
            for i in range(0, len(fixed_params_names)):
                s.set_constant(fixed_params_names[i], fixed_params_values[i])

        # reset timer
        s.set_time(0)

        # Eventually run a pre-run to reach steady-state
        s.pre(pre_run)

        # Run the simulation with starting parameters
        a = s.run(time_max, log_times=time_samples)
        # Convert output in concentration
        output.append(list(a[read_out]))

    return output
