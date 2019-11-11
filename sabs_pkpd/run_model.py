import pints
import myokit
import numpy as np
import matplotlib.pyplot as plt


def set_myokit_simulation(model_filename):
    model, prot, script = myokit.load(model_filename)
    return myokit.Simulation(model, prot)


def simulate_data(fitted_params_values, s, data_exp, pre_run = 0):

    # This function is meant for comparison of simulation conditions to data, or to be called by the PINTS optimisation tool

    # Allocate memory for the output
    output = []

    # Verify that the parameters for fitting and their values have the same length
    if len(fitted_params_values) != len(data_exp.fitting_instructions.fitted_params_annot):
        raise ValueError('Fitted parameters annotations and values should have the same length')

    # Run the model solving for all experiment conditions
    for k in range(0, len(set(data_exp.exp_conds))):
        s.reset()
        # reset timer
        s.set_time(0)

        # Set parameters for simulation
        for i in range(0, len(data_exp.fitting_instructions.fitted_params_annot)):
            s.set_constant(data_exp.fitting_instructions.fitted_params_annot[i], fitted_params_values[i])

        # set the right experimental conditions
        s.set_constant(data_exp.fitting_instructions.exp_cond_param_annot, list(set(data_exp.exp_conds))[k])

        # Eventually run a pre-run to reach steady-state
        s.pre(pre_run)

        # Run the simulation with starting parameters
        a = s.run(data_exp.times[k][-1]*1.001, log_times=data_exp.times[k])
        # Convert output in concentration
        output.append( list(a[data_exp.fitting_instructions.sim_output_param_annot]))
    return output


def quick_simulate(s, time_max, read_out: str,  exp_cond_param_annot = None, exp_cond_param_values = [], fixed_params_annot = [], fixed_params_values = [], pre_run = 0, time_samples = []):

    '''This function is for quick simulation of user selected conditions
    '''

    if time_samples != []:
        if time_samples[-1] > time_max :
            raise ValueError('The time samples have to be within the range (0 , time_max)')

    if len(fixed_params_annot) != len(fixed_params_values):
        raise ValueError('The parameters clamped for the simulation must have the same length for names and values')

    if len(time_samples) == 0:
        time_samples = np.linspace(0,time_max, 100)

    # Allocate memory for the output
    output = []

    # Run the model solving for all experiment conditions
    # In case the user wants some parameter to vary between simulations
    if type(exp_cond_param_values) != float and type(exp_cond_param_values) != int:
        if len(exp_cond_param_values) > 0:
            for k in range(0, len(exp_cond_param_values)):

                s.reset()

                # Set parameters for simulation
                if len(fixed_params_annot) > 0:
                    for i in range(0, len(fixed_params_annot)):
                        s.set_constant(fixed_params_annot[i], fixed_params_values[i])

                # set the right experimental conditions
                s.set_constant(exp_cond_param_annot, exp_cond_param_values[k])

                # reset time
                s.set_time(0)

                # Eventually run a pre-run to reach steady-state
                s.pre(pre_run)

                # Run the simulation with starting parameters
                a = s.run(time_max*1.001, log_times=time_samples)
                # Convert output in concentration
                output.append( list(a[read_out]))
        else:
                s.reset()

                # Set parameters for simulation
                if len(list(fixed_params_annot)) > 0:
                    for i in range(0, len(fixed_params_annot)):
                        s.set_constant(fixed_params_annot[i], fixed_params_values[i])

                # reset timer
                s.set_time(0)

                # Eventually run a pre-run to reach steady-state
                s.pre(pre_run)

                # Run the simulation with starting parameters
                a = s.run(time_max * 1.001, log_times=time_samples)
                # Convert output in concentration
                output.append(list(a[read_out]))

    # In case there is no looping over a v varying parameter
    else:
        s.reset()

        # Set parameters for simulation
        if len(list(fixed_params_annot)) > 0:
            for i in range(0, len(fixed_params_annot)):
                s.set_constant(fixed_params_annot[i], fixed_params_values[i])

        # reset timer
        s.set_time(0)

        # Eventually run a pre-run to reach steady-state
        s.pre(pre_run)

        # Run the simulation with starting parameters
        a = s.run(time_max * 1.001, log_times=time_samples)
        # Convert output in concentration
        output.append(list(a[read_out]))

    return output


def plot_model_vs_data(plotting_parameters_annot, plotting_parameters_values, data_exp, s):

    read_out = data_exp.fitting_instructions.sim_output_param_annot
    exp_cond_param_annot = data_exp.fitting_instructions.exp_cond_param_annot
    fixed_params_annot = list(plotting_parameters_annot)
    fixed_params_annot.append(exp_cond_param_annot)

    number_of_plots = len(data_exp.exp_conds)
    number_of_rows = number_of_plots//2 + number_of_plots % (number_of_plots//2)
    fig1 = plt.figure()

    for i in range(0, number_of_plots):
        plt.subplot(number_of_rows, 2, i+1)

        time_max = data_exp.times[i][-1]
        time_samples = data_exp.times[i]

        fixed_params_values = list(plotting_parameters_values)
        exp_cond_param_values = float(data_exp.exp_conds[i])
        fixed_params_values.append(exp_cond_param_values)

        sim_data = quick_simulate(s, time_max, read_out,  exp_cond_param_annot, exp_cond_param_values, fixed_params_annot, fixed_params_values, time_samples = time_samples)

        plt.plot(time_samples, sim_data[0], label='Simulated values')
        plt.plot(time_samples, data_exp.values[i], label='Experimental data')
        plt.title('Experimental conditions : ' + exp_cond_param_annot + ' = ' + str(exp_cond_param_values))
        plt.xlabel('Time')
        plt.ylabel(data_exp.fitting_instructions.sim_output_param_annot)
        plt.legend()

    return 0