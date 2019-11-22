import sabs_pkpd
import numpy as np
import matplotlib.pyplot as plt


# Prepare the data for example problem
filename = 'C:/Users/yanral/Documents/Software Development/Examples/Example loading CellML model/tentusscher_2006_pints_and_Chons_hERG.mmt'
sabs_pkpd.constants.s = sabs_pkpd.load_model.load_simulation_from_mmt(filename)

fitting_param_annot = ['ikr.scale_kr', 'ical.scale_cal']
value_generate_rescale = [1, 1]

sabs_pkpd.constants.s.reset()

for i in range(len(fitting_param_annot)):
    sabs_pkpd.constants.s.set_constant(fitting_param_annot[i], value_generate_rescale[i])

sabs_pkpd.constants.n = 2

time_max = 1000
times = [np.linspace(0, time_max, 1001)]
exp_nums_list = [1]
exp_conds_list = [310]
exp_cond_annot = 'phys.T'
readout = 'membrane.V'

val = sabs_pkpd.constants.s.run(time_max*1.001, log_times = times[0])
values = [val[readout]]

sabs_pkpd.constants.data_exp = sabs_pkpd.load_data.Data_exp(times, values, exp_nums_list, exp_conds_list)
sabs_pkpd.constants.data_exp.Add_fitting_instructions(fitting_param_annot,exp_cond_annot, readout)


# Select a starting point close to the settings used for generating the synthetic data
Noise_sigma=0.5 # variance of N distributed noise to add to the trace

set_to_test = np.zeros(len(value_generate_rescale))
for i in range(len(value_generate_rescale)):
  set_to_test[i] = np.random.uniform(low=0.9, high=1.11, size=None)*value_generate_rescale[i]
starting_point = [np.array(list(set_to_test) + [Noise_sigma])]


# Run the MCMC routine calling the function in the sabs_pkpd package
chains = sabs_pkpd.pints_problem_def.MCMC_inference_model_params(starting_point,max_iter=3000)
chain_index = 0 # Only one chain is run so this is the chain we select for analysis


# Plot a 2D map of distribution of parameter i against parameter j
fig, ax = plt.subplots(1,1)
sabs_pkpd.pints_problem_def.plot_kde_2d(0, 1, chains, ax, chain_index=chain_index)


# Plot the correlation map of parameters 1-to-1
sabs_pkpd.pints_problem_def.plot_distribution_map(mcmc_chains=chains, RealValue= value_generate_rescale,
                                                 chain_index=chain_index, fig_size=(25,25))


# Plot comparison between synthetic data, the model output for the starting point, and the endpoint of the chain

plt.figure()
for i in range(len(sabs_pkpd.constants.data_exp.exp_conds)):
    sabs_pkpd.constants.s.set_constant(sabs_pkpd.constants.data_exp.fitting_instructions.exp_cond_param_annot, sabs_pkpd.constants.data_exp.exp_conds[i])
    time_samples = sabs_pkpd.constants.data_exp.times[i]

    starting_point = chains[chain_index][0, :]
    sabs_pkpd.constants.s.reset()
    for j in range(len(sabs_pkpd.constants.data_exp.fitting_instructions.fitted_params_annot)):
        sabs_pkpd.constants.s.set_constant(sabs_pkpd.constants.data_exp.fitting_instructions.fitted_params_annot[j], starting_point[j])
    sim = sabs_pkpd.constants.s.run(time_samples[-1]*1.001, log_times = time_samples)
    starting_point_sim = sim[readout]

    endpoint = chains[chain_index][-1, :]
    sabs_pkpd.constants.s.reset()
    for j in range(len(sabs_pkpd.constants.data_exp.fitting_instructions.fitted_params_annot)):
        sabs_pkpd.constants.s.set_constant(sabs_pkpd.constants.data_exp.fitting_instructions.fitted_params_annot[j], endpoint[j])
    sim = sabs_pkpd.constants.s.run(time_samples[-1]*1.001, log_times = time_samples)
    endpoint_sim = sim[readout]

    plt.plot(time_samples, values[i], label='values to fit')
    plt.plot(time_samples, starting_point_sim, label='starting point')
    plt.plot(time_samples, endpoint_sim, label='EndChain')
    plt.legend()
