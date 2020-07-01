# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:55:53 2020

@author: yann-stanislas.barral@roche.com
"""

import sabs_pkpd
import numpy as np
import matplotlib.pyplot as plt

# Prepare the data for example problem

# Load the model
filename = './model.mmt'
sabs_pkpd.constants.s = sabs_pkpd.load_model.load_simulation_from_mmt(filename)
sabs_pkpd.constants.default_state = sabs_pkpd.constants.s.state()
sabs_pkpd.constants.s.set_tolerance(abs_tol = 1e-10, rel_tol = 1e-10)

# Define the model parameters that will be fitted
fitting_param_annot = ['constants.unknown_cst', 'constants.unknown_cst2']

# Set the parameters values to generate the synthetic data
# This will be called the "true" values of these parameters
true_values = [5, 3]
for i, annot in enumerate(fitting_param_annot):
    sabs_pkpd.constants.s.set_constant(annot, true_values[i])
    
# Enter to constants.n the amount of fitted parameters
sabs_pkpd.constants.n = 2

# Set up the simulation to generate the synthetic data
time_max = 0.1
times = [np.linspace(0, time_max, 100)]
exp_nums_list = [1]
exp_conds_list = [37]
exp_cond_annot = 'constants.T'
readout = 'comp1.x'

# Run the simulation to generate synthetic data
val = sabs_pkpd.constants.s.run(time_max*1.001, log_times = times[0])

# Retrieve the output and add noise on top of it
values = [val[readout]]
Noise_sigma=0.006 # variance of N distributed noise to add to the data
values = values + Noise_sigma*np.random.randn(len(values[0]))

# Load the synthetic data to constants.data_exp using the class Data_exp
sabs_pkpd.constants.data_exp = sabs_pkpd.load_data.Data_exp(times, values, exp_nums_list, exp_conds_list)

# Set the fitting information
fitting_param_annot = ['constants.unknown_cst', 'constants.unknown_cst2']
exp_cond_annot = 'constants.T'
readout = 'comp1.x'
# Save the fitting information to the data_exp object
sabs_pkpd.constants.data_exp.Add_fitting_instructions(fitting_param_annot,exp_cond_annot, readout)


# Visualisation of the synthetic data
plt.figure(figsize = (10, 10))
plt.plot(sabs_pkpd.constants.data_exp.times[0], sabs_pkpd.constants.data_exp.values[0])
plt.xlabel('Time', Fontsize = 22)
plt.ylabel('comp1.x', Fontsize = 22)
plt.title('Synthetic data for the illustration example of running a MCMC' +
           ' routine using the SABS package')
plt.show()

# Set a starting point for the MCMC routine
starting_point = [np.array([4, 2, 0.005]), np.array([5, 3, 0.005]), np.array([6, 4, 0.004])]

# Launch the MCMC routine
chains = sabs_pkpd.pints_problem_def.MCMC_routine(starting_point, max_iter=5000)

# Plot the evolution of the chains during the MCMC sampling
fig, axes = sabs_pkpd.pints_problem_def.plot_MCMC_convergence(mcmc_chains=chains, expected_values = true_values+[0.006], bound_max = [12, 8, 0.01], bound_min = [2, 1.5, 0.002])

# Plot the distributions of the values taken by the chains
fig2, axes2 = sabs_pkpd.pints_problem_def.plot_distribution_parameters(chains, [2, 1.5, 0.002], [12, 8, 0.01], chain_index = 1)

# Compare the data with the output of the model, using the starting point's 
# parameter values and the chain's median parameter values
plt.figure()

# Compare the data with the output of the model using the starting point's
# parameters values, and the chain's median parameter values
model = sabs_pkpd.pints_problem_def.MyModel()
plt.plot(sabs_pkpd.constants.data_exp.times[0], 
         sabs_pkpd.constants.data_exp.values[0], 
         label='values to fit')

plt.plot(sabs_pkpd.constants.data_exp.times[0], 
         model.simulate(starting_point[0][:-1], 
                        sabs_pkpd.constants.data_exp.times[0]),
         label='Starting point')
         
plt.plot(sabs_pkpd.constants.data_exp.times[0], 
         model.simulate(np.median(chains[0], axis = 0)[:-1],
                        sabs_pkpd.constants.data_exp.times[0]),
         label='Chain 0 median')
plt.legend()
plt.show()
