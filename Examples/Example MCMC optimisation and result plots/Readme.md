# Use of the simulation and inference tools included in this package.

Check the example.py file for the example code. If you want to run it, please change the directories to the directories of your choice. In the present example, data will be synthetic data generated using the present package.

##  Importing the necessary libraries:

```python
import sabs_pkpd
import numpy as np
import matplotlib.pyplot as plt
```

## Prepare the data for example problem

The model is loaded. Note that it is necessary to load it into sabs_pkpd.constants.s. Indeed, the object ```sabs_pkpd.pints_problem_def.MyModel``` needs to have a fixed variable to be used properly during fitting or MCMC sampling.

```python
filename = './model.mmt'
sabs_pkpd.constants.s = sabs_pkpd.load_model.load_simulation_from_mmt(filename)
```

The fitting instructions are then set up, and the synthetic data is generated.

```python
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
Noise_sigma=0.006 # variance of normally distributed noise to add to the data
values = values + Noise_sigma*np.random.randn(len(values[0]))
```

The data and fitting instructions are then uploaded to a Data_exp object in the ```sabs_pkpd.constants```.

```python
# # Load the synthetic data to constants.data_exp using the class Data_exp
sabs_pkpd.constants.data_exp = sabs_pkpd.load_data.Data_exp(times, values, exp_nums_list, exp_conds_list)

# Set the fitting information
fitting_param_annot = ['constants.unknown_cst', 'constants.unknown_cst2']
exp_cond_annot = 'constants.T'
readout = 'comp1.x'
# Save the fitting information to the data_exp object
sabs_pkpd.constants.data_exp.Add_fitting_instructions(fitting_param_annot,exp_cond_annot, readout)
```
The data generated is plotted on the Figure below:

![synth_data](https://raw.githubusercontent.com/rcw5890/SABS_project/master/Examples/Example%20MCMC%20optimisation%20and%20result%20plots/synthetic%20data.png?token=ANSJY56CP2AJKNQ5PHWKJC27AWEAE)



## Set up the MCMC routine and launch it

The MCMC routine takes as minimal input a starting point for the chains (use the command ```help(sabs_pkpd.pints_problem_def.MCMC_routine)``` in your Python console for more information about the inputs). 

Note that the starting point must have a very particular structure to be understood by the function to launch the MCMC routine. It has to be a list of arrays. The number of arrays defines the number of chains used for the MCMC routine. The length of the arrays must match with the amount of parameters, eventually + 1 for noise, depending on the log-likelihood function chosen. The default log-likelihood used is ```pints.GaussionLogLikelihood```, which samples the noise, hence you need to provide a starting value for noise. If for example you are using the ```pints.GaussianKnownSigmaLogLikelihood```, the noise sigma is fixed so you just have to use arrays of length the amount of parameters.

```python
# Set a starting point for the MCMC routine
starting_point = [np.array([4, 2, 0.005]), np.array([5, 3, 0.005]), np.array([6, 4, 0.004])]

# Launch the MCMC routine
chains = sabs_pkpd.pints_problem_def.MCMC_routine(starting_point, max_iter=5000)
```

## Plot the results of your freshly run MCMC routine

If you want to plot the distributions of parameters one against another, you might want to use pints.plot.pairwise (have a quick look at https://github.com/pints-team/pints/tree/master/examples/plotting). On top of pints possibilities of plotting the results, we propose you to plot the MCMC chains convergence, as well as the distributions of the values taken by your parameters throughout the MCMC routine.

```python
# Plot the evolution of the chains during the MCMC sampling
fig, axes = sabs_pkpd.pints_problem_def.plot_MCMC_convergence(mcmc_chains=chains, expected_values = true_values+[0.006], bound_max = [12, 8, 0.01], bound_min = [2, 1.5, 0.002])
```

This returns plots like the one below for our example problem:

![evolution](https://raw.githubusercontent.com/rcw5890/SABS_project/master/Examples/Example%20MCMC%20optimisation%20and%20result%20plots/chains%20evolution.png?token=ANSJY5YMAJWACDTURUFAAEC7AWJF4)

```python
# Plot the distributions of the values taken by the chains
fig2, axes2 = sabs_pkpd.pints_problem_def.plot_distribution_parameters(chains, [2, 1.5, 0.002], [12, 8, 0.01], chain_index = 1)
```
This returns plots like the one below for our example problem:

![distribution](https://raw.githubusercontent.com/rcw5890/SABS_project/master/Examples/Example%20MCMC%20optimisation%20and%20result%20plots/parameters%20distribution.png?token=ANSJY5YWPAKLCIJE4EBFKFS7AWJL2)

Note that in this example, the parameter 1 (constants.unknown_cst) is not identifiable, as it can visibly take pretty much any value within [2 - 12]

```python
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
```
