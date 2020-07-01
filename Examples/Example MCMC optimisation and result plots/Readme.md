# About data.csv files

The data file should be loaded as a .csv file, with <strong>coma as separator</strong>.

The file should be organised following the template:

| Time | Values | Experiment number | Experimental condition |
| ---- | ------ | ----------------- | ---------------------- | 
|0.00  | -85.47 |123456|310|
|... |...|...|...|...|
|1000  | -85.29 |123456|310

Note : the rows do not need to be sorted. While loading the data table, the table will be sorted with increasing experimental conditions (in a first time), and with increasing times (in a second time), thus resulting in a data structure similar to the one above.

For running the MCMC routine, you can also use synthetic data, that you can generate by using sabs_pkpd.run_model.quick_simulate().


# Architecture of the data loaded from a CSV

The data is loaded as a data_exp structure. <strong>data_exp</strong> is a structure with entries:
  - <strong>times</strong> is a list of all loaded experiments, with all time points for each ;
  - <strong>values</strong> is a list with the same shape, containing the data measured at those time points ;
  - <strong>exp_nums</strong> is a list of the labels of the experiments, as floats.
  - <strong>exp_conds</strong> is a list of the values taken by the experimental condition for each experiment. It is of length the number of experiments.
  - <strong>.fitting_instructions</strong> after call of the function <strong>Add_fitting_instructions</strong>. This subclass has entries:
    - <strong>fitted_params_annot</strong> is a list of strings for the parameters fitted for the model. It is presented as ['compartment.parameter_fitted1', 'compartment.parameter_fitted2', ...]. It should match the annotations of the .mmt model used for simulations/fitting.
    - <strong>exp_cond_param_annot</strong> is a string. It should match the annotation of the .mmt model for the varying experimental condition used to generate the data.
    - <strong>sim_output_param_annot</strong> is a string. It should match the annotation of the .mmt model output.

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

![synth_data](https://github.com/rcw5890/SABS_project/tree/master/Examples/Example%20MCMC%20optimisation%20and%20result%20plots)
## Set up the MCMC routine

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

```


The output is presented in the figure below:
![Sim_vs_exp](./Example_plot_exp_vs_sim.png)
