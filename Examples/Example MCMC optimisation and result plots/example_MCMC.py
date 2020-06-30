import sabs_pkpd
import numpy as np
import matplotlib.pyplot as plt
"""
# Prepare the data for example problem
filename = './model.mmt'
sabs_pkpd.constants.s = sabs_pkpd.load_model.load_simulation_from_mmt(filename)

fitting_param_annot = ['ikr.scale_kr', 'ical.scale_cal']
value_generate_rescale = [1,1]
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

Noise_sigma=0.5 # variance of N distributed noise to add to the trace
values = values + Noise_sigma*np.random.randn(len(values[0]))
"""
sabs_pkpd.constants.data_exp = sabs_pkpd.load_data.load_data_file('C:/Users/yanral/Documents/Software Development/tests/test resources/mcmc_test_data.csv')
fitting_param_annot = ['ikr.scale_kr', 'ical.scale_cal']
exp_cond_annot = 'phys.T'
readout = 'membrane.V'
sabs_pkpd.constants.s = sabs_pkpd.load_model.load_simulation_from_mmt('C:/Users/yanral/Documents/Software Development/Examples/Example loading CellML model/tentusscher_2006_pints_and_Chons_hERG.mmt')
sabs_pkpd.constants.data_exp.Add_fitting_instructions(fitting_param_annot,exp_cond_annot, readout)



RealValue = [1, 1]
Noise_sigma=0.5 # variance of N distributed noise to add to the trace

set_to_test = np.zeros(len(RealValue))
for i in range(len(RealValue)):
    set_to_test[i] = np.random.uniform(low=0.7, high=1.5, size=None)*RealValue[i]
starting_point = [np.array(list(set_to_test) + [Noise_sigma])]

np.random.seed(168584)
chains = sabs_pkpd.pints_problem_def.MCMC_inference_model_params(starting_point,max_iter=3000)

fig, ax = plt.subplots(1, 1)
ax.set_xlabel('parameter 1')
ax.set_ylabel('parameter 2')
sabs_pkpd.pints_problem_def.plot_kde_2d(0, 1, chains, ax)
plt.show()

n_param = sabs_pkpd.constants.n
fig_size = (30, 30)

start_parameter = chains[0][0, :]
sabs_pkpd.pints_problem_def.plot_distribution_map(mcmc_chains=chains,RealValue=RealValue)
# In[]:
plt.show()
fig_size = (60, 60)
plt.figure()
model = sabs_pkpd.pints_problem_def.MyModel()
plt.plot(sabs_pkpd.constants.data_exp.times[0], sabs_pkpd.constants.data_exp.values[0], label='values to fit')
plt.plot(sabs_pkpd.constants.data_exp.times[0], model.simulate(set_to_test, sabs_pkpd.constants.data_exp.times[0]), label='Starting point')
plt.plot(sabs_pkpd.constants.data_exp.times[0], model.simulate(chains[0][-1, :-1], sabs_pkpd.constants.data_exp.times[0]), label='End point')
plt.legend()
