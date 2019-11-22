import sabs_pkpd
import numpy as np
import matplotlib.pyplot as plt

# Prepare the data for example problem
filename = 'C:/Users/yanral/Documents/Software Development/Examples/Example loading CellML model/tentusscher_2006_pints_and_Chons_hERG.mmt'
sabs_pkpd.constants.s = sabs_pkpd.load_model.load_simulation_from_mmt(filename)

fitting_param_annot = ['ikr.scale_kr', 'ical.scale_cal']
value_generate_rescale = [3, 2]
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



RealValue = [1, 1]
Noise_sigma=0.5 # variance of N distributed noise to add to the trace

set_to_test = np.zeros(len(RealValue))
for i in range(len(RealValue)):
  set_to_test[i] = np.random.uniform(low=0.9, high=1.11, size=None)*RealValue[i]
starting_point = [np.array(list(set_to_test) + [Noise_sigma])]

chains = sabs_pkpd.pints_problem_def.MCMC_inference_model_params(starting_point,max_iter=3000)

fig, ax = plt.subplots(1, 1)
ax.set_xlabel('parameter 1')
ax.set_ylabel('parameter 2')
sabs_pkpd.pints_problem_def.plot_kde_2d(chains[0][:, 0], chains[0][:, 1], ax)
plt.show()

n_param = sabs_pkpd.constants.n
fig_size = (30, 30)

start_parameter = chains[0][0, :]
fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)
for i in range(n_param):
    for j in range(n_param):

        # Create subplot
        if i == j:
            # Plot the diagonal
            sabs_pkpd.pints_problem_def.plot_kde_1d(chains[0][:, i], ax=axes[i, j])
            axes[i, j].axvline(RealValue[i], c='g')
            axes[i, j].axvline(start_parameter[i], c='b')
            axes[i, j].legend()
        elif i < j:
            # Upper triangle: No plot
            axes[i, j].axis('off')
        else:
            # Lower triangle: Pairwise plot
            sabs_pkpd.pints_problem_def.plot_kde_2d(chains[0][:, j], chains[0][:, i], ax=axes[i, j])
            axes[i, j].axhline(RealValue[i], c='g')
            axes[i, j].axvline(RealValue[j], c='g')

            axes[i, j].axhline(start_parameter[i], c='b')
            axes[i, j].axvline(start_parameter[j], c='b')

        # Adjust the tick labels
        if i < n_param - 1:
            # Only show x tick labels for the last row
            axes[i, j].set_xticklabels([])
        else:
            # Rotation the x tick labels to fit in the plot
            for tl in axes[i, j].get_xticklabels():
                tl.set_rotation(45)
        if j > 0:
            # Only show y tick labels for the first column
            axes[i, j].set_yticklabels([])

    # Add labels to the subplots at the edges
    axes[i, 0].set_ylabel('parameter %d' % (i + 1))
    axes[-1, i].set_xlabel('parameter %d' % (i + 1))

# In[]:
plt.show()
fig_size = (60, 60)
# .figure()
plt.plot(np.linspace(0, 3999, 4000), values[0], label='values to fit')
plt.plot(np.linspace(0, 3999, 4000), model.simulate(set_to_test, times), label='starting point')
plt.plot(np.linspace(0, 3999, 4000), model.simulate(chains[0][4999, 0:13], times), label='EndChain')
# plt.plot(np.linspace(0, 3999, 4000), model.simulate([9.5, 0.45, 1.3, 1.8, 1.4, 1.5, 4.5, 3.4, 1.9, 2.4, 3.8, 2.9, 0.6],times), label = 'RealValue')
plt.plot(np.linspace(0, 3999, 4000), model.simulate(RealValue, times), label='RealValue')
# plt.plot(np.linspace(0, 3999, 4000), model.simulate(chains[0][4999],times), label = 'end point')
plt.legend()