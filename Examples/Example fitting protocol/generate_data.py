# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:06:09 2019

@author: yanral
"""

import sabs_pkpd
import numpy as np
import matplotlib.pyplot as plt


# In[]
s = sabs_pkpd.run_model.set_myokit_simulation('C:/Users/yanral/Documents/2D maps/.mmt models/tentusscher_2006.mmt')


# In[]
time_max = 1000 # ms
n_timepoints = 4001
time_samples = np.linspace(0, time_max, n_timepoints)
pre_run = 20000
read_out = 'ikr.IKr'

#AP = sabs_pkpd.run_model.quick_simulate(sabs_pkpd.constants.s, time_max, read_out, 
#                                            fixed_params_annot=fixed_params_annot,
#                                                fixed_params_values=fixed_params_values,
#                                            pre_run=pre_run,
#                                            time_samples=time_samples)

exp_clamped_parameter_annot = 'membrane.V'
exp_clamped_parameter_values = -50 + 30 * np.sin(0.015*time_samples)
plt.plot(exp_clamped_parameter_values)

output = []

# In[]
s.reset()
# reset timer
s.set_time(0)

# Eventually run a pre-run to reach steady-state
s.pre(pre_run)

# Run the model step by step
for k in range(10):   
    for t in range(len(time_samples)-1):
        s.set_constant('membrane.V', exp_clamped_parameter_values[t])
        a = s.run(time_samples[t+1] - time_samples[t])
        
# In[]
# Run the model step by step
for t in range(len(time_samples)-1):
    s.set_constant('membrane.V', exp_clamped_parameter_values[t])
    a = s.run(time_samples[t+1] - time_samples[t])
    output.append(a['ikr.IKr'][0])

plt.plot(output)