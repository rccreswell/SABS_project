# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:06:09 2019

@author: yanral
"""

import sabs_pkpd
import numpy as np
import matplotlib.pyplot as plt
import myokit

modelname = 'C:/Users/yanral/Documents/Software Development/tests/test resources/pints_problem_def_test.mmt'
# In[]
m, p, script = myokit.load(modelname)


# Design the clamp protocol
time_max = 30
n_timepoints = 100
time_samples = np.linspace(0, time_max, n_timepoints)
read_out = 'comp1.x'
exp_clamped_parameter_annot = 'comp1.y'
exp_clamped_parameter_values = 1 + np.sin(time_samples)

p = myokit.Protocol()
for i in range(len(time_samples) - 1):
    p.schedule(exp_clamped_parameter_values[i], time_samples[i], time_samples[i + 1] - time_samples[i])


# Change the model to clamp the selected value
original_protocol_component = m.get('comp1',
                                            class_filter=myokit.Component)
for variable in original_protocol_component.variables():
    if variable.name() == 'y':
        variable.set_rhs(0)


# Save the new model and protocol
newmodelname = 'C:/Users/yanral/Documents/Software Development/tests/test resources/model_clamped.mmt'
myokit.save(newmodelname, model=m, protocol=p)


# Run the new model
s = myokit.Simulation(m, p)
s.reset()
# reset timer
s.set_time(0)

a = s.run(time_max, log_times=time_samples)
output = a[read_out]


