# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:06:09 2019

@author: yanral
"""

import sabs_pkpd
import numpy as np
import matplotlib.pyplot as plt
import myokit


# In[]
filename1 = 'C:/Users/yanral/Documents/Software Development/Examples/Example loading CellML model/tentusscher_2006.mmt'
save_filename1 = 'C:/Users/yanral/Documents/Software Development/Examples/Example fitting protocol/tt06_clamp_prot.mmt'
filename2 = 'C:/Users/yanral/Documents/Software Development/Examples/Example loading CellML model/ohara_rudy_cipa_v1_2017.mmt'
save_filename2 = 'C:/Users/yanral/Documents/Software Development/Examples/Example fitting protocol/ORd_2017_clamp_prot.mmt'
s = sabs_pkpd.load_model.load_simulation_from_mmt(filename1)

# In[]
time_max = 600 # ms
n_timepoints = 5001
time_samples = np.linspace(0, time_max, n_timepoints)
pre_run = 20000

read_out1 = 'ikr.IKr'
read_out2 = 'IKr.IKr'
exp_clamped_parameter_annot1 = 'membrane.V'
exp_clamped_parameter_annot2 = 'membrane.v'
pace_annot1 = 'engine.pace'
pace_annot2 = 'environment.pace'

# Run one time the model to get the voltage clamped to the shape of the AP
s.pre(pre_run)
a = s.run(time_max*1.001, log_times=time_samples)
exp_clamped_parameter_values = a[exp_clamped_parameter_annot1]

model1 = sabs_pkpd.clamp_experiment.clamp_experiment_model(filename1, exp_clamped_parameter_annot1, pace_variable_annotation=pace_annot1, save_new_mmt_filename=save_filename1)
model2 = sabs_pkpd.clamp_experiment.clamp_experiment_model(filename2, exp_clamped_parameter_annot2,pace_variable_annotation=pace_annot2, save_new_mmt_filename=save_filename2)

s1 = sabs_pkpd.load_model.load_simulation_from_mmt(save_filename1)
s2 = sabs_pkpd.load_model.load_simulation_from_mmt(save_filename2)

p = myokit.Protocol()
for i in range(len(time_samples)-1):
    p.schedule(exp_clamped_parameter_values[i], time_samples[i], time_samples[i+1] - time_samples[i])

s1.set_protocol(p)
s2.set_protocol(p)

a1 = s1.run(time_max*1.001, log_times= time_samples)
a2 = s2.run(time_max*1.001, log_times= time_samples)

response1 = a1[read_out1]
response2 = a2[read_out2]

plt.figure(0)
plt.subplot(3,1,1)
plt.plot(time_samples, exp_clamped_parameter_values)
plt.subplot(3,1,2)
plt.plot(time_samples, response1)
plt.subplot(3,1,3)
plt.plot(time_samples, response2)

plt.figure(1)
plt.plot(time_samples, a1['membrane.V'])