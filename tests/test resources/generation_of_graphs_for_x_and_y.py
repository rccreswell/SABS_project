# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:58:08 2019

@author: yanral
"""

import sabs_pkpd
import numpy as np
import matplotlib.pyplot as plt


# In[]

sabs_pkpd.constants.s = sabs_pkpd.load_model.load_simulation_from_mmt(
    'C:/Users/yanral/Documents/Software Development'
    '/tests/test resources/pints_problem_def_test.mmt')
time_max = 10
read_out1 = 'comp1.x'
read_out2 = 'comp1.y'
exp_cond_param_annot = 'constants.T'
exp_cond_param_values = [20, 37]
fixed_params_annot = ['constants.unknown_cst', 'constants.unknown_cst2']
fixed_params_values = [0.1, 0.1]
time_samples = [0, 0.01, 0.05, 0.1, 0.3, 0.5, 1, 5]

x = sabs_pkpd.run_model.quick_simulate(sabs_pkpd.constants.s,
                                       time_max,
                                       read_out1,
                                       exp_cond_param_annot,
                                       exp_cond_param_values,
                                       fixed_params_annot,
                                       fixed_params_values,
                                       time_samples=time_samples)
y = sabs_pkpd.run_model.quick_simulate(sabs_pkpd.constants.s,
                                       time_max,
                                       read_out2,
                                       exp_cond_param_annot,
                                       exp_cond_param_values,
                                       fixed_params_annot,
                                       fixed_params_values,
                                       time_samples=time_samples)


# In[]

plt.figure()
plt.plot(time_samples, x[0], label='comp1.x')
plt.plot(time_samples, y[0], label='comp1.y')
plt.xlabel('Time')
plt.ylabel('model output')
plt.legend()
plt.title('T = 20')

plt.figure()
plt.plot(time_samples, x[1], label='comp1.x')
plt.plot(time_samples, y[1], label='comp1.y')
plt.xlabel('Time')
plt.ylabel('model output')
plt.legend()
plt.title('T = 37')
