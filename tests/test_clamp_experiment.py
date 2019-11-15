# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:06:09 2019

@author: yanral
"""
import sabs_pkpd
import numpy as np
import matplotlib.pyplot as plt
import myokit


def test_clamp_experiment_model():
    modelname = 'C:/Users/yanral/Documents/Software Development/tests/test resources/pints_problem_def_test.mmt'

    # Design the clamp protocol
    time_max = 30
    n_timepoints = 100
    time_samples = np.linspace(2, time_max, n_timepoints)
    read_out = 'comp1.x'
    exp_clamped_parameter_annot = 'comp1.y'
    exp_clamped_parameter_values = 1 + np.sin(time_samples)

    p = myokit.Protocol()
    for i in range(len(time_samples) - 1):
        p.schedule(exp_clamped_parameter_values[i], time_samples[i], time_samples[i + 1] - time_samples[i])

    # Save the new model and protocol if the user provided the argument save_new_mmt_filename
    newmodelname = 'C:/Users/yanral/Documents/Software Development/tests/test resources/model_clamped.mmt'

    m = sabs_pkpd.clamp_experiment.clamp_experiment_model(modelname, exp_clamped_parameter_annot, p, newmodelname)

    s = sabs_pkpd.load_model.load_simulation_from_mmt(newmodelname)

    s.reset()
    # reset timer
    s.set_time(0)

    a = s.run(time_max, log_times=time_samples)
    output = a[read_out]

    #expected_output = np.array([0.00901, 0.06237, 0.06472, 0.11874, 0.12048, 0.17507, 0.17629, 0.23134, 0.23215])
    #diff = np.linalg.norm(output-expected_output)

    #assert diff < 0.0001

    return output

res = test_clamp_experiment_model()
plt.plot(res)