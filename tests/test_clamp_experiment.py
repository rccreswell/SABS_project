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
    modelname = './tests/test resources/pints_problem_def_test.mmt'

    # Design the clamp protocol
    time_max = 30
    n_timepoints = 10
    time_samples = np.linspace(2, time_max, n_timepoints)
    read_out = 'comp1.x'
    exp_clamped_parameter_annot = 'comp1.y'
    exp_clamped_parameter_values = 1 + np.sin(time_samples)

    p = myokit.Protocol()
    for i in range(len(time_samples) - 1):
        p.schedule(exp_clamped_parameter_values[i], time_samples[i], time_samples[i + 1] - time_samples[i])

    # Save the new model and protocol if the user provided the argument save_new_mmt_filename
    newmodelname = './tests/test resources/model_clamped.mmt'

    m = sabs_pkpd.clamp_experiment.clamp_experiment_model(modelname, exp_clamped_parameter_annot, 'engine.pace', p, newmodelname)

    s = sabs_pkpd.load_model.load_simulation_from_mmt(newmodelname)

    s.reset()
    # reset timer
    s.set_time(0)

    a = s.run(time_max, log_times=time_samples)
    output = a[read_out]

    expected_output = np.array([0.00901, 0.02621, 0.00976, 0.02642, 0.00952, 0.02661, 0.00935, 0.02676, 0.00922])
    diff = np.linalg.norm(output-expected_output)

    assert diff < 0.001
