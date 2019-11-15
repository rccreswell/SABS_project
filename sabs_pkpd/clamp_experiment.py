# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:06:09 2019

@author: yanral
"""

import sabs_pkpd
import numpy as np
import matplotlib.pyplot as plt
import myokit

def clamp_experiment_model(model_filename, clamped_param_annot:str, protocol=None, save_new_mmt_filename=None):
    """
    This function loads a mmt model, sets the equation for the desired variable to engine.pace (bound with the protocol)
    , and returns the Myokit.model generated this way. If the user provides the argument for save_new_mmt_filename, the
    new model is also saved.
    :param model_filename: str
        Path and filename to the MMT model loaded.
    :param clamped_param_annot: str

    :param protocol:
    :param save_new_mmt_filename:
    :return:
    """
    if protocol is not None:
        m = myokit.load_model(model_filename)
    else:
        m, protocol, script = myokit.load(model_filename)

    # Analyse the clamped_param_annot to find component name and variable name
    i = clamped_param_annot.index('.')
    component_name = clamped_param_annot[0:i]
    variable_name = clamped_param_annot[i+1:]

    # Change the model to clamp the selected value
    original_protocol_component = m.get(component_name,
                                                class_filter=myokit.Component)

    for variable in original_protocol_component.variables():
        if variable.name() == variable_name:
            variable.set_rhs('engine.pace')

    # Save the new model and protocol if the user provided the argument save_new_mmt_filename
    if save_new_mmt_filename is not None:
        myokit.save(save_new_mmt_filename, model=m, protocol=protocol)

    return m


    # Run the new model
    s = myokit.Simulation(m, p)
    s.reset()
    # reset timer
    s.set_time(0)

    a = s.run(time_max, log_times=time_samples)
    output = a[read_out]



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
        variable.set_rhs('engine.pace')


# Save the new model and protocol if the user provided the argument save_new_mmt_filename
newmodelname = 'C:/Users/yanral/Documents/Software Development/tests/test resources/model_clamped.mmt'
myokit.save(newmodelname, model=m, protocol=p)


# Run the new model
s = myokit.Simulation(m, p)
s.reset()
# reset timer
s.set_time(0)

a = s.run(time_max, log_times=time_samples)
output = a[read_out]


