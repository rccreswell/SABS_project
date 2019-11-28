# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:06:09 2019

@author: yanral
"""

import sabs_pkpd
import numpy as np
import matplotlib.pyplot as plt
import myokit

def clamp_experiment_model(model_filename, clamped_variable_annot:str, pace_variable_annotation:str, protocol=None, save_new_mmt_filename=None):
    """
    This function loads a mmt model, sets the equation for the desired variable to engine.pace (bound with the protocol)
    , and returns the Myokit.model generated this way. If the user provides the argument for save_new_mmt_filename, the
    new model is also saved.
    :param model_filename: str
        Path and filename to the MMT model loaded.
    :param clamped_param_annot: str
        MMT model annotation for the model variable clamped. This variable's values will be set by the protocol.
    :param pace_variable_annotation: str
        Model annotation for the variable bound to pace, used to read out information from the Myokit protocol. Usually,
        the annotation is either engine.pace or environment.pace
    :param protocol: Myokit.Protocol()
        If specified by the user, the protocol will be added to the MMT file saved.
    :param save_new_mmt_filename: str
        Path and filename to the location where the user wants to save the model ready for clamping simulations.
    :return: m: Myokit.Model()
        Returns a Myokit.Model() ready to be used with a clamping protocol to generate a Myokit.Simulation().
    """
    if protocol is not None:
        m = myokit.load_model(model_filename)
    else:
        m, protocol, script = myokit.load(model_filename)

    # Analyse the clamped_param_annot to find component name and variable name
    i = clamped_variable_annot.index('.')
    component_name = clamped_variable_annot[0:i]
    variable_name = clamped_variable_annot[i+1:]

    # Change the model to clamp the selected value
    original_protocol_component = m.get(component_name,
                                                class_filter=myokit.Component)

    variable_found = False

    for variable in original_protocol_component.variables():
        if variable.name() == variable_name:
            if variable.is_state() == True:
                variable.demote()
            variable.set_rhs(pace_variable_annotation)
            variable_found = True

    if variable_found == False:
        raise ValueError('The variable ' + clamped_variable_annot + ' could not be found.')

    # Save the new model and protocol if the user provided the argument save_new_mmt_filename
    if save_new_mmt_filename is not None:
        if protocol is not None:
            myokit.save(save_new_mmt_filename, model=m, protocol=protocol)
        else:
            myokit.save(save_new_mmt_filename, model=m)

    return m