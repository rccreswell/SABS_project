# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:06:09 2019

@author: yanral
"""

import sabs_pkpd
import numpy as np
import myokit
import os


def get_steady_state(s, time_to_steady_state, data_exp=None, save_location=None, list_of_models_names =None):
    """
    This function returns the (pseudo-)steady-state of the models provided. The output of the function is a list of
    length the number of models, and each index contains the number of different experimental conditions provided with
    data_exp.

    :param s:
    myokit.Simulation or list of myokit.Simulation. Contains the model(s) for which the steady-state will be computed.

    :param data_exp:
    sabs_pkpd.load_data.Data_exp. Contains the experimental conditions that the user wants to use to compute different
    steady-states. If not provided, the steady-state is also directly saved as default state of the model in s.

    :param time_to_steady_state:
    float. Time at which to read out

    :return: steady-state
    list. List of length the number of models, and each index contains the number of different experimental conditions
    provided with data_exp. Refer to steady_state[i][j] for the i-th model steady-state under j-th experimental
    conditions
    """

    # test types of variables provided
    if data_exp is not None:
        if type(data_exp) != sabs_pkpd.load_data.Data_exp:
            raise ValueError('data_exp should be provided as a class sabs_pkpd.load_data.Data_exp.')

    if type(s) != myokit._sim.cvodesim.Simulation:
        if type(s) != list:
            raise ValueError('s should be provided as a myokit.Simulation or list of myokit.Simulation')
        else:
            for j in range(len(s)):
                if type(s[j]) != myokit._sim.cvodesim.Simulation:
                    raise ValueError('s should be provided as a myokit.Simulation or list of myokit.Simulation')

    if data_exp is not None:
        if not hasattr(data_exp, 'fitting_instructions'):
            raise ValueError('data_exp should contain the fitting_instructions attribute (at least exp_cond_param_annot)'
                             'to allow set up of the experimental conditions to compute steady-state')

    # Distinguish one or more different models
    if type(s) == list:
        if data_exp is None:
            steady_state = []
            for i in range(len(s)):
                s[i].reset()
                s[i].run(time_to_steady_state)
                steady_state.append([s[i].state()])
                s[i].set_default_state(steady_state[i][0])

        else:
            steady_state = []
            for i in range(len(s)):
                model_ss = []
                for j in range(len(data_exp.exp_conds)):
                    s[i].reset()
                    s[i].set_constant(data_exp.fitting_instructions.exp_cond_param_annot, data_exp.exp_conds[j])
                    s[i].run(time_to_steady_state)
                    model_ss.append(s[i].state())
                steady_state.append(model_ss)

    else:
        if data_exp is None:
            s.reset()
            s.run(time_to_steady_state)
            steady_state = [[s.state()]]
            s.set_default_state(steady_state[0][0])
        else:
            steady_state = [[]]
            for i in range(len(data_exp.exp_conds)):
                s.reset()
                s.set_constant(data_exp.fitting_instructions.exp_cond_param_annot, data_exp.exp_conds[i])
                s.run(time_to_steady_state)
                steady_state[0].append(s.state())

    if save_location is not None:
        if list_of_models_names is None:
            if hasattr(sabs_pkpd.constants, 'protocol_optimisation_instructions'):
                list_of_models_names = sabs_pkpd.constants.protocol_optimisation_instructions.models
            else:
                list_of_models_names = []
                for i in range(len(steady_state)):
                    list_of_models_names.append('model ' + str(i))
            save_steady_state_to_mmt(s, steady_state, list_of_models_names, save_location)

    return steady_state


def save_steady_state_to_mmt(s, steady_state, list_of_models_names, save_location):
    """
    Saves the steady states as new mmt models. The structure of the (if necessary created) folder is:
    save_location\\one folder per model\\one .mmt model per experimental condition.

    :param s:
    myokit.Simulation or list of myokit.Simulation. Contains the model(s) for which the steady-state will be computed.

    :param steady-state:
    list.  List of length the number of models, and each index contains the number of different experimental conditions
    provided with data_exp. Refer to steady_state[i][j] for the i-th model steady-state under j-th experimental
    conditions

    :param save_filename:
    string. Filename for the location where to save the new model produced. If more than one model is provided,
    sub-directories matching the models names in list_of_models_names will be created, and a .mmt file created for each
    steady-state of the model (generated previously with different experimental conditions).

    :param list_of_models_names:
    list of strings. List of the names of the models. If not specified, the models will be named model #.

    :return:
    """

    if len(steady_state) != list_of_models_names:
        raise ValueError('Steady_state and list_of_models_names should have the same length.')

    if not os.path.exists(save_location):
        os.makedirs(save_location)
    for i in range(len(list_of_models_names)):
        folder = os.path.join(save_location, list_of_models_names[i])
        if not os.path.exists(folder):
            os.makedirs(folder)

        for j in range(len(steady_state[0])):
            save_filename = os.path.join(folder, list_of_models_names[i] +'_exp_cond_' + str(j) + '.mmt')
            model_to_save = s[i][j]._model
            myokit.save_model(save_filename, model_to_save)


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