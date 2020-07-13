# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:06:09 2019

@author: yanral
"""

import sabs_pkpd
import myokit
import os


def get_steady_state(s,
                     time_to_steady_state: float,
                     data_exp: sabs_pkpd.load_data.Data_exp = None,
                     save_location: str = None,
                     list_of_models_names: list = None):
    """
    This function returns the (pseudo-)steady-state of the models provided.

    The output of the function is a list of length the number of models, and
    each index contains the number of different experimental conditions
    provided with data_exp.

    :param s:
    myokit.Simulation or list of myokit.Simulation. Contains the model(s) for
    which the steady-state will be computed.

    :param time_to_steady_state:
    float. Time at which to read out (pseudo-)steady-state.

    :param data_exp:
    sabs_pkpd.load_data.Data_exp. Contains the experimental conditions that the
    user wants to use to compute different steady-states. If not provided, the
    steady-state is also directly saved as default state of the model in s.

    :param save_location:
    str. Path to the folder where to save the produced mmt models at
    steady-state. If None, the MMT files aren't saved.

    :param list_of_models_names:
    list. List of the models names for save.

    :return: steady-state
    list. List of length the number of models, and each index contains the
    number of different experimental conditions provided with data_exp. Refer
    to steady_state[i][j] for the i-th model steady-state under j-th
    experimental conditions
    """

    # test types of variables provided
    if type(s) != myokit._sim.cvodesim.Simulation:
        if type(s) != list:
            raise ValueError('s should be provided as a myokit.Simulation or '
                             'list of myokit.Simulation')
        else:
            for j in range(len(s)):
                if type(s[j]) != myokit._sim.cvodesim.Simulation:
                    raise ValueError('s should be provided as a '
                                     'myokit.Simulation or list of '
                                     'myokit.Simulation')

    if data_exp is not None:
        if not hasattr(data_exp, 'fitting_instructions'):
            raise ValueError('data_exp should contain the '
                             'fitting_instructions attribute (put at least '
                             'exp_cond_param_annot) to allow set up of the '
                             'experimental conditions to compute steady-state')

    # Distinguish when several different models are provided
    if type(s) == list:
        # In case particular experimental conditions are not specified
        if data_exp is None:
            steady_state = []
            for i in range(len(s)):
                s[i].reset()
                s[i].run(time_to_steady_state)
                steady_state.append([s[i].state()])
                s[i].set_default_state(steady_state[i][0])

        # In case experimental conditions are specified, steady-state variables
        # are listed for each experimental condition
        else:
            steady_state = []
            for i in range(len(s)):
                model_ss = []
                for j in range(len(data_exp.exp_conds)):
                    s[i].reset()
                    s[i].set_constant(
                        data_exp.fitting_instructions.exp_cond_param_annot,
                        data_exp.exp_conds[j])
                    s[i].run(time_to_steady_state)
                    model_ss.append(s[i].state())
                steady_state.append(model_ss)

    # Distinguish when only one model is provided
    else:
        if data_exp is None:
            # In case particular experimental conditions are not specified
            s.reset()
            s.run(time_to_steady_state)
            steady_state = [[s.state()]]
            s.set_default_state(steady_state[0][0])
        else:
            # In case experimental conditions are specified, steady-state
            # variables are listed for each experimental condition
            steady_state = [[]]
            for i in range(len(data_exp.exp_conds)):
                s.reset()
                s.set_constant(
                    data_exp.fitting_instructions.exp_cond_param_annot,
                    data_exp.exp_conds[i])
                s.run(time_to_steady_state)
                steady_state[0].append(s.state())

    # Set the save of the models if relevant
    if save_location is not None:
        if list_of_models_names is None:
            if hasattr(sabs_pkpd.constants,
                       'protocol_optimisation_instructions'):
                list_of_models_names = sabs_pkpd.constants.\
                    protocol_optimisation_instructions.models
            else:
                list_of_models_names = []
                for i in range(len(steady_state)):
                    list_of_models_names.append('model ' + str(i))
        if type(s) == list:
            if len(list_of_models_names) != len(s):
                raise ValueError('')
        save_steady_state_to_mmt(s,
                                 steady_state,
                                 list_of_models_names,
                                 save_location)

    return steady_state


def save_steady_state_to_mmt(s,
                             steady_state: list,
                             list_of_models_names: list,
                             save_location: str):
    """
    Saves the steady states as new mmt models. The structure of the (if
    necessary created) folder is: save_location\\one folder per model\\one
    .mmt model per experimental condition.

    :param s:
    myokit.Simulation or list of myokit.Simulation. Contains the model(s) for
    which the steady-state will be computed.

    :param steady_state:
    list.  List of length the number of models, and each index contains the
    number of different experimental conditions provided with data_exp. Refer
    to steady_state[i][j] for the i-th model steady-state under j-th
    experimental conditions

    :param list_of_models_names:
    list of strings. List of the names of the models. If not specified, the
    models will be named model #.

    :param save_location:
    string. Link to the folder where to save the new model produced. If more
    than one model is provided, sub-directories matching the models names in
    list_of_models_names will be created, and a .mmt file created for each
    steady-state of the model (generated previously with different experimental
    conditions).

    :return: None
    """

    # Check the inputs in the function
    if len(steady_state) != len(list_of_models_names):
        raise ValueError('Steady_state and list_of_models_names should have '
                         'the same length.')

    # Create the folder of save if not existing yet
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    # Write the models into the folder, with one folder per model
    for i in range(len(list_of_models_names)):
        folder = os.path.join(save_location, list_of_models_names[i])
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Create a sub-folder for each experimental condition desired
        for j in range(len(steady_state[0])):
            save_filename = os.path.join(folder,
                                         list_of_models_names[i] +
                                         '_exp_cond_' + str(j) + '.mmt')
            s[i].set_default_state(steady_state[i][j])
            model_to_save = s[i]._model
            myokit.save_model(save_filename, model_to_save)

    return None


def clamp_experiment_model(model_filename,
                           clamped_variable_annot: str,
                           pace_variable_annotation: str,
                           protocol=None,
                           save_new_mmt_filename=None):
    """
    This function loads a mmt model, sets the equation for the desired variable
    to engine.pace (bound with the protocol), and returns the Myokit.model
    generated this way. If the user provides the argument for
    save_new_mmt_filename, the new model is also saved in the hard drive.

    :param model_filename: str
        Path and filename to the MMT model loaded.
    :param clamped_variable_annot: str
        MMT model annotation for the model variable clamped. This variable's
        values will be set by the protocol.
    :param pace_variable_annotation: str
        Model annotation for the variable bound to pace, used to read out
        information from the Myokit protocol. Usually, the annotation is either
        engine.pace or environment.pace
    :param protocol: Myokit.Protocol()
        If specified by the user, the protocol will be added to the MMT file
        saved.
    :param save_new_mmt_filename: str
        Path and filename to the location where the user wants to save the
        model ready for clamping simulations.
    :return: m: Myokit.Model()
        Returns a Myokit.Model() ready to be used with a clamping protocol to
        generate a Myokit.Simulation().
    """
    # Check that model_filename is provided as a mmt file
    if model_filename[-4:] != '.mmt':
        raise ValueError('The model_filename should lead to a MMT model.')

    # Load the MMT file depending on whether a protocol is entered as function
    # input
    if protocol is not None:
        m = myokit.load_model(model_filename)
    else:
        m, protocol, script = myokit.load(model_filename)

    # Analyse the clamped_variable_annot to find component name and variable
    # name
    i = clamped_variable_annot.index('.')
    component_name = clamped_variable_annot[0:i]
    variable_name = clamped_variable_annot[i + 1:]

    # Change the model to clamp the selected value
    original_protocol_component = m.get(component_name,
                                        class_filter=myokit.Component)

    variable_found = False

    for variable in original_protocol_component.variables():
        if variable.name() == variable_name:
            if variable.is_state():
                # Set the variable type to constant if needed
                variable.demote()
            # Bind the variable to the pace (= read from protocol)
            variable.set_rhs(pace_variable_annotation)
            variable_found = True

    if not variable_found:
        raise ValueError('The variable ' + clamped_variable_annot +
                         ' could not be found.')

    # Save the new model and protocol if the user provided the argument
    # save_new_mmt_filename
    if save_new_mmt_filename is not None:
        if protocol is not None:
            myokit.save(save_new_mmt_filename, model=m, protocol=protocol)
        else:
            myokit.save(save_new_mmt_filename, model=m)

    return m
