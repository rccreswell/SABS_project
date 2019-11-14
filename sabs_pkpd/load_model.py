"""This file contains functions for loading a model from file.

Functionality for parsing CellML models including conversion of stimulus
protocol) is included.
"""

import myokit
import myokit.formats.cellml
import myokit.formats
import matplotlib.pyplot as plt


def convert_protocol(model):
    """Convert the cellml stimulus protocol the the correct format for myokit.

    CellML models may contain a stimulus current. Myokit's CellML import does
    not automatically convert the stimulus current into the Myokit pacing
    protocol. This function detects the stimulus protocol model component after
    import from CellML and parses the contents for rewriting in Myokit pacing
    format.

    Parameters
    ----------
    model : myokit.Model
        The model as imported from cellml

    Returns
    -------
    myokit.Model
        The model with corrected stimulus protocol
    myokit.Protocol
        stimulus protocol using Myokit's pacing system
    """
    # Get the original protocol from cellml
    original_protocol_component = model.get('stimulus_protocol',
                                            class_filter=myokit.Component)

    # Read the parameters from the original protocol
    equations = original_protocol_component.equations()
    pacing_parameters = {}
    for equation in equations:
        if equation.lhs.pystr() in ['i_Stim_Start',
                                    'i_Stim_Period',
                                    'i_Stim_PulseDuration']:
            pacing_parameters[equation.lhs.pystr()] = equation.rhs
        if equation.lhs.pystr() == 'i_Stim_Amplitude':
            amplitude = equation.rhs

    # Add the level variable which is bound to pace
    level = original_protocol_component.add_variable('level')
    level.set_rhs(0)
    level.set_binding('pace')

    # Delete those variables which are no longer needed in the stimulus model
    # component
    variables_to_delete = []
    for variable in original_protocol_component.variables():
        if variable.name() == 'i_Stim_Amplitude':
            variable.set_rhs(0.5)
        elif variable.name() == 'Istim':
            variable.set_rhs(0.5)
        elif variable.name() == 'level':
            pass
        else:
            variables_to_delete.append(variable)

    for variable in variables_to_delete:
        original_protocol_component.remove_variable(variable)

    # Write the protocol in Myokit format
    new_protocol = myokit.Protocol()
    new_protocol.schedule(1.0,
                          pacing_parameters['i_Stim_Start'],
                          pacing_parameters['i_Stim_PulseDuration'],
                          period=pacing_parameters['i_Stim_Period'],
                          multiplier=0)

    return model, new_protocol


def load_simulation_from_cellml(filename):
    """Load a model into Myokit from cellml file format.

    Parameters
    ----------
    filename : str
        Path to the CellML file

    Returns
    -------
    myokit.Simulation
        Myokit Simulation object from the cellml file
    """
    if 'cellml' not in myokit.formats.importers():
        raise Exception('cellml support not detected in your Myokit')


    importer = myokit.formats.importer('cellml')
    model = importer.model(filename)

    model, protocol = convert_protocol(model)
    s = myokit.Simulation(model, protocol)

    return s

    # Code for a simulation
    # d = s.run(1000)
    # first_state = next(model.states())
    # var = first_state.qname()
    # plt.plot(d.time(), d[var])
    # plt.title(var)
    # plt.show()


def load_simulation_from_mmt(filename):
    """Load a model into Myokit from MMT file format.

    Parameters
    ----------
    filename : str
        Path to the MMT file

    Returns
    -------
    myokit.Simulation
        Myokit Simulation object from the MMT file
    """
    model, prot, script = myokit.load(filename)
    return myokit.Simulation(model, prot)


if __name__ == '__main__':
    load_simulation_from_cellml('beeler_reuter_1977.cellml')
    exit()

    print(myokit.formats.cellml.CellMLImporter)
    fname = '../tests/test resources/pints_problem_def_test.mmt'
    x = load_simulation_from_mmt(fname)
    print(x)
