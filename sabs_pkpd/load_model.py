"""Functions for loading a model from file.
"""

import myokit
import myokit.formats.cellml
import myokit.formats
import matplotlib.pyplot as plt


def convert_protocol(stimulus_protocol):
    """Convert a cellml stimulus protocol to Myokit format.

    Parameters
    ----------
    stimulus_protocol : str
        The cellml stimulus protocol

    Returns
    -------
    protocol...
    """
    pass


def load_simulation_from_cellml(filename):
    """Load a model into Myokit from cellml file format.

    Parameters
    ----------
    filename : str
        Path to the CellML file

    Returns
    -------
    myokit._sim.cvodesim.Simulation
        Myokit Simulation object from the cellml file
    """
    if 'cellml' not in myokit.formats.importers():
        raise Exception('cellml support not detected in your Myokit')


    importer = myokit.formats.importer('cellml')
    model = importer.model(filename)

    # p = myokit.Protocol()
    # s = myokit.Simulation(model, p)
    #
    # d = s.run(1000)
    # first_state = next(model.states())
    # var = first_state.qname()
    # plt.plot(d.time(), d[var])
    # plt.title(var)
    # plt.show()

    if model.has_component('stimulus_protocol'):
        protocol_component = model.get('stimulus_protocol', class_filter=myokit.Component)

        equations = protocol_component.equations()
        # for eq in equations:
        #     print(eq.lhs, eq.lhs.pystr(), type(eq.lhs))
        #     #print(eq.rhs, eq.rhs.unit(), type(eq.rhs))

        # Get the Start, Duration, and Period, which will go in the pacing protocol
        # Get IstimAmplitude, which is kept as is
        pacing_parameters = {}
        amplitude = None
        for equation in equations:
            if equation.lhs.pystr() in ['IstimStart', 'IstimPeriod', 'IstimPulseDuration']:
                pacing_parameters[equation.lhs.pystr()] = equation.rhs
            if equation.lhs.pystr() == 'IstimAmplitude':
                amplitude = equation

        new_stimulus_protocol = myokit.Component(model, 'stimulus_protocol')
        new_protocol = myokit.Protocol()

        # Make the level bind
        new_amplitude = myokit.Variable(new_stimulus_protocol, 'IstimAmplitude')
        new_amplitude.set_rhs(amplitude.rhs)
        #new_amplitude.set_unit()
        # new_stimulus_protocol.add_variable('IstimAmplitude')

        level = myokit.Variable(new_stimulus_protocol, 'level')
        level.set_rhs(0)
        level.set_binding('pace')

        # Make a new Istim =
        istim = myokit.Variable(new_stimulus_protocol, 'Istim')
        istim.set_rhs(myokit.Multiply(myokit.Name(level), myokit.Name(new_amplitude)))

        protocol_component = new_stimulus_protocol

        new_protocol.schedule(1.0, pacing_parameters['IstimStart'], pacing_parameters['IstimPulseDuration'], period=pacing_parameters['IstimPeriod'], multiplier=0)

        # s = myokit.Simulation(model, new_protocol)
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
    myokit._sim.cvodesim.Simulation
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
