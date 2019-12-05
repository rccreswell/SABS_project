import sabs_pkpd
import myokit
import numpy as np
import scipy


def objective(starting_times, duration, amplitude, baseline):
    events_list = sabs_pkpd.protocols.TimeSeriesFromSteps(starting_times, duration, amplitude, baseline=baseline)
    prot = sabs_pkpd.protocols.MyokitProtocolFromTimeSeries(events_list)

    sample_timepoints = 10000

    response = np.zeros((len(sabs_pkpd.constants.s), sample_timepoints))
    for i in range(len(sabs_pkpd.constants.s)):
        simulation = myokit.Simulation(model=sabs_pkpd.constants.s[i], protocol=prot)
        simulated = simulation.run(sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time*1.0001,
                                   log_times=np.linspace(0, sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time, sample_timepoints))
        response[i, :] = simulated[sabs_pkpd.constants.protocol_optimisation_instructions.model_readout]

    score = 0
    for i in range(len(sabs_pkpd.constants.s)):
        score_model = 0
        for j in range(len(sabs_pkpd.constants.s) - i):
            score_model += np.sum(np.square(response[i, :] - response[j, :]))
        score_model = np.log(score_model)
        score += score_model

    return -1.0 * score




