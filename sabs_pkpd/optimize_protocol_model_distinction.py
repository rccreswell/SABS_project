import sabs_pkpd
import myokit
import numpy as np
import scipy


def objective(duration, amplitude, sample_timepoints = 1000):
    prot = sabs_pkpd.protocols.MyokitProtocolFromTimeSeries(duration, amplitude)

    response = np.zeros((len(sabs_pkpd.constants.s), sample_timepoints))
    for i in range(len(sabs_pkpd.constants.s)):
        simulation = myokit.Simulation(model=sabs_pkpd.constants.s[i], protocol=prot)
        simulated = simulation.run(sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time*1.0001,
                                   log_times=np.linspace(0, sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time, sample_timepoints))
        response[i, :] = simulated[sabs_pkpd.constants.protocol_optimisation_instructions.model_readout]
        response[i, :] = (response[i, :] - np.min(response[i, :])) / (np.max(response[i, :]) - np.min(response[i, :]))

    score = 0
    for i in range(len(sabs_pkpd.constants.s)-1):
        score_model = 0
        for j in range(len(sabs_pkpd.constants.s) - i):
            score_model += np.sum(np.square(response[i, :] - response[j, :]))
        score_model = np.log(score_model)
        score -= score_model

    return score




