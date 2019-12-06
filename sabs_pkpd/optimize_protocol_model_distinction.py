import sabs_pkpd
import myokit
import numpy as np


def objective(duration, amplitude, sample_timepoints = 1000, normalise_output=True):
    prot = sabs_pkpd.protocols.MyokitProtocolFromTimeSeries(duration, amplitude)

    response = np.zeros((len(sabs_pkpd.constants.s), sample_timepoints))
    for i in range(len(sabs_pkpd.constants.s)):
        sabs_pkpd.constants.s[i].set_time(0)
        sabs_pkpd.constants.s[i].reset()
        sabs_pkpd.constants.s[i].set_protocol(prot)
        simulated = sabs_pkpd.constants.s[i].run(sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time*1.0001,
                                   log_times=np.linspace(0, sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time, sample_timepoints))
        response[i, :] = simulated[sabs_pkpd.constants.protocol_optimisation_instructions.model_readout]
        if normalise_output == True:
            response[i, :] = (response[i, :] - np.min(response[i, :])) / (np.max(response[i, :]) - np.min(response[i, :]))

    score = 0
    for i in range(len(sabs_pkpd.constants.s)-1):
        score_model = 0
        for j in range(len(sabs_pkpd.constants.s) - i):
            score_model += np.sum(np.square(response[i, :] - response[j, :]))
        score_model = np.log(score_model)
        score -= score_model

    return score
