import sabs_pkpd
import myokit
import numpy as np


def objective(duration, amplitude, sample_timepoints = 1000, normalise_output=True):
    """
    This function returns the score of separation of the models provided by sabs_pkpd.constants.s

    :param duration:
    list or numpy.array. Contains the list of durations of all of the steps for the step phase of the protocol


    :param amplitude:
    :param sample_timepoints:
    :param normalise_output:
    :return:
    """

    step_score = step_phase_score(duration, amplitude, sample_timepoints, normalise_output)
    # fourier_score = fourier_phase_score(low_freq, high_freq, freq_sampling, sample_timepoints)
    fourier_score =0
    score = step_score + fourier_score

    return score


def step_phase_score(duration, amplitude, sample_timepoints = 1000, normalise_output=True):
    """
    This function returns the score of separation of the models provided by sabs_pkpd.constants.s for the steps phase

    :param duration:
    list or numpy.array. Contains the list of durations of all of the steps for the step phase of the protocol

    :param amplitude:
    list or numpy.array. Contains the list of amplitudes of all of the steps for the step phase of the protocol

    :param sample_timepoints:
    :param normalise_output:
    :return:
    """

    if len(duration) != len(amplitude):
        raise ValueError('Durations and Amplitudes for the step phase of the protocol must have the same number of values.')

    prot = sabs_pkpd.protocols.MyokitProtocolFromTimeSeries(duration, amplitude)

    response = np.zeros((len(sabs_pkpd.constants.s), sample_timepoints))
    for i in range(len(sabs_pkpd.constants.s)):
        sabs_pkpd.constants.s[i].set_time(0)
        sabs_pkpd.constants.s[i].reset()
        sabs_pkpd.constants.s[i].set_protocol(prot)
        simulated = sabs_pkpd.constants.s[i].run(
            sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time * 1.0001,
            log_times=np.linspace(0, sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time,
                                  sample_timepoints))
        response[i, :] = simulated[sabs_pkpd.constants.protocol_optimisation_instructions.model_readout]
        if normalise_output == True:
            response[i, :] = (response[i, :] - np.min(response[i, :])) / (
                        np.max(response[i, :]) - np.min(response[i, :]))

    score = 0
    for i in range(len(sabs_pkpd.constants.s) - 1):
        score_model = 0
        for j in range(len(sabs_pkpd.constants.s) - i):
            score_model += np.sum(np.square(response[i, :] - response[j, :]))
        score_model = np.log(score_model)
        score -= score_model

    return score


def fourier_phase_score(low_freq, high_freq, freq_sampling, sample_timepoints, normalise_output=True):

    if len(low_freq) >= len(high_freq):
        raise ValueError('Lowest frequency must be lower than highest frequency for the Fourier phase of the protocol.')

    duration, amplitude = sabs_pkpd.protocols.EventsListFromFourier(low_freq, high_freq, freq_sampling)

    prot = sabs_pkpd.protocols.MyokitProtocolFromTimeSeries(duration, amplitude)

    response = np.zeros((len(sabs_pkpd.constants.s), sample_timepoints))
    for i in range(len(sabs_pkpd.constants.s)):
        sabs_pkpd.constants.s[i].set_time(0)
        sabs_pkpd.constants.s[i].reset()
        sabs_pkpd.constants.s[i].set_protocol(prot)
        simulated = sabs_pkpd.constants.s[i].run(
            sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time * 1.0001,
            log_times=np.linspace(0, sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time,
                                  sample_timepoints))
        response[i, :] = simulated[sabs_pkpd.constants.protocol_optimisation_instructions.model_readout]
        if normalise_output == True:
            response[i, :] = (response[i, :] - np.min(response[i, :])) / (
                        np.max(response[i, :]) - np.min(response[i, :]))

    score = 0
    for i in range(len(sabs_pkpd.constants.s) - 1):
        score_model = 0
        for j in range(len(sabs_pkpd.constants.s) - i):
            score_model += np.sum(np.square(response[i, :] - response[j, :]))
        score_model = np.log(score_model)
        score -= score_model

    return None
