import sabs_pkpd
import myokit
import numpy as np

class Constraint:
    def __init__(self, constraint_matrix, lower_bound=None, upper_bound=None):
        """
        To check whether a tested matrix M verifies the constraints conditions, a dot product is applied to each dimension
        of M and the constraint matrix and the following inequation is asserted:
        lb < constraint_matrix.dot(M) < ub

        :param constraint_matrix:
        list of numpy.array or numpy.array of dimension 3

        :param lower_bound:


        :param upper_bound:

        """
        self.matrix = constraint_matrix

        if lower_bound is not None:
            self.lb = lower_bound
            if len(self.matrix) != len(self.lb):
                raise ValueError('The constraint matrix length must match lower boundaries length')

        if upper_bound is not None:
            self.ub = upper_bound
            if len(self.matrix) != len(self.ub):
                raise ValueError('The constraint matrix length must match upper boundaries length')


    def verification(self, M):
        res = np.dot(self.matrix, M)
        verif = True
        if self.lb is not None:
            if np.shape(self.lb) != (len(self.matrix), np.shape(M)[1]):
                raise ValueError('')
            verif = (res > self.lb).all()



def objective_step_phase(duration, amplitude, sample_timepoints = 1000, normalise_output=True, constraint=None):

    """
    This function returns the score of separation of the models provided by sabs_pkpd.constants.s for the steps phase

    :param duration:
    list or numpy.array. Contains the list of durations of all of the steps for the step phase of the protocol

    :param amplitude:
    list or numpy.array. Contains the list of amplitudes of all of the steps for the step phase of the protocol

    :param sample_timepoints:
    int. Amount of points defining times at which the output is sampled, linearly spaced from 0 to sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time.

    :param normalise_output:
    bool. Defines whether the model output is normalised to the interval [0, 1] or not. True if not specified.

    :param constraint_matrix:
    numpy.array. Defines the constraint on the parameters. The constraint is verified by computing

    :return: score
    float. The score is computed as log of the sum of distances between each models.

    """
    if constraint_matrix is not None:

        return np.inf

    if len(duration) != len(amplitude):
        raise ValueError('Durations and Amplitudes for the step phase of the protocol must have the same number of values.')

    prot = sabs_pkpd.protocols.MyokitProtocolFromTimeSeries(duration, amplitude)

    response = np.zeros((len(sabs_pkpd.constants.s), sample_timepoints))
    for i in range(len(sabs_pkpd.constants.s)):
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


def objective_fourier_phase(low_freq, high_freq, freq_sampling, sample_timepoints, normalise_output=True):
    """

    :param low_freq:
    :param high_freq:
    :param freq_sampling:
    :param sample_timepoints:
    :param normalise_output:
    :return:
    """
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
