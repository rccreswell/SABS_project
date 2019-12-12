import sabs_pkpd
import myokit
import numpy as np

class Constraint:
    def __init__(self, fun, lower_bound=None, upper_bound=None):
        """
        To check whether a tested matrix M verifies the constraints conditions, a dot product is applied to each dimension
        of M and the constraint matrix and the following inequation is asserted:
        lb[i] < numpy.matmul(constraint_matrix[i], M[i]) < ub[i]

        :param constraint_matrix:
        list of numpy.array or numpy.array of dimension 3

        :param lower_bound:
        list or numpy.array. Lower boundary for the result of the matrix multiplication of constraint matrix by tested
        matrix.

        :param upper_bound:
        list or numpy.array. Lower boundary for the result of the matrix multiplication of constraint matrix by tested
        matrix.
        """
        self.fun = fun

        if lower_bound is not None:
            self.lb = lower_bound

        if upper_bound is not None:
            self.ub = upper_bound

    def verification(self, M):
        res = self.fun(M)

        if self.lb is not None and self.ub is not None:
            if (res > self.lb).all() and (res < self.ub).all():
                verif = True
            else:
                return False

        elif self.lb is None and self.ub is not None:
            if (res < self.ub).all():
                verif = True
            else:
                return False

        elif self.lb is not None and self.ub is None:
            if (res > self.lb).all():
                verif = True
            else:
                return False

        elif self.lb is None and self.ub is None:
            verif = True

        return verif


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
    sabs_pkpd.optimize_protocol_model_distinction.Constraint. Used to verify that provided parameters are verifying the
    constraints.

    :return: score
    float. The score is computed as log of the sum of distances between each models.

    """
    if constraint is not None:
        verification = constraint.verification([duration, amplitude])
        if verification == False:
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
