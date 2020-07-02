"""This file contains code for input stimulus protocols.

In cardiac electrophysiology models, an input signal is provided to the cell.
This file collects functions which are used for modelling certain forms of
input protocol.
"""

import numpy as np
import scipy.interpolate
import myokit
import sabs_pkpd


def one_step_protocol(amplitude, duration):
    """A protocol with one square step of arbitrary duration and magnitude.

    The start of the step is fixed at t=1.0.

    Parameters
    ----------
    amplitude : float
        The magnitude of the step.
    duration : float
        Time length of the step

    Returns
    -------
    function
        The protocol signal as a function of time
    """
    return lambda times: np.array(((times > 1.0) & (times < 1.0 + duration)))\
        .astype(float) * amplitude


def sine_wave_protocol(amplitude, frequency):
    """A protocol based on a sine wave form.

    Parameters
    ----------
    amplitude : float
        Amplitude of the sine wave
    frequency : float
        Frequency of the sine wave

    Returns
    -------
    function
        The protocol signal as a function of time
    """
    return lambda times: amplitude * np.sin(frequency * times)


def pointwise_protocol(times, values):
    """An arbitrary protocol based on a sequence of values.

    A linear interpolant is used between the points.

    Parameters
    ----------
    times : list of float
        Time points where the protocol is defined
    values : list of float
        The value of the protocol at each given time point

    Returns
    -------
    scipy.interpolate.interpolate.interp1d
        The protocol signal as a function of time
    """
    times = np.array(times)
    values = np.array(values)

    f_int = scipy.interpolate.interp1d(times,
                                       values,
                                       fill_value=0.0,
                                       bounds_error=False)

    return f_int


def TimeSeriesFromSteps(start_times_list,
                        duration_list,
                        amplitude_list,
                        baseline=-80):
    """Returns time series of a protocol defined by steps on top of each other.

    Parameters
    ----------
    start_times_list : list or numpy.array
        List of times of start of each step.
    duration_list : list or numpy.array
        List of durations of each step
    amplitude_list : list or numpy.array
        List of amplitudes of each step
    baseline : float
        Defines the baseline of the protocol, to which the steps are added. If
        not specified, the default value is -80.

    Returns
    -------
    array
        Time series of the model parameter clamped during the protocol.
    """
    times = np.array([0, start_times_list[0],
                      start_times_list[0] + duration_list[0]])
    values = np.array([baseline, baseline + amplitude_list[0], baseline])

    for i in range(1, len(start_times_list)):
        index_start = np.max(np.where(times <= start_times_list[i]))
        index_end = np.max(np.where(times <= start_times_list[i] +
                                    duration_list[i]))

        if times[index_start] == start_times_list[i]:
            times = np.insert(times, index_end + 1, start_times_list[i] +
                              duration_list[i])
            values = np.insert(values, index_end + 1, values[index_end])
            values[index_start:index_end + 1] += amplitude_list[i]

        elif times[index_end] == start_times_list[i] + duration_list[i]:
            times = np.insert(times, index_start + 1, start_times_list[i])
            values = np.insert(values, index_start + 1, values[index_start])
            values[index_start + 1:index_end] += amplitude_list[i]

        else:
            times = np.insert(times, index_start + 1, start_times_list[i])
            times = np.insert(times, index_end + 2, start_times_list[i] +
                              duration_list[i])
            values = np.insert(values, index_start + 1, values[index_start])
            values = np.insert(values, index_end + 2, values[index_end + 1])
            values[index_start + 1:index_end + 2] += amplitude_list[i]

    return np.vstack((times, values))


def MyokitProtocolFromTimeSeries(durations, amplitudes):
    """Translates a time series of events to a Myokit Protocol.

    Parameters
    ----------
    durations : numpy.array
        Array of shape(1, number of steps). Contains the durations of all steps
        of the protocol
    amplitudes : numpy.array
        Array of shape(1, number of steps). Contains the amplitudes of all
        steps of the protocol

    Returns
    -------
    myokit.protocol
        The output protocol in Myokit format
    """

    prot = myokit.Protocol()
    starting_time = 0
    for i, duration in enumerate(durations):
        prot.schedule(amplitudes[i], starting_time, duration)
        starting_time += duration

    sim_time = sabs_pkpd.constants.protocol_optimisation_instructions.\
        simulation_time

    if starting_time < sim_time and \
            sabs_pkpd.constants.protocol_optimisation_instructions != []:
        prot.schedule(amplitudes[-1],
                      starting_time,
                      sim_time - starting_time + 100)

    return prot


def MyokitProtocolFromFourier(real_part, imag_part, low_freq, high_freq):
    """This function provides the Myokit Protocol for a Fourier spectrum
    defined by its real and imaginary parts. Please note that the returned
    protocol starts at time t=0. It may lead to issues if the Myokit Simulation
    time is not reset.

    Parameters
    ----------
    real_part : 1D-list or 1D-numpy.array
        Contains the real part of the Fourier spectrum.
    imag_part : 1D-list or 1D-numpy.array
        Contains the imaginary part of the Fourier spectrum.
    low_freq : float
        Defines the lowest frequency of the Fourier transform
    high_freq : float
        Defines the highest frequency of the Fourier transform

    Returns
    -------
    Myokit.Protocol
        Myokit Protocol corresponding to the provided Fourier spectrum.
    """
    fourier_spectrum, frequencies = \
        sabs_pkpd.fourier.fourier_spectrum_from_parameters(
            real_part, imag_part, low_freq, high_freq)
    values, times = \
        sabs_pkpd.fourier.time_series_from_fourier_spectrum(
            fourier_spectrum, frequencies)

    durations = np.ones(len(values)) * (times[1] - times[0])
    prot = MyokitProtocolFromTimeSeries(durations, values)

    return prot
