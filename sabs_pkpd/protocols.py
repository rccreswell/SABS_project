"""Classes for parametric forms of protocols.

May be useful for protocol optimization.
"""

import pandas
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import myokit
import sabs_pkpd
from operator import itemgetter

class Protocol:
    """Class for any protocol.

    Methods
    -------
    plot
        View a plot of the protocol.
    """
    def plot(self):
        """View a plot of the protocol.
        """
        plt.plot(self.relevant_times(), self.value(self.relevant_times()))
        plt.show()


class OneStepProtocol(Protocol):
    """The stimulus starts and ends at zero with a step of some magnitude.

        :------:    < magnitude
        |      |
        |      |
    ----:      :----------   < 0
        ^      ^
      start   start+duration
    """
    def __init__(self, start, duration, magnitude):
        """
        Parameters
        ----------
        start : float
            The time when the step starts
        duration : float
            How long the step lasts after it starts
        magnitude : float
            The magnitude of the stimulus when the step is active
        """
        self.start = start
        self.duration = duration
        self.magnitude = magnitude


    def value(self, t):
        """Calculate the stimulus signal over time.

        Parameters
        ----------
        t : np.ndarray
            The times at which to evaluate

        Returns
        -------
        np.ndarray
            The values of the stimulus signal at the given times
        """
        return ((t > self.start) & (t < self.start + self.duration)).astype(float) \
               * self.magnitude


    def relevant_times(self, time_points=1000):
        """Get an set of time points containing all the activity.

        Parameters
        ----------
        time_points : int, optional (1000)
            Number of time points

        Returns
        -------
        np.ndarray
            The time points covering all protocol activity
        """
        return np.linspace(self.start - 0.1*self.duration,
                           self.start + 1.5*self.duration,
                           1000)

    def to_myokit(self):
        """Convert the protocol to Myokit format.

        Returns
        -------
        myokit.Protocol
            The protocol as a Myokit object
        """
        p = myokit.Protocol()
        p.schedule(self.magnitude, self.start, self.duration)
        return p


class TwoStepProtocol(Protocol):
    """Protocol with two consecutive steps of different magnitudes.

            :-------:     < magnitude_1
            |       |
            |       |
            |       |
    --------:       |    :------------   < 0
                    |    |
                    :----:     < magnitude_2
            ^       ^    ^
            t0      t1   t2
    t0 = start
    t1 = start + duration_1
    t2 = start + duration_1 + duration_2
    """
    def __init__(self, start, duration_1, duration_2, magnitude_1, magnitude_2):
        """
        Parameters
        ----------
        start : float
            The time when the first step starts
        duration_1 : float
            How long the first step lasts
        duration_2 : float
            How long the second step lasts
        magnitude_1 : float
            The magnitude of the first step
        magnitude_2 : float
            The magnitude of the second step
        """
        self.start = start
        self.duration_1 = duration_1
        self.duration_2 = duration_2
        self.magnitude_1 = magnitude_1
        self.magnitude_2 = magnitude_2


    def value(self, t):
        """Calculate the stimulus signal over time.

        Parameters
        ----------
        t : np.ndarray
            The times at which to evaluate

        Returns
        -------
        np.ndarray
            The values of the stimulus signal at the given times
        """
        return ((t > self.start) & (t < self.start + self.duration_1)).astype(float) \
               * self.magnitude_1 + ((t > self.start + self.duration_1) & \
               (t < self.start + self.duration_1 + self.duration_2)).astype(float) \
               * self.magnitude_2


    def relevant_times(self, time_points=1000):
        """Get an set of time points containing all the activity.

        Parameters
        ----------
        time_points : int, optional (1000)
            Number of time points

        Returns
        -------
        np.ndarray
            The time points covering all protocol activity
        """
        return np.linspace(self.start - 0.1*self.duration_1,
                           self.start + 1.5*(self.duration_1 + self.duration_2),
                           1000)

    def to_myokit(self):
        """Convert the protocol to Myokit format.

        Returns
        -------
        myokit.Protocol
            The protocol as a Myokit object
        """
        p = myokit.Protocol()
        p.schedule(self.magnitude_1, self.start, self.duration_1)
        p.schedule(self.magnitude_2, self.start+self.duration_1, self.duration_2)
        return p


class SineWaveProtocol(Protocol):
    """Signal given by a sine wave oscillation.

    y(t) = A sin(omega * t + phi)
    A = amplitude
    omega = frequency
    phi = phase
    """
    def __init__(self, amplitude, frequency, phase, start=0, duration=None):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.start = start
        if duration is None:
            self.duration = 2 * np.pi / self.frequency
        else:
            self.duration = duration


    def value(self, t):
        """Calculate the stimulus signal over time.

        Parameters
        ----------
        t : np.ndarray
            The times at which to evaluate

        Returns
        -------
        np.ndarray
            The values of the stimulus signal at the given times
        """
        return (self.amplitude * np.sin(self.frequency * t + self.phase)) \
               * (self.start <= t).astype(float) \
               * (t<=self.start + self.duration).astype(float)


    def relevant_times(self, time_points=1000):
        """Get an set of time points containing all the activity.

        Parameters
        ----------
        time_points : int, optional (1000)
            Number of time points

        Returns
        -------
        np.ndarray
            The time points covering all protocol activity
        """
        return np.linspace(0, self.duration*1.1, time_points)


    def to_myokit(self):
        """Convert the protocol to Myokit format.

        Returns
        -------
        myokit.Protocol
            The protocol as a Myokit object
        """
        p = myokit.Protocol()
        times = self.relevant_times()
        delta_t = times[1] - times[0]
        values = self.value(t)
        for t, v in zip(times, values):
            p.schedule(v, t, delta_t)
        return p


class PointwiseProtocol(Protocol):
    """A protocol given by arbitrary values at specified times.

    This protocol allows the user to load any precalculated shape, either from
    file or from python lists. It can also be used to hold protocols whose
    shapes do not conform to standard step or sine wave patterns.
    """
    def __init__(self, times=None, values=None, filename=None):
        """Initialize either from lists or from a file.

        There are two choices to initialize an object of this class. Either
        times and values can be specified as python lists, or a filename is
        provided and times and values are read from the file. The file must be
        a csv file with two columns, one with title 'times' and one with title
        'values'.

        Parameters
        ----------
        times : list of float, optional (None)
            Time points where the protocol is specified
        values : list of float, optional (None)
            Values of the protocol at the given times
        filename : str, optional (None)
            Path to the file containing the saved protocol
        """
        if (times is not None and values is not None) and filename is not None:
            raise ValueError('cannot load from both lists and file')

        elif filename is not None:
            data = pandas.read_csv(filename)
            data = data.sort_values(by=['times'])
            self.times = data['times']
            self.values = data['values']

        else:
            self.times, self.values = zip(*sorted(zip(times, values)))

        self.times = np.array(self.times)
        self.values = np.array(self.values)


    def value(self, t=None):
        """Calculate the stimulus signal over time.

        Linear interpolation is used for any time points falling in between
        those that are specified. Outside the range of specified time points,
        the signal is assumed to be 0.

        Parameters
        ----------
        t : np.ndarray, optional (None)
            The times at which to evaluate. If no times are supplied, the
            original grid of times from initialization is automatically used.

        Returns
        -------
        np.ndarray
            The values of the stimulus signal at the given times
        """
        if t is None:
            return self.values
        else:
            return scipy.interpolate.interp1d(self.times, self.values, fill_value=0.0, bounds_error=False)(t)


    def relevant_times(self):
        return self.times

def TimeSeriesFromSteps(start_times_list, duration_list, amplitude_list, baseline=-80):
    """
    Returns a time series of a protocol defined by steps on top of each other.
    :param start_times_list: list or numpy.array
    List of times of start of each step.
    :param duration_list: list or numpy.array
    List of durations of each step
    :param amplitude_list: list or numpy.array
    List of amplitudes of each step
    :param baseline: float
    Defines the baseline of the protocol, to which the steps are added. If not specified, the default value is -80.
    :return: time_series: array
    Time series of the model parameter clamped during the protocol.
    """

    times = np.array([0, start_times_list[0], start_times_list[0] + duration_list[0]])
    values = np.array([baseline, baseline + amplitude_list[0], baseline])

    for i in range(1, len(start_times_list)):
        index_start = np.max(np.where(times <= start_times_list[i]))
        index_end = np.max(np.where(times <= start_times_list[i] + duration_list[i]))
        if times[index_start] == start_times_list[i]:
            times = np.insert(times, index_end + 1, start_times_list[i] + duration_list[i])
            values = np.insert(values, index_end + 1, values[index_end])
            values[index_start:index_end + 1] += amplitude_list[i]

        elif times[index_end] == start_times_list[i] + duration_list[i]:
            times = np.insert(times, index_start + 1, start_times_list[i])
            values = np.insert(values, index_start + 1, values[index_start])
            values[index_start + 1:index_end] += amplitude_list[i]

        else:
            times = np.insert(times, index_start + 1, start_times_list[i])
            times = np.insert(times, index_end + 2, start_times_list[i] + duration_list[i])
            values = np.insert(values, index_start + 1, values[index_start])
            values = np.insert(values, index_end + 2, values[index_end + 1])
            values[index_start + 1:index_end + 2] += amplitude_list[i]

    return np.vstack((times, values))

  
def MyokitProtocolFromTimeSeries(durations, amplitudes):
    """
    Translates a time series of events to a Myokit Protocol.

    :param durations:
    numpy.array .Array of shape(1, number of steps). Contains the durations of all steps of the protocol

    :param amplitudes:
    numpy.array. Array of shape(1, number of steps). Contains the amplitudes of all steps of the protocol

    :return: prot: myokit.protocol
    """

    prot = myokit.Protocol()
    starting_time = 0
    for i in range(len(durations)):
        prot.schedule(amplitudes[i], starting_time, durations[i])
        starting_time += durations[i]

    if starting_time < sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time and \
            sabs_pkpd.constants.protocol_optimisation_instructions != []:
        prot.schedule(amplitudes[-1], starting_time, sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time - starting_time + 100)

    return prot


def TimeSeriesFromSteps(start_times_list, duration_list, amplitude_list, baseline=-80):
    """
    Returns a time series of a protocol defined by steps on top of each other.
    :param start_times_list: list or numpy.array
    List of times of start of each step.
    :param duration_list: list or numpy.array
    List of durations of each step
    :param amplitude_list: list or numpy.array
    List of amplitudes of each step
    :param baseline: float
    Defines the baseline of the protocol, to which the steps are added. If not specified, the default value is -80.
    :return: time_series: array
    Time series of the model parameter clamped during the protocol.
    """

    times = np.array([0, start_times_list[0], start_times_list[0] + duration_list[0]])
    values = np.array([baseline, baseline + amplitude_list[0], baseline])

    for i in range(1, len(start_times_list)):
        index_start = np.max(np.where(times <= start_times_list[i]))
        index_end = np.max(np.where(times <= start_times_list[i] + duration_list[i]))
        if times[index_start] == start_times_list[i]:
            times = np.insert(times, index_end + 1, start_times_list[i] + duration_list[i])
            values = np.insert(values, index_end + 1, values[index_end])
            values[index_start:index_end + 1] += amplitude_list[i]

        elif times[index_end] == start_times_list[i] + duration_list[i]:
            times = np.insert(times, index_start + 1, start_times_list[i])
            values = np.insert(values, index_start + 1, values[index_start])
            values[index_start + 1:index_end] += amplitude_list[i]

        else:
            times = np.insert(times, index_start + 1, start_times_list[i])
            times = np.insert(times, index_end + 2, start_times_list[i] + duration_list[i])
            values = np.insert(values, index_start + 1, values[index_start])
            values = np.insert(values, index_end + 2, values[index_end + 1])
            values[index_start + 1:index_end + 2] += amplitude_list[i]

    return np.vstack((times, values))

  
def MyokitProtocolFromTimeSeries(durations, amplitudes):
    """
    Translates a time series of events to a Myokit Protocol.

    :param durations:
    numpy.array .Array of shape(1, number of steps). Contains the durations of all steps of the protocol

    :param amplitudes:
    numpy.array. Array of shape(1, number of steps). Contains the amplitudes of all steps of the protocol

    :return: prot: myokit.protocol
    """

    prot = myokit.Protocol()
    starting_time = 0
    for i in range(len(durations)):
        prot.schedule(amplitudes[i], starting_time, durations[i])
        starting_time += durations[i]

    if starting_time < sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time and \
            sabs_pkpd.constants.protocol_optimisation_instructions != []:
        prot.schedule(amplitudes[-1], starting_time, sabs_pkpd.constants.protocol_optimisation_instructions.simulation_time - starting_time + 100)

    return prot


def MyokitProtocolFromFourier(real_part, imag_part, low_freq, high_freq):
    """
    This function provides the Myokit Protocol for a Fourier spectrum defined by its real and imaginary parts. Please
    note that the returned protocol starts at time t=0. It may lead to issues if the Myokit Simulation time is not reset.

    :param real_part:
    1D-list or 1D-numpy.array. Contains the real part of the Fourier spectrum.

    :param imag_part:
    1D-list or 1D-numpy.array. Contains the imaginary part of the Fourier spectrum.

    :param low_freq:
    float. Defines the lowest frequency of the Fourier transform

    :param high_freq:
    float. Defines the highest frequency of the Fourier transform

    :return: prot
    Myokit.Protocol. Myoktit Protocol corresponding to the provided Fourier spectrum.
    """

    fourier_spectrum, frequencies = sabs_pkpd.fourier.fourier_spectrum_from_parameters(real_part, imag_part, low_freq, high_freq)
    values, times = sabs_pkpd.fourier.time_series_from_fourier_spectrum(fourier_spectrum, frequencies)
    durations = np.ones(len(values))*(times[1]-times[0])
    prot = MyokitProtocolFromTimeSeries(durations, values)

    return prot