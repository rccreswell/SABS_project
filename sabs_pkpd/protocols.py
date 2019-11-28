"""Classes for parametric forms of protocols.

May be useful for protocol optimization.
"""

import pandas
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import myokit

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
            sel.duration = duration


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


if __name__ == '__main__':
    p = SineWaveProtocol(2.5, 10, 0)
    p.plot()
    exit()

    t = [1,2,4,3]
    v = [0,2,0,2]
    p = PointwiseProtocol(times=t, values=v)
    test_times = np.linspace(-1, 5, 1000)
    plt.plot(test_times, p.value(test_times))
    plt.show()

    t = [1,2,4,3]
    v = [0,2,0,2]
    p = PointwiseProtocol(times=t, values=v)
    p.plot()

    p2 = PointwiseProtocol(filename='abc.txt')
    p2.plot()
    exit()

    t = np.linspace(0, 5, 100)
    p = OneStepProtocol(1, 2, 5.6)
    p.to_myokit()

    p = TwoStepProtocol(1, 3, 1.2, 4.5, -0.5)
    p.to_myokit()
    p.plot()