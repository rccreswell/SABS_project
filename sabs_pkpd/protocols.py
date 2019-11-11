"""Classes for parametric forms of protocols.

May be useful for protocol optimization.
"""

import numpy as np
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
    def __init__(self, amplitude, frequency, phase):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase


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
        return self.amplitude * np.sin(self.frequency * t + self.phase)


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
        return np.linspace(0, 2.5 * np.pi / self.frequency)


    def to_myokit(self):
        """Convert the protocol to Myokit format.

        Returns
        -------
        myokit.Protocol
            The protocol as a Myokit object
        """
        raise NotImplementedError


if __name__ == '__main__':
    t = np.linspace(0, 5, 100)
    p = OneStepProtocol(1, 2, 5.6)
    p.to_myokit()

    p = TwoStepProtocol(1, 3, 1.2, 4.5, -0.5)
    p.to_myokit()
    p.plot()

    p = SineWaveProtocol(2.5, 10, 1)
    p.plot()
