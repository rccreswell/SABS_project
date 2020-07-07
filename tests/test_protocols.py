import numpy as np
import sabs_pkpd


def test_one_step_protocol():
    amplitude = 1.0
    duration = 1.5
    p = sabs_pkpd.protocols.one_step_protocol(amplitude, duration)

    test_times = np.linspace(0, 10, 1000)
    assert(max(p(test_times)) == 1.0)
    assert(min(p(test_times)) == 0.0)


def test_sine_wave_protocol():
    amplitude = 1.5
    frequency = 1.0
    p = sabs_pkpd.protocols.sine_wave_protocol(amplitude, frequency)

    p_evaluated = p(np.linspace(0, 10, 10000))

    assert np.allclose(max(p_evaluated), 1.5)
    assert np.allclose(min(p_evaluated), -1.5)


def test_pointwise_protocol():
    times = [0.0, 1.0, 2.0, 2.5, 2.75, 3.0]
    points = [-1.0, -0.5, 5.0, 6.0, 5.0, 6.0]
    p = sabs_pkpd.protocols.pointwise_protocol(times, points)

    assert p(1) == -0.5
    assert -0.5 <= p(1.5) <= 5
    assert p(2) == 5
    assert p(2.5) == 6


def test_TimeSeriesFromSteps():
    start_times_list = [1.0, 2.0, 3.0, 4.0, 1.5]
    duration_list = [0.1, 0.5, 1.0, 1.0, 0.5]
    amplitude_list = [10.0, 20.0, -10.0, -50.0, 0.0]

    p = sabs_pkpd.protocols.TimeSeriesFromSteps(start_times_list,
                                                duration_list,
                                                amplitude_list)

    times = p[0, :]
    values = p[1, :]

    assert min(values) == -130
    assert max(values) == -60
    assert(2.5 in times)  # Check that the step starting at 2.0 and lasting 0.5


def test_MyokitProtocolFromTimeSeries():
    sabs_pkpd.constants.protocol_optimisation_instructions = \
        sabs_pkpd.constants.Protocol_optimisation_instructions(
            ['model1'], ['clamp variable annot'], ['pace'], 1500, 'readout')

    durations = [50, 20, 30, 400, 50, 200, 300, 10, 200, 40]
    amplitudes = [-85, -50, -55, 20, -30, 20, 0, -40, 10, -75]
    prot = sabs_pkpd.protocols.MyokitProtocolFromTimeSeries(durations,
                                                            amplitudes)

    assert np.array_equal(prot.log_for_times([0, 50, 70, 100, 500, 550, 750,
                                              1050, 1060, 1260, 1300])['pace'],
                          [-85.0, -50.0, -55.0, 20.0, -30.0, 20.0, 0.0, -40.0,
                           10.0, -75.0, -75.0])


def test_MyokitProtocolFromFourier():
    low_freq = 0
    high_freq = 100
    w = np.linspace(low_freq, high_freq, 11)
    real_part = np.cos(2*np.pi*w/25)
    imag_part = np.sin(2*np.pi*w/12.5)
    prot = sabs_pkpd.protocols.MyokitProtocolFromFourier(real_part,
                                                         imag_part,
                                                         low_freq,
                                                         high_freq)
    values = prot.log_for_times(np.linspace(0, 11, 12))['pace']

    expected_values = [0.09090909090909106, 0.14138365080049745,
                       0.5523186154284712, 0.01675948404164788,
                       0.36157679274667687,0.22300143413482795,
                       0.08807202790591157, 0.5247644916149904,
                       0.30631082576308805, 0.3263374809772719,
                       0.049851011157463695, 0.049851011157463695]
    assert(np.allclose(values, expected_values, rtol=1e-3, atol=1e-6))
