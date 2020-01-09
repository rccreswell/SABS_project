import numpy as np


def test_Fourier_to_protocol():
    low_freq = 0
    high_freq = 100
    w = np.linspace(low_freq, high_freq, 11)
    real_part = np.cos(2*np.pi*w/25)
    imag_part = np.sin(2*np.pi*w/12.5)
    fourier_spectrum, frequencies = fourier_spectrum_from_parameters(real_part, imag_part, low_freq, high_freq)

    expected_fourier_spectrum = np.array([ 1 + 0.00000000e+00j, -0.80901699 - 9.51056516e-01j, 0.30901699 - 5.87785252e-01j,
                                        0.30901699 + 5.87785252e-01j, -0.80901699 + 9.51056516e-01j,  1 - 9.79717439e-16j,
                                        -0.80901699 - 9.51056516e-01j,  0.30901699 - 5.87785252e-01j,
                                        0.30901699 + 5.87785252e-01j, -0.80901699 + 9.51056516e-01j, 1 - 1.95943488e-15j])
    assert(np.allclose(fourier_spectrum, expected_fourier_spectrum, rtol=1.e-3))

    values,times = time_series_from_fourier_spectrum(fourier_spectrum, frequencies)

    expected_values = [0.09090909090909106, 0.14138365080049745, 0.5523186154284712, 0.01675948404164788, 0.36157679274667687,
                       0.22300143413482795, 0.08807202790591157, 0.5247644916149904, 0.30631082576308805, 0.3263374809772719,
                       0.049851011157463695]
    assert(np.allclose(values, expected_values, rtol=1.e-3))

