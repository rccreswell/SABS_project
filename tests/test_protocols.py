import math
import os
import numpy as np
import sabs_pkpd

def test_pointwiseprotocol_lists():
    """Test loading the pointwise protocol from lists.
    """
    t = [1,2,4,3]
    v = [0,2,0,2]
    p = sabs_pkpd.protocols.PointwiseProtocol(times=t, values=v)
    test_times = np.linspace(-1, 5, 1000)
    test_stimulus = p.value(test_times)
    assert math.isclose(max(test_stimulus), 2.0)
    assert math.isclose(min(test_stimulus), 0.0)
    assert test_stimulus[0] == 0
    assert test_stimulus[-1] == 0


def test_pointwiseprotocol_file():
    """Test loading the pointwise protocol from a file.
    """
    p = sabs_pkpd.protocols.PointwiseProtocol(filename=os.path.join('tests', 'test resources', 'input_protocol.csv'))
    test_stimulus = p.value()
    assert math.isclose(max(test_stimulus), 1)
    assert math.isclose(min(test_stimulus), 0)
    assert test_stimulus[0] == 0
    assert test_stimulus[-1] == 0


def test_onestepprotocol():
    """Test the shape of the one step protocol.
    """
    t = np.linspace(0, 20, 1000)
    p = sabs_pkpd.protocols.OneStepProtocol(2.5, 0.5, 10.5)
    assert max(p.value(t)) == 10.5
    assert p.value(t)[0] == 0
    assert p.value(t)[-1] == 0


def test_onestepprotocol_myokit():
    """Test the conversion to Myokit format for the one step protocol.
    """
    p = sabs_pkpd.protocols.OneStepProtocol(2.5, 0.5, 10.5)
    myokit_protocol = p.to_myokit()
    assert myokit_protocol.in_words() == \
       'Stimulus of 10.5 times the normal level applied at t=2.5, lasting 0.5.'


def test_twostepprotocol():
    """Test the shape of the two step protocol.
    """
    t = np.linspace(0, 20, 1000)
    p = sabs_pkpd.protocols.TwoStepProtocol(1.0, 3.0, 1.2, 4.5, -0.5)
    assert max(p.value(t)) == 4.5
    assert min(p.value(t)) == -0.5
    assert p.value(t)[0] == 0
    assert p.value(t)[-1] == 0


def test_twostepprotocol_myokit():
    """Test the conversion to Myokit format for the one step protocol.
    """
    p = sabs_pkpd.protocols.TwoStepProtocol(1.7, 3.5, 1.2, 4.5, -0.5)
    myokit_protocol = p.to_myokit()
    assert 'Stimulus of 4.5 times the normal level applied at t=1.7, lasting 3.5' in myokit_protocol.in_words()
    assert 'Stimulus of -0.5 times the normal level applied at t=5.2, lasting 1.2' in myokit_protocol.in_words()


def test_sinewaveprotocol():
    """Test the shape of the sine wave protocol.
    """
    t = np.linspace(0, 20, 1000)
    p = sabs_pkpd.protocols.SineWaveProtocol(1.5, 1.0, 0.5)
    assert math.isclose(max(p.value(t)), 1.5, abs_tol=0.01)
    assert math.isclose(min(p.value(t)), -1.5, abs_tol=0.01)
    t = np.linspace(0, 2*math.pi, 1000)
    values = p.value(t)
    assert math.isclose(values[0], values[-1], abs_tol=0.01)


def test_MyokitProtocolFromTimeSeries():
    sabs_pkpd.constants.protocol_optimisation_instructions = sabs_pkpd.constants.Protocol_optimisation_instructions(
        ['model1'], ['clamp variable annot'], ['pace'], 1500, 'readout')

    durations = [50, 20, 30, 400, 50, 200, 300, 10, 200, 40]
    amplitudes = [-85, -50, -55, 20, -30, 20, 0, -40, 10, -75]
    prot = sabs_pkpd.protocols.MyokitProtocolFromTimeSeries(durations, amplitudes)

    assert np.array_equal(prot.log_for_times([0, 50, 70, 100, 500, 550, 750, 1050, 1060, 1260, 1300])['pace'],
                          [-85.0, -50.0, -55.0, 20.0, -30.0, 20.0, 0.0, -40.0, 10.0, -75.0, -75.0])


def test_Constraints_def():
    def function(x):
        intermediate = np.ones(np.shape(x))
        return np.linalg.norm(intermediate)

    lb1 = 3
    ub1 = 4

    def function2(x):
        return np.sum(x)

    lb2 = 22
    ub2 = 24

    matrix_to_test_true = [[1, 1, 2, 3, 4], [0, 1, 2, 4, 5]]
    matrix_to_test_false = [0]

    con = sabs_pkpd.optimize_protocol_model_distinction.Constraint(function, lb1, ub1)
    verif1 = con.verification(matrix_to_test_true)
    verif3 = con.verification(matrix_to_test_false)

    con = sabs_pkpd.optimize_protocol_model_distinction.Constraint(function2, lb2, ub2)
    verif2 = con.verification(matrix_to_test_true)
    verif4 = con.verification(matrix_to_test_false)

    assert verif1 == True
    assert verif2 == True
    assert verif3 == False
    assert verif4 == False

    return verif1, verif2, verif3, verif4


def test_MyokitProtocolFromFourier():
    low_freq = 0
    high_freq = 100
    w = np.linspace(low_freq, high_freq, 11)
    real_part = np.cos(2*np.pi*w/25)
    imag_part = np.sin(2*np.pi*w/12.5)
    prot = sabs_pkpd.protocols.MyokitProtocolFromFourier(real_part, imag_part, low_freq, high_freq)
    values = prot.log_for_times(np.linspace(0, 11, 12))['pace']

    expected_values = [0.09090909090909106, 0.14138365080049745, 0.5523186154284712, 0.01675948404164788, 0.36157679274667687,
                           0.22300143413482795, 0.08807202790591157, 0.5247644916149904, 0.30631082576308805, 0.3263374809772719,
                           0.049851011157463695, 0.049851011157463695]
    assert(np.allclose(values, expected_values, rtol=1e-3, atol=1e-6))