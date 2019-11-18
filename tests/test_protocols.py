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
