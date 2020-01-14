import myokit


def test_myokit_import():
    """Test that Myokit can be loaded.

    Uses the Myokit builtin example.
    """
    m, p, x = myokit.load('example')
    s = myokit.Simulation(m, p)
    assert True
