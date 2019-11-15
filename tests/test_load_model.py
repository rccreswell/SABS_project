import sabs_pkpd


def test_cellml_load():
    """Test that a CellML model can be loaded.
    """
    cellml_file = './tests/test resources/beeler_reuter_1977.cellml'
    s = sabs_pkpd.load_model.load_model_from_cellml(cellml_file)
    assert True

'''
def test_cellml_protocol_convert():
    """Test that the stimulus in a CellML file is converted to Myokit pace.

    This test is based on the Myokit user guide on CellML imports [1]_. It
    requires that the simulation output meets some conditions which fail before
    the CellML stimulus protocol is correctly converted to Myokit pace.

    References
    ----------
    .. [1] CellML. Myokit API and User Guide.
       <<https://myokit.readthedocs.io/api_formats/cdellml.html>>
    """
    cellml_file = './tests/test resources/beeler_reuter_1977.cellml'
    s = sabs_pkpd.load_model.load_model_from_cellml(cellml_file)
    sim_result = s.run(1000)
    result = sim_result['membrane.V']
    assert result[0] < -80 and max(result) > 20 and result[-1] < -80
'''