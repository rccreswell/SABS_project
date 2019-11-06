import sabs_pkpd

import io
import pytest

def test_load_data_file():
    """ Test the CSV file is correctly loaded    """
    a = sabs_pkpd.load_data.load_data_file('./tests/test resources/load_data_test.csv')
    test = ( a.times == [0, 5, 15, 35, 60, 100, 0, 5, 15, 35, 60] )
    assert test.all() == True

    test = ( a.values == np.transpose([0, 0.126, 0.333, 0.612, 0.802, 0.933, 0, 0.221, 0.528, 0.826, 0.950]))
    assert test.all() == True

