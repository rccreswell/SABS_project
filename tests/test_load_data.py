import sabs_pkpd

import io
import pytest

def test_load_data_file():
    """ Test the CSV file is correctly loaded    """
    a = sabs_pkpd.load_data.load_data_file('./tests/test resources/load_data_test.csv')
    test = ( a.times == [0, 5, 15, 35, 60, 100, 0, 5, 15, 35, 60] )
    assert test.all() == True

    test = ( a.values == [0, 0.12640215, 0.33329353, 0.61168793, 0.80242196, 0.93297588, 0, 0.22119922, 0.52763345, 0.82622606, 0.95021293] )
    assert test.all() == True
