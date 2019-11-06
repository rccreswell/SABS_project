import sabs_pkpd

import io
import pytest

def test_load_data_file():
    """ Test the CSV file is correctly loaded    """
    a = Data_exp()
    load_data_file(a,'./test resources/load_data_test.csv')
    assert a.times == [0, 5, 15, 35, 60, 100, 0, 5, 15, 35, 60]
    assert a.values == [0, 0.12640215, 0.33329353, 0.61168793, 0.80242196, 0.93297588, 0, 0.22119922, 0.52763345, 0.82622606, 0.95021293]
