import sabs_pkpd

import io
import pytest
import numpy as np

def test_load_data_file():
    """ Test the CSV file is correctly loaded    """

    a = sabs_pkpd.load_data.load_data_file('./tests/test resources/load_data_test.csv')

    assert np.array_equal(a.times[0],  np.transpose(np.array([0, 5, 15, 35, 60, 100])))
    assert np.array_equal(a.times[1],  np.transpose(np.array([0, 5, 15, 35, 60])))

    assert np.array_equal(a.values[0], np.transpose(np.array([0, 0.126, 0.333, 0.612, 0.802, 0.933])))
    assert np.array_equal(a.values[1], np.transpose(np.array([0, 0.221, 0.528, 0.826, 0.950])))


