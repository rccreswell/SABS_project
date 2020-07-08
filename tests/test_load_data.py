import sabs_pkpd
import io
import pytest
import numpy as np


def test_load_data_file():
    """Test the CSV file is correctly loaded"""

    a = sabs_pkpd.load_data.load_data_file(
        './tests/test resources/load_data_test.csv')

    assert np.array_equal(a.times[0],
                          np.transpose(np.array([0, 0.01, 0.05, 0.1, 0.3, 0.5,
                                                 1, 5])))

    assert np.array_equal(a.values[0],
                          np.transpose(np.array([0, 0.0198, 0.094, 0.1772,
                                                 0.4263, 0.5851, 0.7913,
                                                 0.9967])))
