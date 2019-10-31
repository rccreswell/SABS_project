from sabs_pkpd import create_app

import io
import pytest
from flask import g, session

def test_settings_page(client, app):
    """Test that you get to the initial page.
    """
    assert client.get('/').status_code == 200


def test_model_selection(client, app):
    """Test the selection of a model.
    """
    response = client.post('/', data={'options': 'two'})
    assert 'Model two selected successfully' in str(response.data)


def test_data_upload(client, app):
    """Test the upload of data file.
    """
    # Make a test file
    test_file = (io.BytesIO(b'abc123'), 'x.txt')

    response = client.post('/', data={'data_form': '', 'filename': test_file})

    assert 'Data x.txt uploaded successfully' in str(response.data)


def test_form_submission(client, app):
    """Test the submission of all settings.
    """
    client.post('/', data={'options': 'two'})
    test_file = (io.BytesIO(b'abc123'), 'x.txt')
    client.post('/', data={'data_form': '', 'filename': test_file})
    response = client.post('/', data={'go': 'Go'})

    assert 'Running analysis here' in str(response.data)
