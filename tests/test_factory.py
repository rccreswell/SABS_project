from sabs_pkpd import create_app

def test_config():
    """Test passing config settings to create_app.

    create_app is called twice once with the extra config setting TESTING as
    True
    """
    assert not create_app().testing
    assert create_app({'TESTING': True}).testing


def test_hello(client):
    """Test that the test page (/hello) is returning the correct message.
    """
    response = client.get('/hello')
    assert response.data == b'Page returned successfully'
