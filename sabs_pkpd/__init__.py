import os
from flask import Flask
import pints
import myokit
import matplotlib.pyplot as plt
import numpy as np

def create_app(test_config=None):
    """Factory for creating and configuring the web app.
    """
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )

    if test_config is not None:
        app.config.from_mapping(test_config)

    UPLOAD_FOLDER = 'test_upload'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    try:
        os.makedirs(app.config['UPLOAD_FOLDER'])
    except OSError:
        pass

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Make a test page. This is a simple case so we can check it is working
    @app.route('/hello')
    def hello():
        return 'Page returned successfully'


    #register the main blueprint
    from . import pkpd
    app.register_blueprint(pkpd.bp)

    return app

from . import load_data
from . import run_model
from . import pints_problem_def
from . import constants