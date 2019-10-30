"""Main page for running analysis.
"""

import os
from flask import Blueprint, render_template, request, current_app, flash, redirect
from werkzeug.utils import secure_filename

bp = Blueprint('pkpd', __name__)

user_config_settings = {}

@bp.route('/')
def index():
    return render_template('pkpd/index.html', settings=user_config_settings)

@bp.route('/', methods=['GET', 'POST'])
def get_user_settings():
    if request.method == 'POST':
        if 'options' in request.form:
            option = request.form['options']
            user_config_settings['model_version'] = option
            return render_template('pkpd/index.html', settings=user_config_settings)

        if 'file' in request.files:
            file = request.files['file']

            filename = secure_filename(file.filename)
            file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
            #return redirect(url_for('uploaded_file', filename=filename))

            user_config_settings['data_file'] = filename

            return render_template('pkpd/index.html', settings=user_config_settings)

        if 'go' in request.form:
            print('Running Python analysis here')
            print('My settings are:')
            print(user_config_settings)
            print('My data is located at:')
            print(os.path.join(current_app.config['UPLOAD_FOLDER'], user_config_settings['data_file']))


    return render_template('pkpd/index.html', settings=user_config_settings)
