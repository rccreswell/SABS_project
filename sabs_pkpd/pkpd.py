"""Main page for running analysis.
"""

import os
from flask import Blueprint, render_template, request, current_app
from werkzeug.utils import secure_filename

bp = Blueprint('pkpd', __name__)

@bp.route('/')
def index():
    return render_template('pkpd/index.html', submit=True)


@bp.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        if 'file' not in request.files:
            flash('No file')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('no selected file')
            return redirect(request.url)

        if file and True:
            filename = secure_filename(file.filename)
            file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
            #return redirect(url_for('uploaded_file', filename=filename))
            return render_template('pkpd/index.html', submit=False)

    return render_template('pkpd/index.html', posts=[1,2])
