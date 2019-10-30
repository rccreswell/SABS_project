from flask import Flask, request, render_template
from flaskwebgui import FlaskUI #get the FlaskUI class

app = Flask(__name__)

ui = FlaskUI(app)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    processed_text = text.upper()
    return processed_text

ui.run()
