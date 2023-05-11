
import os
from flask import Flask, render_template, url_for, request, redirect, send_from_directory

# Model work

# Basic GUI - can worry about this later
app = Flask(__name__)

@app.route("/")
def index():
    """
    Renders the HTML page.
    """
    return render_template("index.html")

if (__name__ == "__main__"):
    app.run()