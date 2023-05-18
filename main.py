
import os
from typing import Any
import pandas as pd
import librosa as lib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, render_template, url_for, request, redirect

# Notes
"""
For audio 54 in the Jazz genre, the file was corrupted so I deleted it.
For the blues genre, 1-7 files were missing as well as the 100th song.
Still need to add on the other 19 MFCC features.
"""

# relative path to the genre directory
path = "audio_music/genres"
music_df = pd.DataFrame() # global access to the dataframe


# Model work
def extract_features() -> None:
    """
    Loops through the genre directory and extracts the MFCC features
    from the audio files and builds a dataframe out of those features. 
    """
    global music_df
    features = []
    label = ["blues", "classical", "country", "disco",
             "hiphop", "jazz", "metal", "pop", "reggae", 
             "rock"]

    for idx, genre in enumerate(os.listdir(path)):
        genre_path = path + '/' + genre + '/'
        for audio_file in os.listdir(genre_path):
            curr_mfcc = calculate_mfcc(genre_path + audio_file)
            features.append([label[idx], curr_mfcc])
    
    music_df = pd.DataFrame(features, columns=["label", "feature"])


def calculate_mfcc(file_path: str) -> list[float]:
    """
    Loads the audio file and retrieves the signal and sample rate
    to calculate the MFCC feature of each audio file. Returns
    a list of the mean MFCC values acorss the time axis.
    """
    signal, sample_rate = lib.load(file_path)
    mfccs = lib.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=20)
    return np.mean(mfccs.T, axis=0)


def select_file(file_path: str) -> None | str:
    """
    Extracts the features from the audio file uploaded by the user
    and uses the trained model to make a predicted on the extracted
    features.
    """
    if (not file_path):
        return None
    
    # Extract features from the selected file
    # features = extract_features(file_path)
    # label = model.predict([features])[0]

def train_model(data: pd.DataFrame) -> str:
    """
    Only because we have 1 feature rn, TESTING purporses.
    Training the model, and making predictions on the genres.
    Current accuracy score: 19% (need more features)
    """
    global music_df
    features = music_df.loc[:, "MFCC-1"]
    features = features.values.reshape(-1, 1)
    label = music_df["Genre-Names"]

    """
    features_train, features_test, label_train, label_test =
                    train_test_split(features, label, test_size=0.2)
    """
    
    model = DecisionTreeClassifier()
    model.fit(features, label)
    
    data = np.reshape(data, (-1, 1))
    prediction = model.predict(data)
    return prediction[-1]


def main():
    """
    Calls extract_features() for feature engineering.
    """
    extract_features()


# Creates an instance of the flask web application
app = Flask(__name__)


@app.route("/")
def index():
    """
    When the website is loaded, it renders the HTML home page. It represents
    the main entry to our web application.
    """
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    A POST request for when a file is uploaded, and renders the results.html
    page to display the accuracy score and genre classification. If the file
    is invalid, user will receive an error message to resubmit a .wav file.
    """
    f = request.files["file"]
    if (not f.filename.endswith(".wav")):
        return redirect(url_for("display_error",
                                message="Please upload a .wav audio file!"))
    f.save(os.path.join("audio_music/uploads", f.filename))
    return redirect(url_for("calculate", file_name=f.filename))


@app.route("/calculate/<file_name>")
def calculate(file_name: str):
    """
    If the user uploads a valid .wave file, then the features will be extracted
    from the uploaded audio and a prediction will be made. The accuracy score
    will be displayed as well as the name of genre classified by the model.
    """
    frequency = select_file("audio_music/uploads/" + file_name)
    genre = train_model(frequency)
    # os.remove(file_name) remove file from server after getting genre
    return render_template("results.html", outcome=f"Genre: {genre}")


@app.route("/display_error/<message>")
def display_error(message: str):
    """
    Rerenders the home page to display an error if the file upload
    was either not a .wave file or if no file was uploaded at all.
    """
    return render_template("index.html", message=message)


@app.route("/<path:path>")
def redirect_to_home(path: str):
    """
    If user tries to access the "results" page through typing on the domain
    without uploading a file or if the user access non-existent page, it'll
    return the user back to the home page.
    """
    return redirect(url_for("index"))


if (__name__ == "__main__"):
    main()
    app.run(debug=True)