
import os
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
    Loops through the genre directory, extracts the signal and sample
    rate from the audio files, calculates the MFCC features and calls
    the build_features() to create a dataframe. 
    """
    frequency = []

    for genre in os.listdir(path):
        genre_path = path + '/' + genre + '/'
        curr_frequency = []
        for audio_file in os.listdir(genre_path):
            signal, sample_rate = lib.load(genre_path + audio_file)
            mfccs = lib.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=1) # PUT 20
            curr_frequency.append(np.mean(mfccs.T, axis=0))
        frequency.append(curr_frequency)

    build_features(frequency)


def build_features(frequency: list[list[list[float]]]) -> None:
    """
    Passes in the frequency, and creates a dataframe of the MFCC feature.
    Will need to add the rest of the 19 MFCC features, too much data to loop 
    """
    global music_df
    data_frame = []
    label = ["blues", "classical", "country", "disco",
             "hiphop", "jazz", "metal", "pop", "reggae", 
             "rock"]
    
    for idx, freq in enumerate(frequency):
        for mfcc in freq:
            for val in mfcc:
                data_frame.append([label[idx], val])
    
    music_df = pd.DataFrame(data_frame, columns=["Genre-Names", "MFCC-1"])


def calculate_upload(file_path: str) -> list[float]:
    signal, sample_rate = lib.load("audio_music/uploads/" + file_path)
    mfccs = lib.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=1) # PUT 20
    return np.mean(mfccs.T, axis=0)
    
    
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
    ["classical"]
    features_train, features_test, label_train, label_test =
                    train_test_split(features, label, test_size=0.2)
    """
    
    model = DecisionTreeClassifier()
    model.fit(features, label)
    
    data = np.reshape(data, (-1, 1))
    prediction = model.predict(data)
    return prediction[-1]


def main():
    extract_features()

# Creates an instance of the flask web application
app = Flask(__name__)


@app.route("/")
def index():
    """
    When the website is loaded, it renders the HTML home page.
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
    frequency = calculate_upload(file_name)
    genre = train_model(frequency)
    # os.remove(file_name) # remove file from server after getting genre
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
    Passes in the invalid path and redirects users back to the home page
    if they type a page in the search bar that doesn't exist in the domain.
    """
    return redirect(url_for("index"))


if (__name__ == "__main__"):
    main()
    app.run(debug=True)