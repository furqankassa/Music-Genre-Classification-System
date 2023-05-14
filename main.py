
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


def build_features(frequency: list[list[list[float]]]) -> pd.DataFrame:
    """
    Passes in the frequency, and creates a dataframe of the MFCC feature.
    Will need to add the rest of the 19 MFCC features, too much data to loop 
    """
    data_frame = []
    label = ["blues", "classical", "country", "disco",
             "hiphop", "jazz", "metal", "pop", "reggae", 
             "rock"]
    
    for idx, freq in enumerate(frequency):
        for mfcc in freq:
            for val in mfcc:
                data_frame.append([label[idx], val])
    
    df = pd.DataFrame(data_frame, columns=["Genre-Names", "MFCC-1"])    

    train_model(df)


def train_model(data: pd.DataFrame) -> float:
    """
    Only because we have 1 feature rn, TESTING purporses.
    Training the model, and making predictions on the genres.
    Current accuracy score: 19% (need more features)
    """
    features = data.loc[:, "MFCC-1"]
    features = features.values.reshape(-1, 1) 
    label = data["Genre-Names"]

    features_train, features_test, label_train, label_test = \
                    train_test_split(features, label, test_size=0.2)
    
    model = DecisionTreeClassifier()
    model.fit(features_train, label_train)

    prediction = model.predict(features_test)
    return accuracy_score(label_test, prediction)

def main():
    extract_features()

# Basic GUI - can worry about this later
app = Flask(__name__)

@app.route("/")
def index():
    """
    Renders the HTML page.
    """
    return render_template("index.html")


if (__name__ == "__main__"):
    # app.run(debug=True)
    main()