
import os
import pandas as pd
import librosa as lib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, url_for, request, redirect


# Global access to the model.
model = None


def extract_features() -> None:
    """
    Loops through the genre directory and extracts the MFCC features
    from the audio files and builds a dataframe out of those features. 
    """
    path = "audio_music/genres"
    features = []
    label = ["blues", "classical", "country", "disco",
             "hiphop", "jazz", "metal", "pop", "reggae",
             "rock"]

    for idx, genre in enumerate(os.listdir(path)):
        genre_path = os.path.join(path, genre)
        for audio_file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, audio_file)
            print(file_path)
            extracted_feat = build_features(file_path)
            curr_mfcc = [mfcc for mfcc in extracted_feat[-1]]
            features.append([label[idx]] + extracted_feat[:-1] + curr_mfcc)

    columns = ["Genre", "Centroid", "Bandwith", "Tonnetz", "Chroma", "RMS", 
               "Flat", "Contrast", "Zero-Crossing-Rate", "Roll-Off"] + \
              [f"MFCC-{i + 1}" for i in range(20)]
    df = pd.DataFrame(features, columns=columns)
    df.to_csv("audio_features.csv", index=False)


def build_features(file_path: str) -> list[float]:
    """
    Loads the audio file and retrieves the signal and sample rate
    to calculate the MFCC, centroid, bandwidth, and chroma features
    of each audio file. Returns a list of the mean values of the features.
    """
    signal, sample_rate = lib.load(file_path)
    centroid = lib.feature.spectral_centroid(y=signal, sr=sample_rate)
    bandwith = lib.feature.spectral_bandwidth(y=signal, sr=sample_rate)
    tonnetz = lib.feature.tonnetz(y=lib.effects.harmonic(signal), 
                                  sr=sample_rate)
    chroma = lib.feature.chroma_cens(y=signal, sr=sample_rate)
    rms = lib.feature.rms(y=signal)
    flat = lib.feature.spectral_flatness(y=signal)
    contrast = lib.feature.spectral_contrast(S=np.abs(lib.stft(signal)), 
                                             sr=sample_rate)
    zcr = lib.feature.zero_crossing_rate(y=signal)
    roll_off = flat = lib.feature.spectral_rolloff(y=signal, sr=sample_rate)
    mfccs = lib.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=20)
    return [np.mean(centroid), np.mean(bandwith), np.mean(tonnetz), 
            np.mean(chroma), np.mean(rms), np.mean(contrast), np.mean(flat),
            np.mean(zcr), np.mean(roll_off), np.mean(mfccs.T, axis=0)]


def train_model(data: pd.DataFrame) -> float:
    """
    Passes in the datframe and trains the model to predict
    the genre by utilizing the extracted features. Split the
    data for testing/training and returns an accuracy score.
    """
    global model
    features = data.loc[:, "Centroid":"MFCC-20"]
    label = data["Genre"]

    features_train, features_test, label_train, label_test = \
                    train_test_split(features, label, test_size=0.2)
    
    model = RandomForestClassifier(n_estimators=144, random_state=56, max_depth=15)
    model.fit(features_train, label_train)

    prediction = model.predict(features_test)
    return accuracy_score(label_test, prediction)


def select_file(file_path: str) -> tuple[str, float]:
    """
    Extracts the features from the audio file uploaded by the user
    and uses the trained model to make a predicted on the extracted
    features.
    """
    global model
    features = build_features(file_path)
    curr_mfcc = [mfcc for mfcc in features[-1]]
    accuracy = train_model(pd.read_csv("audio_features.csv"))
    prediction = model.predict([features[:-1] + curr_mfcc])[0]
    os.remove(file_path)
    return prediction, accuracy



def main():
    """
    Calls extract_features() for feature engineering.
    """
    if (not os.path.exists("audio_features.csv")):
        extract_features()
    train_model(pd.read_csv("audio_features.csv"))
    app.run(debug=True)


# Creates an instance of the flask web application.
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
    genre, accuracy = select_file("audio_music/uploads/" + file_name)
    return render_template("results.html", genre=genre, accuracy=f"{accuracy * 100:.2f}%")


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
    If user tries to access the "results" page without uploading a 
    ile or if the user access non-existent page, it'll return the
    user back to the home page.
    """
    return redirect(url_for("index"))


if (__name__ == "__main__"):
    main()