import tensorflow.keras as keras
import librosa
import numpy as np


MODEL_PATH = 'model/model.h5'
NUM_OF_SAMPLES_TO_CONSIDER = 22050 # worth 1 sec of sound


class _KeywordSpottingService:

    model = None
    _mapping = ["right", "eight", "bed", "happy", "go", "no", "wow", "nine", "left", "stop", "three", "sheila",
                "one", "zero", "seven", "up", "two", "house", "down", "six", "yes", "on", "five", "off", "four"]
    instance = None

    def predict(self, file_path):
        # extract mfccs
        mfccs = self.preprocess(file_path) # shape is (# segments, #coefficients)

        # convert 2d array into 4d array > (# sample, # segments, #coefficients, #channels=1)
        mfccs = mfccs[np.newaxis, ..., np.newaxis]

        # make predictions
        predictions = self.model.predict(mfccs) # output [ [0.1, 0.6, 0.1,....] ]

        # prediction index
        prediction_index = np.argmax(predictions)

        predicted_keyword = self._mapping[prediction_index]

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, hop_length=512, n_fft=2048):

        # load the audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency
        if len(signal) > NUM_OF_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_OF_SAMPLES_TO_CONSIDER]

        # extract mfccs
        mfccs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

        return mfccs.T


def keyword_spotting_service():
    # insuring we only have 1 instance
    if _KeywordSpottingService.instance is None:
        _KeywordSpottingService.instance = _KeywordSpottingService()
        _KeywordSpottingService.model = keras.models.load_model(MODEL_PATH)
    return _KeywordSpottingService.instance


if __name__ == '__main__':
    kss = keyword_spotting_service()
    keyword = kss.predict('/Users/Yogesh/Downloads/speech_commands_v0.01/right/0a7c2a8d_nohash_0.wav')
    print(f"predicted keyword {keyword}")
