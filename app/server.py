from flask import Flask,  request, jsonify
import random
import os
import sys
sys.path.append('/Users/Yogesh/PycharmProjects/SpeechRecognition/')
from model.keyword_spotting_service import keyword_spotting_service


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():

    # get audio file and save it
    audio_file = request.files['file']
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)

    # invoke keyword spotting service
    kss = keyword_spotting_service()

    # make a prediction
    predicted_keyword = kss.predict(file_name)

    # remove audio file
    os.remove(file_name)

    # send back the predicted keyword in json
    data = {
        'keyword': predicted_keyword
    }

    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=False)
