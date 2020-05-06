import requests


URL = 'http://localhost:5050/predict'
TEST_AUDIO_FILE_PATH = '/Users/Yogesh/Downloads/speech_commands_v0.01/bed/0b77ee66_nohash_1.wav'

if __name__ == '__main__':
    audio_file = open(TEST_AUDIO_FILE_PATH, 'rb')
    values = {
        'file': (TEST_AUDIO_FILE_PATH, audio_file, 'audio/wav'),
    }
    response = requests.post(URL, files=values)
    data = response.json()
    print(f"predicted keyword {data['keyword']}")
