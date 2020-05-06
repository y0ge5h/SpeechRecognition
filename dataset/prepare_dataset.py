import librosa
import os
import json

DATASET_PATH = '/Users/Yogesh/Desktop/speech_commands'
JSON_PATH = 'data.json'
SAMPLES_TO_CONSIDER = 22050  # 1 sec worth of sound


def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    # data dict
    data = {
        "mapping": [],
        "label": [],  # target outputs
        "mffc": [],
        "file": []
    }

    # loop through all the subdirs
    for i, (dirpath, dirname, filesnames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:
            # update mappings in data dictionary
            category = dirpath.split("/")[-1]
            data['mapping'].append(category)
            print(f"processing {category}")
            # loop the file names and extract mfccs
            for filename in filesnames:

                # construct file path
                filepath = os.path.join(dirpath, filename)

                # load audio file
                signal, sample_rate = librosa.load(filepath)

                # ensure audio is at least 1 sec
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    # enforce 1 sec long signal
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract mfccs
                    mfccs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

                    # store into the data - label encoding categories
                    data['label'].append(i-1)
                    data['mffc'].append(mfccs.T.tolist())
                    data['file'].append(filepath)
                    print(f"{filepath}:{i-1}")

    # store in json file
    with open(json_path, 'w') as jf:
        json.dump(data, jf, indent=4)


if __name__ == '__main__':
    prepare_dataset(DATASET_PATH, JSON_PATH)
