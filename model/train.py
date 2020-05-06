import json
import numpy as np
from sklearn.model_selection import train_test_split as tts
import tensorflow.keras as keras

DATA_PATH = 'dataset/data.json'
LEARNING_RATE = 0.0001
EPOCHS = 10
BATCH_SIZE = 32
SAVED_MODEL_PATH = 'model/model.h5'
NUM_KEYWORDS = 25


def load_dataset(data_path):

    # loading
    with open(data_path,'r') as jf:
        data = json.load(jf)

    # inputs and targets
    X = np.array(data['mffc'])
    y = np.array(data['label'])

    # return
    return X, y


def get_data_splits(data_path, test_size=0.1, validation_size=0.1):
    # load dataset
    X, y = load_dataset(data_path)

    # create train , test , val splits
    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = tts(X_train, y_train, test_size=validation_size)

    # convert inputs from 2d to 3d arrays
    # (# segments, mfcc 13, 1)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_shape, learning_rate, error='sparse_categorical_crossentropy'):

    # build network
    model = keras.Sequential()

    # conv layer 1
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001),
                                  input_shape=input_shape))

    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))

    # conv layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))

    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))

    # conv layer 3
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))

    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))

    # flatten output as input to dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # softmax classifier
    model.add(keras.layers.Dense(NUM_KEYWORDS, activation='softmax'))

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=error, metrics=['accuracy'])

    # model overview
    model.summary()

    return model


def main():
    # load train validation test data
    X_train,  X_val, X_test, y_train, y_val, y_test = get_data_splits(DATA_PATH)

    # build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # (# of segments, # of coefficients 13, 1)
    model = build_model(input_shape, LEARNING_RATE)

    # train model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

    # evaluate the model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"test error {test_error} , test accuracy {test_accuracy}")

    # save the model
    model.save(SAVED_MODEL_PATH)


if __name__=='__main__':
    main()