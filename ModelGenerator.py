import keras

from keras import models
from keras import layers

from keras.optimizers import RMSprop, SGD, Adagrad, Adadelta, Adamax
from keras.optimizers import Adam

img_rows = 128
img_cols = 128


def new_model_rms():  # Creating new model
    temporary_model = models.Sequential()
    temporary_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Flatten())
    temporary_model.add(layers.Dense(256, activation='relu'))
    temporary_model.add(layers.Dense(10, activation='softmax'))

    temporary_model.summary()

    temporary_model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
        metrics=['acc']
    )

    return temporary_model


def new_model_adam():  # Creating new model
    temporary_model = models.Sequential()
    temporary_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Flatten())
    temporary_model.add(layers.Dense(256, activation='relu'))
    temporary_model.add(layers.Dense(10, activation='softmax'))

    temporary_model.summary()

    temporary_model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=1e-4),
        metrics=['acc']
    )

    return temporary_model


def new_model_sdg():  # Creating new model
    temporary_model = models.Sequential()
    temporary_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Flatten())
    temporary_model.add(layers.Dense(256, activation='relu'))
    temporary_model.add(layers.Dense(10, activation='softmax'))

    temporary_model.summary()

    temporary_model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(learning_rate=0.01, momentum=0.9),
        metrics=['acc']
    )

    return temporary_model


def new_model_adagrad():  # Creating new model
    temporary_model = models.Sequential()
    temporary_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Flatten())
    temporary_model.add(layers.Dense(256, activation='relu'))
    temporary_model.add(layers.Dense(10, activation='softmax'))

    temporary_model.summary()

    temporary_model.compile(
        loss='categorical_crossentropy',
        optimizer=Adagrad(learning_rate=0.01),
        metrics=['acc']
    )

    return temporary_model


def new_model_adadelta():  # Creating new model
    temporary_model = models.Sequential()
    temporary_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Flatten())
    temporary_model.add(layers.Dense(256, activation='relu'))
    temporary_model.add(layers.Dense(10, activation='softmax'))

    temporary_model.summary()

    temporary_model.compile(
        loss='categorical_crossentropy',
        optimizer=Adadelta(learning_rate=1.0),
        metrics=['acc']
    )

    return temporary_model


def new_model_adamax():  # Creating new model
    temporary_model = models.Sequential()
    temporary_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    temporary_model.add(layers.MaxPooling2D((2, 2)))
    temporary_model.add(layers.Flatten())
    temporary_model.add(layers.Dense(256, activation='relu'))
    temporary_model.add(layers.Dense(10, activation='softmax'))

    temporary_model.summary()

    temporary_model.compile(
        loss='categorical_crossentropy',
        optimizer=Adamax(learning_rate=0.002),
        metrics=['acc']
    )

    return temporary_model
