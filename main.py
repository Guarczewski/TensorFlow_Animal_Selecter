import tensorflow
import keras
import numpy

import matplotlib.pyplot as pyplot
from keras.callbacks import EarlyStopping
from keras.layers import Dropout

from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import ImageDataGenerator

import SetupModule
import ModelGenerator

learningIteration = 1

img_rows = 128
img_cols = 128

_globalBatchSize = 25
_globalStepsPerEpoch = 100
_globalEpochAmount = 50
_globalValidationSteps = 100

# Check if libraries work
print("TensorFlow version: " + tensorflow.__version__)
print("Keras version: " + keras.__version__)
print("Numpy version: " + numpy.__version__)


SetupModule.check_directories()
SetupModule.separate_files()

all_models = []
all_acc = []
all_loss = []
all_val_acc = []
all_val_loss = []


def plot_accuracy(accuracy, val_accuracy, loss, val_loss, cm, lab='*'):  # Create plot
    epochs = range(len(accuracy))
    local_fig, local_axes = pyplot.subplots(3, 1, figsize=(8, 12))

    local_axes[0].plot(epochs, accuracy, 'b', label='Training accuracy ' + lab)
    local_axes[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy ' + lab)
    local_axes[0].set_title('Comparison of Training')
    local_axes[0].set_xlabel('Epochs')
    local_axes[0].set_ylabel('Accuracy')
    local_axes[0].legend(['Train', 'Test'], loc='upper left')

    local_axes[1].plot(epochs, loss, 'b', label='Training loss for ' + lab)
    local_axes[1].plot(epochs, val_loss, 'b', label='Validation loss for ' + lab)
    local_axes[1].set_title('Comparison of Training and Validation Losses')
    local_axes[1].set_xlabel('Epochs')
    local_axes[1].set_ylabel('Loss')
    local_axes[1].legend(['Train', 'Test'], loc='upper left')

    local_axes[2].imshow(cm, cmap='Blues')
    local_axes[2].set_title("Confusion Matrix")
    local_axes[2].set_xlabel("Predicted Class")
    local_axes[2].set_ylabel("True Class")
    local_axes[2].grid(False)
    for i in range(len(cm)):
        for j in range(len(cm)):
            local_axes[2].text(j, i, cm[i, j], ha='center', va='center', color='black')

    pyplot.tight_layout()
    pyplot.show()


def new_confusion_matrix(model, validation_generator):
    validation_images = []
    validation_labels = []
    validation_generator.reset()

    while len(validation_images) < 1000:
        images, labels = validation_generator.next()
        for image in images:
            validation_images.append(numpy.array(image))
        validation_labels.extend(labels)
        if len(validation_images) >= 1000:
            break

    validation_images = numpy.array(validation_images)
    validation_labels = numpy.array(validation_labels)
    validation_labels = numpy.argmax(validation_labels, axis=1)

    predictions = model.predict(validation_images)
    predicted_labels = numpy.argmax(predictions, axis=1)

    temp_confusion_matrix = confusion_matrix(validation_labels, predicted_labels)

    return temp_confusion_matrix


def model_rms_01():
    ModelRMS_01 = ModelGenerator.new_model_rms()  # Create new RMS Model

    trainDataGenRMS_01 = ImageDataGenerator(rescale=1. / 255)
    testDataGenRMS_01 = ImageDataGenerator(rescale=1. / 255)

    trainGeneratorRMS_01 = trainDataGenRMS_01.flow_from_directory(
        SetupModule.trainDirectory,
        target_size=(img_rows, img_cols),
        batch_size=_globalBatchSize,
        class_mode='categorical'
    )

    validationGeneratorRMS_01 = testDataGenRMS_01.flow_from_directory(
        SetupModule.validationDirectory,
        target_size=(img_rows, img_cols),
        batch_size=_globalBatchSize,
        class_mode='categorical'
    )

    historyRMS_01 = ModelRMS_01.fit(
        trainGeneratorRMS_01,
        steps_per_epoch=_globalStepsPerEpoch,
        epochs=_globalEpochAmount,
        validation_data=validationGeneratorRMS_01,
        validation_steps=_globalValidationSteps
    )

    # Save Output
    ModelRMS_01.save(SetupModule.modelsDirectory + SetupModule.saveFileName.format('RMS_01'))

    accuracyRMS_01 = historyRMS_01.history['acc']
    val_accuracyRMS_01 = historyRMS_01.history['val_acc']
    lossRMS_01 = historyRMS_01.history['loss']
    val_lossRMS_01 = historyRMS_01.history['val_loss']

    confusion_matrixRMS_01 = new_confusion_matrix(ModelRMS_01, validationGeneratorRMS_01)

    plot_accuracy(
        accuracyRMS_01,
        val_accuracyRMS_01,
        lossRMS_01,
        val_lossRMS_01,
        confusion_matrixRMS_01,
        lab='ModelRMS_01'
    )

    # History
    all_models.append('RMS_01')
    all_acc.append(max(historyRMS_01.history['acc']))
    all_loss.append(min(historyRMS_01.history['loss']))
    all_val_acc.append(max(historyRMS_01.history['val_acc']))
    all_val_loss.append(min(historyRMS_01.history['val_loss']))


def model_adam_01():
    ModelADAM_01 = ModelGenerator.new_model_adam()  # Create new ADAM Model

    trainDataGenAdam_01 = ImageDataGenerator(rescale=1. / 255)
    testDataGenAdam_01 = ImageDataGenerator(rescale=1. / 255)

    trainGeneratorADAM_01 = trainDataGenAdam_01.flow_from_directory(
        SetupModule.trainDirectory,
        target_size=(img_rows, img_cols),
        batch_size=_globalBatchSize,
        class_mode='categorical'
    )

    validationGeneratorADAM_01 = testDataGenAdam_01.flow_from_directory(
        SetupModule.validationDirectory,
        target_size=(img_rows, img_cols),
        batch_size=_globalBatchSize,
        class_mode='categorical'
    )

    historyADAM_01 = ModelADAM_01.fit(
        trainGeneratorADAM_01,
        steps_per_epoch=_globalStepsPerEpoch,
        epochs=_globalEpochAmount,
        validation_data=validationGeneratorADAM_01,
        validation_steps=_globalValidationSteps
    )

    # Save Output
    ModelADAM_01.save(SetupModule.modelsDirectory + SetupModule.saveFileName.format('ADAM_01'))

    accuracyADAM_01 = historyADAM_01.history['acc']
    val_accuracyADAM_01 = historyADAM_01.history['val_acc']
    lossADAM_01 = historyADAM_01.history['loss']
    val_lossADAM_01 = historyADAM_01.history['val_loss']

    confusion_matrixADAM_01 = new_confusion_matrix(ModelADAM_01, validationGeneratorADAM_01)

    plot_accuracy(
        accuracyADAM_01,
        val_accuracyADAM_01,
        lossADAM_01,
        val_lossADAM_01,
        confusion_matrixADAM_01,
        lab='ADAM_01'
    )

    # History
    all_models.append('ADAM_01')
    all_acc.append(max(historyADAM_01.history['acc']))
    all_loss.append(min(historyADAM_01.history['loss']))
    all_val_acc.append(max(historyADAM_01.history['val_acc']))
    all_val_loss.append(min(historyADAM_01.history['val_loss']))


def model_sdg_01():
    ModelSDG_01 = ModelGenerator.new_model_sdg()  # Create new SDG Model

    trainDataGenSDG_01 = ImageDataGenerator(rescale=1. / 255)
    testDataGenSDG_01 = ImageDataGenerator(rescale=1. / 255)

    trainGeneratorSDG_01 = trainDataGenSDG_01.flow_from_directory(
        SetupModule.trainDirectory,
        target_size=(img_rows, img_cols),
        batch_size=_globalBatchSize,
        class_mode='categorical'
    )

    validationGeneratorSDG_01 = testDataGenSDG_01.flow_from_directory(
        SetupModule.validationDirectory,
        target_size=(img_rows, img_cols),
        batch_size=_globalBatchSize,
        class_mode='categorical'
    )

    historySDG_01 = ModelSDG_01.fit(
        trainGeneratorSDG_01,
        steps_per_epoch=_globalStepsPerEpoch,
        epochs=_globalEpochAmount,
        validation_data=validationGeneratorSDG_01,
        validation_steps=_globalValidationSteps
    )

    # Save Output
    ModelSDG_01.save(SetupModule.modelsDirectory + SetupModule.saveFileName.format('SDG_01'))

    accuracySDG_01 = historySDG_01.history['acc']
    val_accuracySDG_01 = historySDG_01.history['val_acc']
    lossSDG_01 = historySDG_01.history['loss']
    val_lossSDG_01 = historySDG_01.history['val_loss']

    confusion_matrixSDG_01 = new_confusion_matrix(ModelSDG_01, validationGeneratorSDG_01)

    plot_accuracy(
        accuracySDG_01,
        val_accuracySDG_01,
        lossSDG_01,
        val_lossSDG_01,
        confusion_matrixSDG_01,
        lab='SDG_01'
    )

    # History
    all_models.append('SDG_01')
    all_acc.append(max(historySDG_01.history['acc']))
    all_loss.append(min(historySDG_01.history['loss']))
    all_val_acc.append(max(historySDG_01.history['val_acc']))
    all_val_loss.append(min(historySDG_01.history['val_loss']))


def model_adagrad_01():
    ModelADAGRAD_01 = ModelGenerator.new_model_adagrad()  # Create new ADAGRAD Model

    trainDataGenADAGRAD_01 = ImageDataGenerator(rescale=1. / 255)
    testDataGenADAGRAD_01 = ImageDataGenerator(rescale=1. / 255)

    trainGeneratorADAGRAD_01 = trainDataGenADAGRAD_01.flow_from_directory(
        SetupModule.trainDirectory,
        target_size=(img_rows, img_cols),
        batch_size=_globalBatchSize,
        class_mode='categorical'
    )

    validationGeneratorADAGRAD_01 = testDataGenADAGRAD_01.flow_from_directory(
        SetupModule.validationDirectory,
        target_size=(img_rows, img_cols),
        batch_size=_globalBatchSize,
        class_mode='categorical'
    )

    historyADAGRAD_01 = ModelADAGRAD_01.fit(
        trainGeneratorADAGRAD_01,
        steps_per_epoch=_globalStepsPerEpoch,
        epochs=_globalEpochAmount,
        validation_data=validationGeneratorADAGRAD_01,
        validation_steps=_globalValidationSteps
    )

    # Save Output
    ModelADAGRAD_01.save(SetupModule.modelsDirectory + SetupModule.saveFileName.format("ADAGRAD_01"))

    accuracyADAGRAD_01 = historyADAGRAD_01.history['acc']
    val_accuracyADAGRAD_01 = historyADAGRAD_01.history['val_acc']
    lossADAGRAD_01 = historyADAGRAD_01.history['loss']
    val_lossADAGRAD_01 = historyADAGRAD_01.history['val_loss']

    confusion_matrixADAGRAD_01 = new_confusion_matrix(ModelADAGRAD_01, validationGeneratorADAGRAD_01)

    plot_accuracy(
        accuracyADAGRAD_01,
        val_accuracyADAGRAD_01,
        lossADAGRAD_01,
        val_lossADAGRAD_01,
        confusion_matrixADAGRAD_01,
        lab='ADAGRAD_01'
    )

    # History
    all_models.append('ADAGRAD_01')
    all_acc.append(max(historyADAGRAD_01.history['acc']))
    all_loss.append(min(historyADAGRAD_01.history['loss']))
    all_val_acc.append(max(historyADAGRAD_01.history['val_acc']))
    all_val_loss.append(min(historyADAGRAD_01.history['val_loss']))


def model_adadelta_01():
    ModelADADELTA_01 = ModelGenerator.new_model_adadelta()  # Create new ADADELTA Model

    trainDataGenADADELTA_01 = ImageDataGenerator(rescale=1. / 255)
    testDataGenADADELTA_01 = ImageDataGenerator(rescale=1. / 255)

    trainGeneratorADADELTA_01 = trainDataGenADADELTA_01.flow_from_directory(
        SetupModule.trainDirectory,
        target_size=(img_rows, img_cols),
        batch_size=_globalBatchSize,
        class_mode='categorical'
    )

    validationGeneratorADADELTA_01 = testDataGenADADELTA_01.flow_from_directory(
        SetupModule.validationDirectory,
        target_size=(img_rows, img_cols),
        batch_size=_globalBatchSize,
        class_mode='categorical'
    )

    historyADADELTA_01 = ModelADADELTA_01.fit(
        trainGeneratorADADELTA_01,
        steps_per_epoch=_globalStepsPerEpoch,
        epochs=_globalEpochAmount,
        validation_data=validationGeneratorADADELTA_01,
        validation_steps=_globalValidationSteps
    )

    # Save Output
    ModelADADELTA_01.save(SetupModule.modelsDirectory + SetupModule.saveFileName.format('ADADELTA_01'))

    accuracyADADELTA_01 = historyADADELTA_01.history['acc']
    val_accuracyADADELTA_01 = historyADADELTA_01.history['val_acc']
    lossADADELTA_01 = historyADADELTA_01.history['loss']
    val_lossADADELTA_01 = historyADADELTA_01.history['val_loss']

    confusion_matrixADADELTA_01 = new_confusion_matrix(ModelADADELTA_01, validationGeneratorADADELTA_01)

    plot_accuracy(
        accuracyADADELTA_01,
        val_accuracyADADELTA_01,
        lossADADELTA_01,
        val_lossADADELTA_01,
        confusion_matrixADADELTA_01,
        lab='ADADELTA_01'
    )

    # History
    all_models.append('ADADELTA_01')
    all_acc.append(max(historyADADELTA_01.history['acc']))
    all_loss.append(min(historyADADELTA_01.history['loss']))
    all_val_acc.append(max(historyADADELTA_01.history['val_acc']))
    all_val_loss.append(min(historyADADELTA_01.history['val_loss']))


def model_adamax_01():
    ModelADAMAX_01 = ModelGenerator.new_model_adamax()  # Create new ADAMAX Model

    trainDataGenADAMAX_01 = ImageDataGenerator(rescale=1. / 255)
    testDataGenADAMAX_01 = ImageDataGenerator(rescale=1. / 255)

    trainGeneratorADAMAX_01 = trainDataGenADAMAX_01.flow_from_directory(
        SetupModule.trainDirectory,
        target_size=(img_rows, img_cols),
        batch_size=_globalBatchSize,
        class_mode='categorical'
    )

    validationGeneratorADAMAX_01 = testDataGenADAMAX_01.flow_from_directory(
        SetupModule.validationDirectory,
        target_size=(img_rows, img_cols),
        batch_size=_globalBatchSize,
        class_mode='categorical'
    )

    historyADAMAX_01 = ModelADAMAX_01.fit(
        trainGeneratorADAMAX_01,
        steps_per_epoch=_globalStepsPerEpoch,
        epochs=_globalEpochAmount,
        validation_data=validationGeneratorADAMAX_01,
        validation_steps=_globalValidationSteps
    )

    # Save Output
    ModelADAMAX_01.save(SetupModule.modelsDirectory + SetupModule.saveFileName.format('ADAMAX_01'))

    accuracyADAMAX_01 = historyADAMAX_01.history['acc']
    val_accuracyADAMAX_01 = historyADAMAX_01.history['val_acc']
    lossADAMAX_01 = historyADAMAX_01.history['loss']
    val_lossADAMAX_01 = historyADAMAX_01.history['val_loss']

    confusion_matrixADAMAX_01 = new_confusion_matrix(ModelADAMAX_01, validationGeneratorADAMAX_01)

    plot_accuracy(
        accuracyADAMAX_01,
        val_accuracyADAMAX_01,
        lossADAMAX_01,
        val_lossADAMAX_01,
        confusion_matrixADAMAX_01,
        lab='ADAMAX_01'
    )

    # History
    all_models.append('ADAMAX_01')
    all_acc.append(max(historyADAMAX_01.history['acc']))
    all_loss.append(min(historyADAMAX_01.history['loss']))
    all_val_acc.append(max(historyADAMAX_01.history['val_acc']))
    all_val_loss.append(min(historyADAMAX_01.history['val_loss']))


def model_adam_01_es():
    ModelADAM_01_ES = ModelGenerator.new_model_adam()

    trainDataGenADAM_01_ES = ImageDataGenerator(rescale=1. / 255)
    testDataGenADAM_01_ES = ImageDataGenerator(rescale=1. / 255)

    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitorujemy stratę na zbiorze walidacyjnym
        patience=3,  # Liczba epok bez poprawy, po których trening zostanie zatrzymany
        restore_best_weights=True  # Przywrócenie wag modelu z epoki o najlepszych wynikach
    )

    trainGeneratorADAM_01_ES = trainDataGenADAM_01_ES.flow_from_directory(
        SetupModule.trainDirectory,
        target_size=(img_rows, img_cols),
        batch_size=_globalBatchSize,
        class_mode='categorical'
    )

    validationGeneratorADAM_01_ES = testDataGenADAM_01_ES.flow_from_directory(
        SetupModule.validationDirectory,
        target_size=(img_rows, img_cols),
        batch_size=_globalBatchSize,
        class_mode='categorical'
    )

    historyADAM_01_ES = ModelADAM_01_ES.fit(
        trainGeneratorADAM_01_ES,
        steps_per_epoch=_globalStepsPerEpoch,
        epochs=_globalEpochAmount,
        validation_data=validationGeneratorADAM_01_ES,
        validation_steps=_globalValidationSteps,
        callbacks=[early_stopping]
    )

    # Save Output
    ModelADAM_01_ES.save(SetupModule.modelsDirectory + SetupModule.saveFileName.format('ADAM_01_ES'))

    accuracyADAM_01_ES = historyADAM_01_ES.history['acc']
    val_accuracyADAM_01_ES = historyADAM_01_ES.history['val_acc']
    lossADAM_01_ES = historyADAM_01_ES.history['loss']
    val_lossADAM_01_ES = historyADAM_01_ES.history['val_loss']

    confusion_matrixADAM_01_ES = new_confusion_matrix(ModelADAM_01_ES, validationGeneratorADAM_01_ES)

    plot_accuracy(
        accuracyADAM_01_ES,
        val_accuracyADAM_01_ES,
        lossADAM_01_ES,
        val_lossADAM_01_ES,
        confusion_matrixADAM_01_ES,
        lab='Model ModelADAM_01_ES'
    )

    # History
    all_models.append('ADAM_01_ES')
    all_acc.append(max(historyADAM_01_ES.history['acc']))
    all_loss.append(min(historyADAM_01_ES.history['loss']))
    all_val_acc.append(max(historyADAM_01_ES.history['val_acc']))
    all_val_loss.append(min(historyADAM_01_ES.history['val_loss']))


def model_rotated_images_01():
    ModelRotated_01 = ModelGenerator.new_model_adamax()
    ModelRotated_01.add(Dropout(0.05))

    testDataGenRMS_02 = ImageDataGenerator(rescale=1. / 255)

    trainDataGenRMS_02 = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,  # Randomly rotate images in the range (0-40 degrees)
        width_shift_range=0.15,  # Randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # Randomly shift images vertically (fraction of total height)
        shear_range=0.15,  # Shear intensity (shear angle in counter-clockwise direction in degrees)
        zoom_range=0.15,  # Range for random zoom
        horizontal_flip=True,  # Randomly flip images horizontally
        fill_mode='nearest'  # Fill mode for filling in newly created pixels after rotation or shifting
    )

    trainGeneratorRMS_02 = trainDataGenRMS_02.flow_from_directory(
        SetupModule.trainDirectory,
        target_size=(img_rows, img_cols),
        batch_size=_globalBatchSize,
        class_mode='categorical'
    )

    validationGeneratorRMS_02 = testDataGenRMS_02.flow_from_directory(
        SetupModule.validationDirectory,
        target_size=(img_rows, img_cols),
        batch_size=_globalBatchSize,
        class_mode='categorical'
    )

    historyRMS_02 = ModelRotated_01.fit(
        trainGeneratorRMS_02,
        steps_per_epoch=_globalStepsPerEpoch,
        epochs=_globalEpochAmount,
        validation_data=validationGeneratorRMS_02,
        validation_steps=_globalValidationSteps
    )

    # Save Output
    ModelRotated_01.save(SetupModule.modelsDirectory + SetupModule.saveFileName.format('RMS_02'))

    accuracyRMS_02 = historyRMS_02.history['acc']
    val_accuracyRMS_02 = historyRMS_02.history['val_acc']
    lossRMS_02 = historyRMS_02.history['loss']
    val_lossRMS_02 = historyRMS_02.history['val_loss']

    confusion_matrixRMS_02 = new_confusion_matrix(ModelRotated_01, validationGeneratorRMS_02)

    plot_accuracy(
        accuracyRMS_02,
        val_accuracyRMS_02,
        lossRMS_02,
        val_lossRMS_02,
        confusion_matrixRMS_02,
        lab='Model RMS_02'
    )

    # History
    all_models.append('Rotated')
    all_acc.append(max(historyRMS_02.history['acc']))
    all_loss.append(min(historyRMS_02.history['loss']))
    all_val_acc.append(max(historyRMS_02.history['val_acc']))
    all_val_loss.append(min(historyRMS_02.history['val_loss']))


# Start Models

# model_rms_01()
# model_adam_01()
model_adamax_01()
model_rotated_images_01()

# Summary
pyplot.plot(all_models, all_acc, 'bo-', label='Top Accuracy')
pyplot.plot(all_models, all_loss, 'ro-', label='Top Loss')
pyplot.plot(all_models, all_val_acc, 'b*-', label='Top Validation Accuracy')
pyplot.plot(all_models, all_val_loss, 'r*-', label='Top Validation Loss')

pyplot.xlabel('Model')
pyplot.ylabel('Accuracy')

pyplot.title('Model Comparison')
pyplot.legend()

# Wyświetlanie wykresu
pyplot.show()
