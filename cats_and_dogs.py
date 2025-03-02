import os

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers, utils


def main():
    data_dir = 'PetImages'
    for cat in ('Cat', 'Dog'):
        for file in os.listdir(os.path.join(data_dir, cat)):
            filename = os.path.join(data_dir, cat, file)
            Image.open(filename, formats=('JPEG', ))
    training_ds, validation_ds = utils.image_dataset_from_directory(
        data_dir,
        label_mode='binary',
        image_size=(150, 150),
        seed=1,
        validation_split=.1,
        subset='both',
    )
    model = keras.models.Sequential([
        keras.Input((150, 150, 3)),
        layers.Rescaling(1/255),

        layers.Conv2D(16, (3, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(512, activation=tf.nn.relu),
        layers.Dropout(.2),
        layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'],
    )
    history = model.fit(
        training_ds,
        epochs=5,
        # epochs=15,
        validation_data=validation_ds,
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))
    plt.figure(layout='constrained')
    plt.plot(epochs, acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure(layout='constrained')
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.legend(loc=0)
    plt.show()


if __name__ == '__main__':
    main()
