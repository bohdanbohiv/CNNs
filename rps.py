import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory


def main():
    train_ds = image_dataset_from_directory(
        'rps',
        image_size=(150, 150),
    )
    class_names = train_ds.class_names
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds = image_dataset_from_directory(
        'rps-test-set',
        image_size=(150, 150),
    ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    model = keras.Sequential([
        keras.Input((150, 150, 3)),
        layers.Rescaling(1/255),

        layers.RandomFlip('horizontal'),
        layers.RandomRotation(.2),
        layers.RandomZoom(.2),

        layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(512, activation=tf.nn.relu),
        layers.Dropout(.2),
        layers.Dense(len(class_names), activation=tf.nn.softmax)
    ])
    model.summary()
    model.compile(
        metrics=['accuracy'],
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.RMSprop(),
    )
    history = model.fit(
        train_ds,
        epochs=5,
        validation_data=val_ds,
    )
    # model.save('rps.keras')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.show()


if __name__ == '__main__':
    main()
