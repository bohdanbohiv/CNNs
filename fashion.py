import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.datasets import ascent
from tensorflow import keras
from tensorflow.keras import layers


def convolution_and_pooling():
    def imshow(i: np.ndarray, axes, title) -> None:
        axes.set_title(title)
        axes.grid(False)
        axes.imshow(i)

    def transform_img(i: np.ndarray, filter: np.ndarray,
            weight=1) -> np.ndarray:
        i_tf = np.copy(i)
        size_x = i_tf.shape[0]
        size_y = i_tf.shape[1]
        for x in range(1, size_x - 1):
            for y in range(1, size_y - 1):
                pixel = (i[x - 1:x + 2, y - 1:y + 2] * filter).sum() * weight
                if pixel < 0:
                    pixel = 0
                elif pixel > 255:
                    pixel = 255
                i_tf[x, y] = pixel
        return i_tf

    def maxpooling_2by2(i: np.ndarray) -> np.ndarray:
        return np.array(
            [i[::2, ::2], i[::2, 1::2], i[1::2, ::2], i[1::2, 1::2]]).max(
            axis=0)

    _, axs = plt.subplots(1, 2, layout='constrained')
    plt.gray()

    i = ascent()
    # imshow(i, axs[0][0], 'original')

    # filter = np.array([[0,  1, 0],
    #                    [1, -4, 1],
    #                    [0,  1, 0]])
    # i_tf = transform_img(i, filter)
    # imshow(i_tf, axs[0][1], 'edges and straight lines')

    filter_horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # i_horizontal = transform_img(i, filter_horizontal)
    # imshow(i_horizontal, axs[1][0], 'horizontal lines')

    i_vertical = transform_img(i, filter_horizontal.T)
    imshow(i_vertical, axs[0], 'vertical lines')

    i_v_pooling = maxpooling_2by2(i_vertical)
    imshow(i_v_pooling, axs[1], 'vertical with pooling')

    plt.show()


def main():
    (train_imgs, train_labels), (
    test_imgs, test_labels) = keras.datasets.fashion_mnist.load_data()
    # normalisation improves accuracy dramatically
    train_imgs, test_imgs = train_imgs / 255, test_imgs / 255

    model = keras.Sequential([
        keras.Input((28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(2, 2), layers.Flatten(),
        layers.Dense(64, activation=tf.nn.relu), layers.Dropout(.2),
        layers.Dense(10, activation=tf.nn.softmax),
    ])
    model.compile(
        keras.optimizers.Adam(),
        keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']  # smart way
    )
    model.summary()

    class AccCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') > 0.95:
                print('\nReached 95% accuracy so cancelling training!')
                self.model.stop_training = True

    model.fit(train_imgs, train_labels, epochs=5, callbacks=[AccCallback()])
    test_loss, test_acc = model.evaluate(test_imgs, test_labels)
    print(f'Test accuracy: {test_acc * 100:.4}%, Test loss: {test_loss:.4}')

    # only the conv / max pool layers
    num_l = 4
    features_model = keras.Model(inputs=model.inputs,
        outputs=[layer.output for layer in model.layers[:num_l]])
    layer_names = [layer.name for layer in model.layers[:num_l]]

    _, axs = plt.subplots(3, num_l, layout='constrained')
    convolution_number = 6
    img_indexes = 0, 23, 28
    for i, img in enumerate(img_indexes):
        features = features_model.predict(test_imgs[img].reshape(1, 28, 28, 1))
        for layer in range(num_l):
            f = features[layer]
            axs[i, layer].imshow(f[0, :, :, convolution_number],
                                 cmap='inferno')
            axs[i, layer].grid(False)
            axs[i, layer].set_title(layer_names[layer])
    plt.show()


if __name__ == '__main__':
    main()
