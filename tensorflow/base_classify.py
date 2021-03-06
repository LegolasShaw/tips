# coding: utf-8
# @author: Shaw
# @datetime: 2019-02-27 9:28
# @Name: base_classify.py

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

if __name__ == "__main__":
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print(train_images.shape)
    print(train_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)

    # plt.figure()
    # plt.imshow(train_images[0])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    #
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])
    #
    # plt.show()

    # 构建神经网络需要配置模型的层，然后编译模型。
    model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                              keras.layers.Dense(128, activation=tf.nn.relu),
                              keras.layers.Dense(10, activation=tf.nn.softmax)])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    model.fit(train_images, train_labels, epochs=5)

    # 评估模型
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("test accuracy", test_acc)

    # img = test_images[0]
    #
    # img = (np.expand_dims(img,0))
    #
    predictions = model.predict(test_images)

    # print(np.argmax(predictions))

    def plot_image(i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)

        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             class_names[true_label]),color=color)


    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        this_plot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)
        this_plot[predicted_label].set_color('red')
        this_plot[true_label].set_color('blue')

    i=0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions,  test_labels)
    plt.show()




