# Summary
This document provides an introductory guide to Deep Neural Networks (DNNs) using Keras and TensorFlow, demonstrating a practical application for classifying handwritten digits from the MNIST dataset. It covers the essential steps of defining network architecture, configuring the training process, and evaluating the model's performance.

## Key Points
* **Problem:** Classify grayscale images of handwritten digits (0-9) from the MNIST dataset.
* **Dataset:** MNIST, comprising 60,000 training images and 10,000 test images, each 28x28 pixels.
* **Frameworks:** TensorFlow and Keras are used for implementation.
* **DNN Workflow:**
    * Build the neural network architecture.
    * Train the network using training data (images and labels).
    * Evaluate predictions on unseen test data.
* **Network Architecture (Simple Dense Network):**
    * Composed of a sequence of two `Dense` (fully-connected) layers.
    * The first `Dense` layer uses 'relu' activation and requires specifying `input_shape`.
    * The final `Dense` layer is a 10-way 'softmax' layer, outputting probability scores for each of the 10 digit classes.
* **Training Configuration (Compilation Step):**
    * **Loss Function:** Measures how well the network is performing (e.g., `categorical_crossentropy` for multi-class classification).
    * **Optimizer:** Algorithm to update network weights based on the loss (e.g., `rmsprop`).
    * **Metrics:** Quantities to monitor during training and testing (e.g., `accuracy`).
* **Data Preprocessing:**
    * Images are reshaped from 2D arrays (28, 28) to 1D vectors (28 * 28).
    * Pixel values are scaled from the [0, 255] range to [0, 1] for optimal neural network input.
    * Labels are converted to categorical (one-hot encoded) format, which is required by `categorical_crossentropy`.
* **Model Evaluation:** Performance is assessed on a separate test set. A noticeable difference between training accuracy and test accuracy indicates overfitting.

## Important Details
* Keras is built on TensorFlow and often comes pre-included in newer TensorFlow versions.
* The `Sequential` API in Keras is introduced for building models layer by layer. The functional API is mentioned as a more advanced alternative for future use.
* The `mnist.load_data()` function directly loads the MNIST dataset as NumPy arrays.
* Training involves calling the `fit` method with parameters like `epochs` (number of full passes through the training data) and `batch_size` (number of samples per gradient update).
* The `evaluate` method provides the test loss and test accuracy.
* Overfitting occurs when a model performs significantly worse on new data than on the data it was trained on, indicating it has memorized the training data rather than learned general patterns.

## Approaches
* **Supervised Learning:** The core approach, where the model learns to map input images to their corresponding labels.
* **Feedforward Neural Network:** The architecture uses `Dense` layers, which are characteristic of feedforward networks where information flows in one direction.
* **Softmax Activation:** Used in the output layer for multi-class classification to produce a probability distribution over the classes.
* **Categorical Crossentropy Loss:** A standard loss function for multi-class classification problems with one-hot encoded labels.
* **RMSprop Optimizer:** An adaptive learning rate optimizer commonly used in deep learning.

## Examples
* **Importing Keras and Loading Data:**
    ```python
    from tensorflow import keras
    from keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    ```
* **Defining Network Architecture:**
    ```python
    from keras import models
    from keras import layers

    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))
    ```
* **Compiling the Network:**
    ```python
    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    ```
* **Converting Labels to Categorical:**
    ```python
    from keras.utils import to_categorical
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    ```
* **Evaluating the Network:**
    ```python
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)
    ```

## References
* MNIST dataset (assembled by the National Institute of Standards and Technology).
* Keras documentation (implicitly referenced through code examples and functionality descriptions).
* TensorFlow documentation (implicitly referenced).