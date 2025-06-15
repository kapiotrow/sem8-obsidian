# Summary
This document provides an introduction to Convolutional Neural Networks (ConvNets) through two distinct classification problems: handwritten digit classification using the MNIST dataset and image classification (cats vs. dogs) with a small custom dataset. It covers ConvNet architecture design, data preprocessing, training, and evaluation, highlighting challenges like overfitting in small datasets.

## Key Points
* **ConvNet Fundamentals:** ConvNets are designed for image data, processing tensors of shape `(image_height, image_width, image_channels)`. They typically consist of `Conv2D` layers for feature extraction and `MaxPooling2D` layers for dimensionality reduction, followed by a `Dense` (classifier) part.
* **MNIST Digit Classification (Part 1):**
    * **Dataset:** MNIST, 60,000 training and 10,000 test grayscale images (28x28 pixels).
    * **Data Preparation:** Images reshaped to `(28, 28, 1)` to match ConvNet input expectations. Labels are typically one-hot encoded for multi-class classification.
    * **Architecture:** A stack of `Conv2D` (e.g., 32, 64, 64 filters with 3x3 kernel) and `MaxPooling2D` (2x2 filter) layers, followed by `Flatten` and `Dense` layers. The final `Dense` layer uses `softmax` activation for 10-class probability distribution.
    * **Training:** Uses `rmsprop` optimizer, `categorical_crossentropy` loss, and `accuracy` metric. Achieves high accuracy (e.g., ~0.99) on the test set.
* **Image Classification for Small Datasets (Cats vs. Dogs - Part 2):**
    * **Dataset:** A smaller custom dataset of 4,000 images (2,000 cats, 2,000 dogs), split into 2,000 training, 1,000 validation, and 1,000 test images.
    * **Data Organization:** Images must be arranged in separate directories for each class (e.g., `train/cats/`, `train/dogs/`).
    * **Data Preprocessing:** Keras's `image_dataset_from_directory` is used to automatically read, decode, resize (e.g., 150x150 pixels), rescale (to [0, 1] interval), and batch images. `label_mode="binary"` is used for binary classification.
    * **Architecture:** A larger ConvNet with more `Conv2D` and `MaxPooling2D` stages (e.g., 4x Conv2D with increasing filter depth from 32 to 128, 4x MaxPooling2D) to handle larger images and more complex patterns. The final `Dense` layer has a single unit with `sigmoid` activation for binary probability output.
    * **Training:** Uses `RMSprop` optimizer with a low learning rate (e.g., `1e-4`), `binary_crossentropy` loss, and `accuracy` metric.
* **Overfitting in Small Datasets:** A prominent issue where training accuracy significantly surpasses validation accuracy, indicating the model is memorizing training data. This section identifies the problem and briefly mentions future mitigation techniques like dropout and data augmentation.

## Important Details
* ConvNet input shape must include the channel dimension (e.g., `(28, 28, 1)` for grayscale, `(150, 150, 3)` for RGB).
* `Conv2D` layers increase feature map depth while `MaxPooling2D` layers decrease spatial dimensions, a common pattern in ConvNets.
* `Flatten` layer is essential to convert 3D convolutional outputs into 1D vectors for `Dense` classification layers.
* `categorical_crossentropy` is standard for multi-class problems (labels as one-hot vectors), while `binary_crossentropy` is for binary problems (single probability output).
* Proper organization of dataset directories is crucial for `image_dataset_from_directory` to infer labels automatically.
* Monitoring validation accuracy and loss during training is key to detecting overfitting.

## Approaches
* **Convolutional Feature Learning:** Using `Conv2D` layers to automatically learn hierarchical spatial features from raw pixel data.
* **Pooling for Downsampling:** Employing `MaxPooling2D` to reduce the spatial size of feature maps, making the model more robust to minor translations and reducing computational cost.
* **Densely Connected Classifier:** Attaching `Dense` layers at the end of the ConvNet to perform classification based on the extracted features.
* **Binary vs. Multi-class Classification Setup:** Adapting the output layer's number of units and activation function (`softmax` for multi-class, `sigmoid` for binary) and the loss function (`categorical_crossentropy` vs. `binary_crossentropy`) based on the problem type.
* **Automated Data Loading and Preprocessing:** Utilizing Keras utilities like `image_dataset_from_directory` to streamline the process of preparing image data for training.

## Examples
* **ConvNet Layer Definition:**
    ```python
    from keras import layers, models
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    # ... more conv/pooling layers ...
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax')) # For MNIST
    ```
* **Model Compilation for MNIST:**
    ```python
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    ```
* **Data Loading and Preprocessing for Cats vs. Dogs:**
    ```python
    from tensorflow.keras.preprocessing import image_dataset_from_directory

    train_dataset = image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        image_size=(150, 150),
        batch_size=20,
        shuffle=True,
    )
    # ... similar for val_dataset ...
    ```
* **Model Compilation for Cats vs. Dogs:**
    ```python
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', # or optimizers.RMSprop(lr=1e-4)
                  metrics=['acc'])
    ```
* **Model Training:**
    ```python
    history = model.fit(train_dataset, epochs=30, validation_data=val_dataset)
    ```

## References
* MNIST dataset.
* Kaggle "Dogs vs. Cats" dataset.
* Keras documentation (for `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `image_dataset_from_directory`, `compile`, `fit`, `evaluate`).