# Summary
This document covers binary classification using Deep Neural Networks (DNNs) with the IMDB movie review dataset and introduces essential concepts for hyperparameter optimization, including validation sets, Keras Callbacks, and automated search algorithms like Grid Search and Keras Tuner.

## Key Points
* **Binary Classification Problem:** Classifying movie reviews as positive or negative based on text content.
* **IMDB Dataset:** Contains 50,000 highly-polarized movie reviews (25,000 for training, 25,000 for testing), pre-processed into sequences of integers representing words. Only the top 10,000 most frequent words are kept.
* **Data Preparation:** Integer sequences must be converted into numerical tensors for neural network input. One-hot encoding is chosen, transforming sequences into 10,000-dimensional binary vectors.
* **Network Architecture for Binary Classification:**
    * Typically consists of two intermediate `Dense` layers with `relu` activation (e.g., 16 hidden units each).
    * A final `Dense` layer with a single unit and `sigmoid` activation, outputting a probability score between 0 and 1.
* **Network Training:**
    * **Loss Function:** `binary_crossentropy` is recommended for binary classification problems where the network outputs probabilities.
    * **Optimizer:** `rmsprop` is a common choice.
    * **Metrics:** `accuracy` is used to monitor performance.
* **Validation:** A dedicated validation set (e.g., 10,000 samples from training data) is crucial to monitor model performance on unseen data during training and detect overfitting.
* **Overfitting:** Occurs when a model performs exceptionally well on training data but poorly on validation/test data, indicating it has memorized the training examples.
* **Hyperparameters:** Parameters that control the learning process, such as the number of epochs, learning rate, and optimizer choice.
* **Callbacks:** Functions that execute at specific stages during training (e.g., `on_epoch_end`) to perform actions like early stopping, model checkpointing, or dynamic parameter adjustment.
* **Hyperparameter Optimization Algorithms:**
    * **Grid Search:** Systematically explores a predefined set of hyperparameter combinations.
    * **Random Search:** Randomly samples hyperparameter combinations, often more efficient than Grid Search for high-dimensional hyperparameter spaces.
    * **Keras Tuner:** A dedicated library for hyperparameter tuning with Keras models, offering various search algorithms.

## Important Details
* `num_words=10000` limits the vocabulary size in the IMDB dataset, managing data dimensionality.
* One-hot encoding creates sparse vectors where each index corresponds to a word.
* `relu` activation functions introduce non-linearity, while `sigmoid` squashes output to a probability range for binary classification.
* `binary_crossentropy` measures the distance between predicted probabilities and true binary labels.
* The `History` object returned by `model.fit()` contains training and validation metrics for plotting.
* Early stopping using callbacks is an effective way to prevent overfitting by stopping training when validation performance no longer improves.
* Automated hyperparameter optimization can be computationally intensive, especially Grid Search with many parameters or values.
* `keras-tuner` requires Python 3.6+ and TensorFlow 2.0+.

## Approaches
* **Data Vectorization:** Transforming raw text data (sequences of integers) into a suitable numerical format (one-hot encoded vectors) for a DNN.
* **Deep Dense Network for Classification:** Designing a feedforward neural network with multiple `Dense` layers for learning complex patterns in vectorized data.
* **Validation Set for Overfitting Detection:** Splitting data into training, validation, and test sets to monitor generalization performance during training.
* **Early Stopping Callback:** Implementing a custom Keras callback or using built-in ones to automatically stop training based on monitored metrics (e.g., validation accuracy).
* **Grid Search for Hyperparameter Tuning:** Using `sklearn.model_selection.GridSearchCV` with `KerasClassifier` (or `scikeras.wrappers.KerasClassifier`) to exhaustively search for optimal hyperparameter combinations.
* **Keras Tuner for Hyperparameter Tuning:** Defining a `build_model` function that accepts `hp` (hyperparameter) objects and using `keras_tuner.RandomSearch` (or other tuners) to automate the search.

## Examples
* **Loading IMDB Dataset:**
    ```python
    from tensorflow.keras.datasets import imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    ```
* **One-Hot Encoding Function:**
    ```python
    import numpy as np
    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results
    ```
* **Custom Keras Callback for Early Stopping:**
    ```python
    import tensorflow as tf
    class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.9):
          print("\nReached 90% accuracy so cancelling training!")
          self.model.stop_training = True
    ```
* **Grid Search Implementation (conceptual):**
    ```python
    from sklearn.model_selection import GridSearchCV
    from keras.wrappers.scikit_learn import KerasClassifier # or scikeras.wrappers.KerasClassifier

    def create_model():
        # ... define your Keras model ...
        return model

    model = KerasClassifier(build_fn=create_model)
    param_grid = dict(batch_size=[32, 64], epochs=[3, 5])
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_results = grid.fit(X,Y)
    ```
* **Keras Tuner Implementation (conceptual):**
    ```python
    import keras_tuner # Requires installation

    def build_model(hp):
        # ... define your Keras model, using hp.Int, hp.Choice, etc. ...
        return model

    tuner = keras_tuner.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5)
    tuner.search(train_images, train_labels, epochs=5, batch_size=128, validation_data=(val_images, val_labels))
    best_model = tuner.get_best_models()[0]
    ```

## References
* Keras Documentation (for `imdb.load_data`, `Sequential`, `Dense`, activation functions, `compile`, `fit`, `evaluate`, Callbacks).
* Scikit-learn Documentation (for `GridSearchCV`, `train_test_split`).
* Keras Tuner Documentation and Tutorials.