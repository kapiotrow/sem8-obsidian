# Summary
This document introduces Recurrent Neural Networks (RNNs) for processing sequential data, specifically focusing on text classification. It covers data vectorization techniques, the architecture and training of simple RNNs, and the advantages and implementation of Long Short-Term Memory (LSTM) layers to address challenges like vanishing gradients.

## Key Points
* **Data Vectorization for Text:**
    * **One-Hot Encoding:** Represents words as sparse binary vectors (e.g., a 10,000-dimensional vector with a '1' at the index corresponding to the word). Useful for `Dense` layers but loses semantic relationships.
    * **Word Embeddings:** A more sophisticated technique that maps words into a lower-dimensional, dense vector space where similar words are located close to each other. The embedding parameters are learned by the network during training, allowing it to capture semantic nuances. The dimensionality of the embedding space is a hyperparameter.
* **Recurrent Neural Networks (RNNs):**
    * A class of neural networks designed to process sequences of inputs, maintaining an internal state (memory) that allows the output at the current timestep to depend on past computations.
    * Unlike feedforward networks, inputs in RNNs are not independent.
    * **`SimpleRNN` Layer:** A basic RNN layer in Keras.
        * Can return either full sequences (`return_sequences=True`) as a 3D tensor `(batch_size, timesteps, output_features)` or only the last output (`return_sequences=False` or default) as a 2D tensor `(batch_size, output_features)`.
        * Multiple `SimpleRNN` layers can be stacked, requiring intermediate layers to have `return_sequences=True`.
* **Long Short-Term Memory (LSTM) Networks:**
    * A specialized type of RNN designed to overcome the vanishing gradient problem, making them more effective at learning long-term dependencies in sequences.
    * LSTMs contain internal "gates" (input, forget, output gates) that regulate the flow of information, allowing them to selectively remember or forget past data.
    * Well-suited for tasks with unknown time lags, such as time series prediction, speech recognition, and text classification.

## Important Details
* Loading IMDB dataset: `imdb.load_data(num_words=10000)` keeps only the top 10,000 most frequent words.
* `sequence.pad_sequences(maxlen=500)` ensures all input sequences have the same length by padding or truncating, which is necessary for batch processing in neural networks.
* The `Embedding` layer is typically the first layer in an RNN when using word embeddings, taking integer-encoded sequences as input and outputting dense word vectors.
* The number of parameters in `SimpleRNN` and `LSTM` layers depends on the input dimensionality and the number of hidden units.
* For binary classification problems like IMDB movie review sentiment, the output layer is a `Dense` layer with one neuron and `sigmoid` activation, using `binary_crossentropy` as the loss function.
* Training RNNs involves specifying `epochs`, `batch_size`, and often a `validation_split` to monitor performance on unseen data.
* LSTMs generally perform better than `SimpleRNN` for tasks involving longer sequences due to their ability to mitigate vanishing gradients.

## Approaches
* **Text Preprocessing for RNNs:**
    * Tokenization (words to integers).
    * Padding sequences to a uniform length.
    * Choosing between one-hot encoding (simple, loses semantics) and word embeddings (more complex, learns semantic relationships).
* **RNN Architecture Design:**
    * **Basic RNN:** `Embedding` layer followed by one or more `SimpleRNN` layers and a final `Dense` classifier.
    * **Stacked RNNs:** Using `return_sequences=True` for all but the last `SimpleRNN` layer to maintain sequential output across layers.
    * **LSTM Network:** Replacing `SimpleRNN` layers with `LSTM` layers for improved long-term dependency learning.
* **Model Compilation:** Using `rmsprop` optimizer, `binary_crossentropy` loss (for binary classification), and `accuracy` metric.
* **Training and Evaluation:** Fitting the model to padded training data and evaluating its performance on a validation set, monitoring loss and accuracy curves to understand model behavior.

## Examples
* **IMDB Data Loading and Padding:**
    ```python
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.preprocessing import sequence

    (input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=10000)
    input_train = sequence.pad_sequences(input_train, maxlen=500)
    input_test = sequence.pad_sequences(input_test, maxlen=500)
    ```
* **SimpleRNN Model Architecture:**
    ```python
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

    model = Sequential()
    model.add(Embedding(10000, 32)) # 10000 max_features, 32-dimensional embedding
    model.add(SimpleRNN(32)) # 32 output units
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    ```
* **LSTM Model Architecture:**
    ```python
    from tensorflow.keras.layers import LSTM # Import LSTM layer

    model = Sequential()
    model.add(Embedding(10000, 32))
    model.add(LSTM(32)) # Replacing SimpleRNN with LSTM
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    ```
* **Model Compilation and Training (shared for SimpleRNN/LSTM):**
    ```python
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(input_train, y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)
    ```

## References
* IMDB dataset.
* Keras documentation (for `imdb.load_data`, `sequence.pad_sequences`, `Embedding`, `SimpleRNN`, `LSTM`, `Dense`, `Sequential`, `compile`, `fit`).
* Information on Word Embeddings (external links provided in original content).