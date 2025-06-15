# Summary
This document introduces two major unsupervised deep learning architectures for image generation and representation learning: Autoencoders and Generative Adversarial Networks (GANs). It explains the core concepts, design, and implementation of different autoencoder types (simple and convolutional) for dimensionality reduction and image reconstruction, and provides a conceptual overview of GANs for generating realistic synthetic images.

## Key Points
* **Image Generation Goal:** To learn a low-dimensional "latent space" where points can be mapped to realistic images.
* **Generator/Decoder:** The module that maps a latent space point to an image. Called a generator in GANs and a decoder in autoencoders.
* **Autoencoders:**
    * Neural networks designed to learn efficient data codings (representations) in an unsupervised manner.
    * Consist of two parts: an **encoder** (maps input to a latent vector space) and a **decoder** (reconstructs the input from the latent vector).
    * Trained to reconstruct their own input, minimizing the difference between the input and its reconstruction.
    * Constraints (e.g., low-dimensional, sparse latent code) force the autoencoder to learn meaningful, compressed representations.
* **Simple Autoencoder:** Uses fully-connected (`Dense`) layers for both encoder and decoder. Effective for basic dimensionality reduction on flattened image data.
* **Convolutional Autoencoder:** Uses `Conv2D` and `MaxPooling2D` layers for the encoder and `Conv2D` and `UpSampling2D` layers for the decoder. More effective for image data due to their ability to capture spatial hierarchies and maintain spatial information.
* **Keras Functional API:** Used for building autoencoders (and other complex models) when a simple linear stack of layers (`Sequential` model) is insufficient. It allows direct manipulation of tensors and chaining layers by treating them as functions.
* **Generative Adversarial Networks (GANs):**
    * Composed of two competing neural networks: a **generator** and a **discriminator**.
    * **Generator Network:** Takes a random vector from the latent space and generates a synthetic image. It aims to produce images that are indistinguishable from real ones to fool the discriminator.
    * **Discriminator Network (Adversary):** Takes an image (either real from the training set or synthetic from the generator) and outputs a probability of whether the image is real or fake. It aims to correctly classify real vs. fake images.
    * **Adversarial Training:** The two networks are trained simultaneously in a minimax game. The generator tries to minimize the discriminator's ability to distinguish real from fake, while the discriminator tries to maximize this ability. This competition drives both networks to improve, ultimately leading the generator to produce highly realistic images.
    * **Complexity:** GANs are generally more complex to implement and train than autoencoders, requiring careful tuning.

## Important Details
* Input data (e.g., MNIST digits) for autoencoders must be normalized (e.g., to [0, 1] range).
* For simple autoencoders, 2D images (e.g., 28x28) are flattened into 1D vectors (e.g., 784-dimensional).
* For convolutional autoencoders, input images retain their 2D (or 3D with channels) shape.
* The output layer of autoencoders often uses a `sigmoid` activation function to ensure pixel values are within the [0, 1] range, matching the normalized input.
* `binary_crossentropy` is a common loss function for autoencoders, as it treats pixel reconstruction as a per-pixel binary classification problem.
* `Adam` optimizer is frequently used for training autoencoders.
* Visualizing reconstructed images is crucial to assess the autoencoder's performance. Loss of detail indicates inadequate `encoding_dim` or model capacity.
* In convolutional autoencoders, `UpSampling2D` layers are used in the decoder to reverse the down-sampling effect of `MaxPooling2D` in the encoder. `padding='same'` helps maintain spatial dimensions during convolution.
* GANs are powerful for generating novel, realistic data, but their training is notoriously challenging and unstable.

## Approaches
* **Data Preparation for Autoencoders:**
    * Loading image datasets (e.g., MNIST).
    * Normalizing pixel values (e.g., to [0, 1]).
    * Reshaping data for appropriate encoder input (flattened for simple, 2D/3D for convolutional).
* **Building Models with Keras Functional API:**
    * Define `Input` layers.
    * Chain layers by passing the output tensor of one layer as the input to the next (`output_tensor = Layer(params)(input_tensor)`).
    * Create `Model` instances specifying overall inputs and outputs (`model = Model(inputs=..., outputs=...)`).
    * Separately define encoder and decoder sub-models for prediction/generation from the latent space.
* **Training Autoencoders:**
    * Compile the autoencoder model with `Adam` optimizer and `binary_crossentropy` loss.
    * Fit the model using input data as both features and targets (`autoencoder.fit(x_train, x_train, ...)`).
* **Evaluating Autoencoder Performance:**
    * Predict on test data using the trained autoencoder.
    * Visually compare original test images with their reconstructed versions.
* **Conceptual Understanding of GANs:** Grasping the adversarial training process between the generator and discriminator.
* **Practical Exploration of GANs:** Running pre-built GAN implementations (e.g., in Google Colab) to observe their behavior and generated outputs.

## Examples
* **Simple Autoencoder Architecture (using Functional API):**
    ```python
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model

    encoding_dim = 32
    input_img = Input(shape=(784,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)
    autoencoder = Model(input_img, decoded)

    encoder = Model(input_img, encoded)
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    ```
* **Convolutional Autoencoder Architecture:**
    ```python
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
    from tensorflow.keras.models import Model

    input_img = Input(shape=(28, 28, 1))
    # Encoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x) # Example simplified
    # Decoder
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    ```

## References
* F. Chollet, "Deep Learning with Python" (book).
* MNIST dataset.
* Keras Documentation.
* Stanford Lectures (PDF and YouTube) for GANs.
* Google Colab (for running pre-implemented GANs).