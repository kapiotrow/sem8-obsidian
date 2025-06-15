# Summary
This document introduces Transformer networks, a cutting-edge deep learning architecture that has revolutionized both Natural Language Processing (NLP) and Computer Vision (CV). It explains the fundamental concepts behind Transformers, including positional encoding and multi-head attention, and demonstrates their practical application for text and image classification tasks, highlighting their advantages over traditional RNNs and CNNs, especially for large datasets.

## Key Points
* **Transformers Overview:**
    * Introduced in 2017 ("Attention is All You Need").
    * Dominated NLP tasks (classification, translation, image captioning).
    * Adapted for computer vision (Vision Transformers - ViT) in 2020, outperforming CNNs on many tasks.
    * Characterized by their attention mechanism, allowing them to focus on relevant parts of input data.
    * Excel in text-to-text, text-to-image conversion, and bounding-box detection.
* **Original Transformers for NLP (Part 1):**
    * **Limitations of RNNs/LSTMs:** Slow training (sequential processing), limited long-term memory (vanishing gradient problem), performance degradation for very long sequences (e.g., >1000 words).
    * **Transformer Solution:**
        * **Positional Encoding:** Maps the relative or absolute position of tokens in a sequence, allowing the model to understand word order, which is crucial since attention mechanisms process inputs in parallel.
        * **Multi-Head Attention:** Calculates the correlation (attention) between each word and all other words in the sequence. It repeats this process multiple times ("heads") and averages the results to create rich "attention vectors" for each word.
        * **Parallel Computation:** Attention vectors are independent, enabling parallel processing on GPUs, leading to significantly faster training than RNNs.
    * **Architecture for Classification:** `TokenAndPositionEmbedding` layer, followed by `TransformerBlock` (containing `MultiHeadAttention`, `Feed-Forward Network`, `LayerNormalization`, `Dropout`), and then a traditional classifier (e.g., `GlobalAveragePooling`, `Dense` layers).
    * **Practical Considerations:** Requires TensorFlow 2.4+ for `MultiHeadAttention` layer. Pre-trained transformers like BERT (from Hugging Face) are widely used for advanced NLP tasks, especially with limited data.
* **Vision Transformers (ViT) (Part 2):**
    * **Concept:** Adapts the Transformer architecture to image data by treating image patches as "words" in a sequence.
    * **Differences from CNNs:** Lack inductive biases of CNNs (translation invariance, local receptive fields). Instead, ViTs look at the whole image and map relations between patches.
    * **Process:** Images are divided into fixed-size patches. These patches are then linearly embedded and combined with positional embeddings. The sequence of patch embeddings is fed into a standard Transformer encoder (Multi-Head Attention blocks). The final feature vector is passed to a Multi-Layer Perceptron (MLP) for classification.
    * **Interpretability:** Inherently interpretable due to their attention mechanisms, allowing visualization of important image regions (attention heatmaps).
    * **Implementation:** `vit_keras` library simplifies using pre-trained ViT models (B16, B32, L16, L32) for transfer learning. Requires `tensorflow-addons`.
    * **Limitations:** Transformers, especially ViTs, typically require *very large amounts of data* to outperform CNNs when trained from scratch. For smaller datasets, using pre-trained CNNs or transfer learning with pre-trained transformers is recommended.

## Important Details
* `maxlen` in NLP Transformers refers to the maximum sequence length after padding.
* The `call` function in custom Keras layers for `TokenAndPositionEmbedding` and `TransformerBlock` defines the forward pass logic.
* The "weird connections" in `TransformerBlock` (`inputs + attn_output`, `out1 + ffn_output`) refer to skip connections and Layer Normalization, critical components for stable training in deep networks.
* Hugging Face is a major hub for pre-trained Transformer models in NLP and increasingly in CV.
* `vit_keras` provides pre-trained ViT models with options for `image_size`, `activation`, `pretrained`, `include_top`, `pretrained_top`, and `classes`.
* Visualizing attention maps (e.g., using `vit_keras.visualize.attention_map`) helps understand which parts of an image the ViT focuses on.

## Approaches
* **Text Data Preprocessing for Transformers:** Padding sequences and custom `TokenAndPositionEmbedding` layer.
* **Building Transformer Blocks (NLP):** Implementing a `TransformerBlock` class with `MultiHeadAttention`, feed-forward layers, `LayerNormalization`, and `Dropout`.
* **NLP Transformer Model Construction:** Combining the `TokenAndPositionEmbedding` and `TransformerBlock` with global pooling and `Dense` layers for classification.
* **Transfer Learning with Vision Transformers:**
    * Importing a pre-trained ViT model (e.g., `vit.vit_b32`) from `vit_keras` with `include_top=False`.
    * Adding custom classification layers (`Dense`, `Flatten`, `Dropout`, `BatchNormalization`) on top of the ViT's feature extractor.
    * Compiling and training the model using image data (from `tf.data` or `ImageDataGenerator`).
* **Interpreting ViT Models:** Using `vit_keras.visualize.attention_map` to generate attention heatmaps on images to understand the model's focus.

## Examples
* **`TokenAndPositionEmbedding` Class Structure:**
    ```python
    class TokenAndPositionEmbedding(layers.Layer):
        def __init__(self, maxlen, vocab_size, embed_dim):
            super(TokenAndPositionEmbedding, self).__init__()
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
            self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        def call(self, x):
            # ... positional encoding logic ...
            return x_token_emb + positions
    ```
* **`TransformerBlock` Class Structure:**
    ```python
    class TransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
            super(TransformerBlock, self).__init__()
            self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),])
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = layers.Dropout(rate)
            self.dropout2 = layers.Dropout(rate)
        def call(self, inputs, training):
            # ... attention, skip connections, FFN ...
            return self.layernorm2(out1 + ffn_output)
    ```
* **ViT Model Import (using `vit_keras`):**
    ```python
    from vit_keras import vit
    vit_model = vit.vit_b32(
            image_size = 224, # Example
            activation = 'softmax', # Or None if you add custom top layers
            pretrained = True,
            include_top = False, # Essential for custom classification
            pretrained_top = False,
            classes = None # None if include_top is False
    )
    ```
* **Visualizing Attention:**
    ```python
    from vit_keras import visualize
    # ... load a single image ...
    attention_map = visualize.attention_map(model=vit_model, image=image)
    ```

## References
* "Attention Is All You Need" (Original Transformer paper, 2017).
* "An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale" (Original Vision Transformer paper, 2020).
* Keras Documentation Examples (for NLP Transformers and ViT implementation details).
* Hugging Face (for pre-trained Transformer models and tutorials).
* `vit_keras` documentation.
* TensorFlow Addons.