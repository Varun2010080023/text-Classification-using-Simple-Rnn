# IMDB Text Classification with TensorFlow Datasets

## Overview
This repository contains a text classification project that uses the IMDB dataset from TensorFlow Datasets (TFDS). The goal is to classify movie reviews as either positive or negative using deep learning techniques.

## Dataset
The IMDB dataset consists of 50,000 movie reviews, with an equal number of positive and negative reviews. It is available through TensorFlow Datasets and is preprocessed for easy use.

## Features
- Uses TensorFlow and Keras for model development.
- Implements a deep learning model using an Embedding layer and an LSTM (Long Short-Term Memory) network.
- Utilizes pre-trained word embeddings (optional) to enhance performance.
- Includes data preprocessing steps such as tokenization and padding.
- Provides model evaluation metrics including accuracy and loss visualization.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/imdb-text-classification.git
   cd imdb-text-classification
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:
   ```bash
   python train.py
   ```

## Usage
- Modify `train.py` to adjust hyperparameters like batch size, learning rate, or model architecture.
- Use `evaluate.py` to test the model on unseen data.
- Visualize training history with `plot_results.py`.

## Project Structure
```
├── data/                # Directory for storing dataset (if needed)
├── models/              # Saved models
├── notebooks/           # Jupyter notebooks for exploration
├── src/
│   ├── train.py         # Training script
│   ├── evaluate.py      # Model evaluation script
│   ├── preprocess.py    # Text preprocessing functions
│   ├── model.py         # Model architecture
├── requirements.txt     # Required dependencies
├── README.md            # Project documentation
```

## Results
- Achieved ~90% accuracy on the test set.
- Training and validation accuracy/loss curves available for reference.

#DataSet
# IMDb Dataset in TensorFlow

## Overview
The IMDb dataset is a widely used dataset for binary sentiment classification, containing 50,000 movie reviews labeled as positive or negative. TensorFlow provides built-in support for this dataset through `tensorflow.keras.datasets.imdb`.

## Dataset Details
- **Number of Samples**: 50,000 (25,000 for training, 25,000 for testing)
- **Classes**: 2 (Positive, Negative)
- **Format**: Each review is represented as a sequence of integers, where each integer maps to a word index in a predefined dictionary.
- **Preprocessing**: The dataset allows specifying the vocabulary size, truncating or padding sequences, and converting words back to text using a word index dictionary.

## Installation
Ensure you have TensorFlow installed:
```bash
pip install tensorflow
```

## Loading the Dataset
You can load the IMDb dataset directly from TensorFlow:
```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb

# Load dataset with a vocabulary size of 10,000
(vocab_size, max_length) = (10000, 500)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

print(f"Training samples: {len(x_train)}")
print(f"Testing samples: {len(x_test)}")
```

## Preprocessing
Before using the dataset in a neural network, we need to pad sequences to ensure uniform input sizes:
```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')
```

## Building a Sentiment Analysis Model
A simple LSTM-based sentiment classifier:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

## Training the Model
```python
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
```

## Evaluating the Model
```python
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

## Word Index Mapping
To decode reviews back to text:
```python
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(text_sequence):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text_sequence])

print(decode_review(x_train[0]))
```

## Conclusion
This README provides an overview of the IMDb dataset in TensorFlow, including dataset details, preprocessing steps, a simple LSTM model for sentiment classification, and evaluation techniques. You can extend this by experimenting with different architectures or fine-tuning hyperparameters to improve performance.

## References
- [TensorFlow IMDb Dataset Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)
- [IMDb Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)


