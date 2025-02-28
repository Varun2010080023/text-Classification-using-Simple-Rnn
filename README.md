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
