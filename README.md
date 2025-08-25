# Deep Learning From Scratch

## Overview

This repository contains some popular deep learning architectures which are implemented using NumPy, without popular libraries like PyTorch or TensorFlow.

These are generally not meant to be optimized and efficient implementations, rather used for understanding how they operate "under the hood".

Architectures currently implemented:

- Multilayer Perceptron (MLP)

- Convolutional Neural Network (CNN)

- Recurrent Neural Network (RNN)

- Long Short Term Memory (LSTM)

- Autoencoder

## uv Setup

1. **Create virtual environment and install dependencies** (optional):

    ```bash
    uv sync
    ```

2. **Install dlfs library**:

    ```bash
    uv pip install -e .
    ```

## pip Setup

1. **Set up a virtual environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Register virtual environment as an available kernel** (optional, for running notebooks):

    ```bash
    pip install ipykernel
    python -m ipykernel install --user
    ```

## Datasets

MNIST - https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Fashion MNIST - https://www.kaggle.com/datasets/zalando-research/fashionmnist

Clean Weather - https://www.kaggle.com/datasets/awsaf49/clean-weather-dataset

Tiny Shakespeare - https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

## Side notes

The dlfs library, classes and methods are written in [Sentdex](https://github.com/Sentdex) style according to the book [Neural Networks from Scratch](https://nnfs.io/). This book however covers only the basics, which is why I would recommend this book to anyone seeking out to understand the basics of neural networks.

Many thanks to the authors [Daniel Kukiela](https://github.com/daniel-kukiela) and [Harrison Kinsley](https://github.com/Sentdex) for laying out the foundations.