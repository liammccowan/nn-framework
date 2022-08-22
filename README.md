# nn-framework
nn-framework is a simple deep learning framework written in only numpy. This framework aims to strike a middle gound between features and readability.

## Features:
nn-framework offers the following features:
- Layers: Linear, Dropout, Batch Normalization, Convolutional, Max Pool, Flatten, LSTM
- Activation functions: ReLU
- Loss functions: Softmax with loss
- Optimizers: SGD, Adam
- A simple model class with training functions

The framework is demonstrated in several notebooks:
- Fully connected layers: `nn_mnist.ipynb`, which trains a model using fully connected layers on the mnist dataset. This notebook uses Keras for the dataset and matplotlib for plotting.
- Convolutional layers: `cnn_mnist.ipynb`, which trains a model using convolutional layers on the mnist dataset. This notebook uses Keras for the dataset and matplotlib for plotting.
- LSTM layers: `lstm_char.ipynb`, which trains a model using stacked LSTM layers to generate text.

## Notes:
nn-framework was initially inspired by the following implementations:

github.com/parmeet/dll_numpy

github.com/Xylambda/Freya

The convolutional and max pooling layers are based off of those presented in Stanford's CS231n: Convolutional Neural Networks for Visual Recognition course.

nn-framework no longer strongly resembles either of the above implementations, but it follows a similar structure.

nn-framework also takes inspiration from github.com/Nico-Curti/NumpyNet, but aims to offer a more compact structure to its code.

The mnist digits notebook is similar to the Keras implementation found at nextjournal.com/gkoehler/digit-recognition-with-keras.
