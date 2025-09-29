import numpy
from helpers import get_image_data

network_size = [784, 16, 16, 10]

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def forward_pass(layer_parameters, inputs, next_layer_size):
    weights = layer_parameters.weights
    biases = layer_parameters.biases

    for i in range(next_layer_size):
        activation = biases[i]
        for j in range(len(inputs)):
            activation += weights[i][j] * inputs[j]
        inputs[i] = sigmoid(activation)
    return inputs

def initialize_network_layer(layer1, layer2):
    a = numpy.sqrt(6/(layer1 + layer2))
    weights = (numpy.random.rand(layer2, layer1)*2 - 1) * a
    biases = numpy.zeros(layer2)
    return { "weights": weights, "biases": biases }

network = []

network.append(initialize_network_layer(network_size[0], network_size[1]))

print(network[0]["weights"].shape)
