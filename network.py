import numpy
import math
import random
from helpers import get_image_data, print_ascii
from param_io import load_params

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def per_layer_forward_pass(layer_parameters, inputs, next_layer_size):
    weights = layer_parameters["weights"]
    biases = layer_parameters["biases"]
    outputs = []

    for i in range(next_layer_size):
        outputs.append(biases[i])
        for j in range(len(inputs)):
            outputs[i] += weights[j][i] * inputs[j]
        outputs[i] = sigmoid(outputs[i])

    return outputs

def forward_pass(inputs):
    network = load_params()
    for i in range(len(network)):
        layer_parameters = network[i]
        next_layer_size = len(layer_parameters["biases"])
        inputs = per_layer_forward_pass(layer_parameters, inputs, next_layer_size)
    return inputs

def argmax(inputs): 
    val = max(inputs)
    for i in range(len(inputs)):
        if inputs[i] == val:
            return i

def predict(image_data):
    outputs = forward_pass(image_data)
    return argmax(outputs)

print("Prediction on this image:")
print_ascii(get_image_data("train",2))
print("Is - ",predict(get_image_data("train",2)))
