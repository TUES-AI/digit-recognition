import numpy
import math
import random
from helpers import get_image_data, print_ascii

network_size = [784, 16, 16, 10]

def initialize_network_layer(layer1, layer2):
    # initial params for sigmoid !!! $
    a = math.sqrt(6/(layer1 + layer2))
    weights = []
    biases = []
    for _ in range(layer2):
        biases.append(0)

    for i in range(layer1):
        weights.append([])
        for _ in range(layer2):
            weights[i].append(((random.random()*2-1)*(layer1* a)))
    return { "weights": weights, "biases": biases }

network = []

for i in range(1, len(network_size)):
    network.append(initialize_network_layer(network_size[i-1], network_size[i]))

for i in range(len(network)):
    print("Layer",i,"",len(network[i]["weights"]),"x",len(network[i]["weights"][0]))
print("----")

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def per_layer_forward_pass(layer_parameters, inputs, next_layer_size):
    weights = layer_parameters["weights"]
    biases = layer_parameters["biases"]
    outputs = []

    for i in range(next_layer_size):
        outputs.append(biases[i])
        for j in range(len(inputs)):
            outputs[i] += weights[j][i] * inputs[i]
        outputs[i] = sigmoid(outputs[i])


    return outputs

def forward_pass(network, inputs):
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

def predict(network, image_data):
    outputs = forward_pass(network, image_data)
    return argmax(outputs)

print("Prediction on this image:")
print_ascii(get_image_data("train",2))
print("Is - ",predict(network, get_image_data("train",2)))
print("Is - ",predict(network, get_image_data("train",1)))
print("Is - ",predict(network, get_image_data("train",4)))
print("Is - ",predict(network, get_image_data("train",5)))
print("Is - ",predict(network, get_image_data("train",7)))
print("Is - ",predict(network, get_image_data("train",8)))
print("Is - ",predict(network, get_image_data("train",0)))
