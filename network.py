import math
from helpers import get_image_data, print_ascii
from param_io import load_params

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def per_layer_forward_pass(layer_parameters, inputs, is_final=False):
    weights = layer_parameters["weights"]
    biases = layer_parameters["biases"]
    outputs = []

    for i in range(len(biases)):
        outputs.append(biases[i])
        for j in range(len(inputs)):
            outputs[i] += weights[j][i] * inputs[j]
        if not is_final:
            outputs[i] = sigmoid(outputs[i])

    return outputs

def forward_pass(network,inputs):
    for i in range(len(network)):
        layer_parameters = network[i]
        if i == len(network) - 1:
            inputs = per_layer_forward_pass(layer_parameters, inputs, is_final=True)
        else:
            inputs = per_layer_forward_pass(layer_parameters, inputs)

    return inputs

def softmax(inputs): 
    sum = 0
    for i in range(len(inputs)):
        sum += math.exp(inputs[i])
    for i in range(len(inputs)):
        inputs[i] = math.exp(inputs[i]) / sum
    return inputs

def argmax(inputs):
    max_index = 0
    for i in range(1, len(inputs)):
        if inputs[i] > inputs[max_index]:
            max_index = i
    return max_index

def predict(network,image_data):
    outputs = forward_pass(network,image_data)
    return argmax(softmax(outputs))

# network = load_params()
# print("Prediction on this image:")
# print_ascii(get_image_data("train",2))
# print("Is - ",predict(network,get_image_data("train",2)))
