import math
from helpers import get_image_data, print_ascii
from param_io import load_params

# Тук ще дефинирам главните параметри:
network_size = [784, 16, 16, 10]
learning_rate = 0.1
number_of_epochs = 1000
images_to_train_on = 10000

def relu(x):
    return max(0, x)

def per_layer_forward_pass(layer_parameters, inputs, is_final=False):
    weights = layer_parameters["weights"]
    biases = layer_parameters["biases"]
    outputs = []

    for i in range(len(biases)):
        outputs.append(biases[i])
        for j in range(len(inputs)):
            outputs[i] += weights[j][i] * inputs[j]
        if not is_final:
            outputs[i] = relu(outputs[i])

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
    max_val = inputs[0]
    for i in range(1, len(inputs)):
        if inputs[i] > max_val:
            max_val = inputs[i]

    exps = []
    sum_exps = 0
    for i in range(len(inputs)):
        exp_val = math.exp(inputs[i] - max_val)
        exps.append(exp_val)
        sum_exps += exp_val

    for i in range(len(inputs)):
        inputs[i] = exps[i] / sum_exps
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
