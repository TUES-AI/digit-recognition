import math
import random
from param_io import save_params

network_size = [784, 16, 16, 10]

def initialize_network_layer(layer1, layer2):
    # initial params for ReLU
    a = math.sqrt(6/(layer1 + layer2))
    weights = []
    biases = []
    for _ in range(layer2):
        biases.append(0)

    for i in range(layer1):
        weights.append([])
        for _ in range(layer2):
            weights[i].append((random.random()*2-1) * a)
    return { "weights": weights, "biases": biases }

network = []

for i in range(1, len(network_size)):
    network.append(initialize_network_layer(network_size[i-1], network_size[i]))

for i in range(len(network)):
    print("Layer",i,"",len(network[i]["weights"]),"x",len(network[i]["weights"][0]))
print("----")

save_params(network)
