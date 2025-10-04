import math
import network as network_code

def relu_derivative(x):
    """Derivative of ReLU: 1 if x > 0, else 0"""
    return 1 if x > 0 else 0

def compute_gradients(network, inputs, target_label):
    """
    Compute gradients for a single training example using backpropagation
    Returns gradients for weights and biases for each layer
    """
    activations = [inputs]
    pre_activations = []

    current_inputs = inputs
    for i in range(len(network)):
        layer = network[i]
        weights = layer["weights"]
        biases = layer["biases"]

        pre_act = []
        for j in range(len(biases)):
            z = biases[j]
            for k in range(len(current_inputs)):
                z += weights[k][j] * current_inputs[k]
            pre_act.append(z)
        pre_activations.append(pre_act)

        if i < len(network) - 1:
            current_inputs = [max(0, z) for z in pre_act]
        else:
            current_inputs = pre_act.copy()
        activations.append(current_inputs)

    gradients_w = []
    gradients_b = []

    output_activations = activations[-1]
    softmax_probs = network_code.softmax(output_activations)

    delta = softmax_probs.copy()
    delta[target_label] -= 1

    layer_grad_w = []
    layer_grad_b = []

    prev_activations = activations[-2]
    for j in range(len(delta)):
        layer_grad_b.append(delta[j])

        neuron_grads = []
        for k in range(len(prev_activations)):
            neuron_grads.append(delta[j] * prev_activations[k])
        layer_grad_w.append(neuron_grads)

    transposed_grad_w = []
    for k in range(len(prev_activations)):
        transposed_grad_w.append([])
        for j in range(len(delta)):
            transposed_grad_w[k].append(layer_grad_w[j][k])

    gradients_w.insert(0, transposed_grad_w)
    gradients_b.insert(0, layer_grad_b)

    for layer_idx in range(len(network) - 2, -1, -1):
        layer = network[layer_idx]
        weights = layer["weights"]
        biases = layer["biases"]

        next_delta = delta
        next_weights = network[layer_idx + 1]["weights"]

        current_delta = []
        for j in range(len(biases)):
            sum_next = 0
            for k in range(len(next_delta)):
                sum_next += next_delta[k] * next_weights[j][k]

            pre_act = pre_activations[layer_idx][j]
            relu_deriv = relu_derivative(pre_act)
            current_delta.append(sum_next * relu_deriv)

        delta = current_delta

        layer_grad_w = []
        layer_grad_b = []

        prev_activations = activations[layer_idx]
        for j in range(len(delta)):
            layer_grad_b.append(delta[j])

            neuron_grads = []
            for k in range(len(prev_activations)):
                neuron_grads.append(delta[j] * prev_activations[k])
            layer_grad_w.append(neuron_grads)

        transposed_grad_w = []
        for k in range(len(prev_activations)):
            transposed_grad_w.append([])
            for j in range(len(delta)):
                transposed_grad_w[k].append(layer_grad_w[j][k])

        gradients_w.insert(0, transposed_grad_w)
        gradients_b.insert(0, layer_grad_b)

    return gradients_w, gradients_b

def update_weights(network, gradients_w, gradients_b, learning_rate):
    """Update network weights and biases using gradients"""
    for layer_idx in range(len(network)):
        layer = network[layer_idx]

        for i in range(len(layer["weights"])):
            for j in range(len(layer["weights"][i])):
                layer["weights"][i][j] -= learning_rate * gradients_w[layer_idx][i][j]

        for j in range(len(layer["biases"])):
            layer["biases"][j] -= learning_rate * gradients_b[layer_idx][j]

def compute_batch_gradients(network, batch_images, batch_labels):
    """Compute average gradients over a batch"""
    batch_size = len(batch_images)

    total_grad_w = []
    total_grad_b = []

    for layer in network:
        weights = layer["weights"]
        biases = layer["biases"]

        layer_grad_w = []
        for i in range(len(weights)):
            layer_grad_w.append([0.0] * len(weights[i]))

        layer_grad_b = [0.0] * len(biases)

        total_grad_w.append(layer_grad_w)
        total_grad_b.append(layer_grad_b)

    for img, label in zip(batch_images, batch_labels):
        grad_w, grad_b = compute_gradients(network, img, label)

        for layer_idx in range(len(network)):
            for i in range(len(grad_w[layer_idx])):
                for j in range(len(grad_w[layer_idx][i])):
                    total_grad_w[layer_idx][i][j] += grad_w[layer_idx][i][j]

            for j in range(len(grad_b[layer_idx])):
                total_grad_b[layer_idx][j] += grad_b[layer_idx][j]

    for layer_idx in range(len(network)):
        for i in range(len(total_grad_w[layer_idx])):
            for j in range(len(total_grad_w[layer_idx][i])):
                total_grad_w[layer_idx][i][j] /= batch_size

        for j in range(len(total_grad_b[layer_idx])):
            total_grad_b[layer_idx][j] /= batch_size

    return total_grad_w, total_grad_b
