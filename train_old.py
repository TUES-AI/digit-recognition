def loss(outputs, target):
    loss = 0
    for i in range(len(outputs)):
        loss += ((outputs[i] - (target if i == target else 0)) ** 2).mean()
    return loss

