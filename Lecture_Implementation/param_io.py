import pickle

def save_params(network):
    with open('params.pkl', 'wb') as f:
        pickle.dump(network,f)

def load_params():
    with open('params.pkl', 'rb') as f:
        network = pickle.load(f)
    return network
