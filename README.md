# DIGIT RECOGNITION
## TUES AI CLUB - Lecture 1

In this repository we implemented the MNIST digit recognition using a simple forward neural network from scratch in Python. 

The network is trained by a temporary JAX backpropagation implementation.

### Requirements
- Python installed system wide
- Create a virtual environment and activate it
- Install the required packages using pip:
```bash
pip install numpy pillow flask jax
```

### Usage
1. Clone the repository:
```bash
git clone https://github.com/tue-ai-club/digit-recognition.git
```
2. Inside the folder run the dowload script to get the MNIST dataset:
```bash
python download.py
# or if you want the images to be able to see them:
pyhon download.py --export-png
```
3. Initialize the model with:
```bash
python initialize_network.py
```
4. Train the model with:
```bash
python train.py
```
5. Run the web app with:
```bash
python server.py
```
6. Open your browser and go to `http://localhost:6969` to use the digit recognition app.

That is it!

### Making modifications to the network:
You can modify the network parameters in `network.py`:
```python
# Тук ще дефинирам главните параметри:
network_size = [784, 16, 16, 10]
learning_rate = 0.1
number_of_epochs = 1000
images_to_train_on = 10000
```

You can also change make other changes like adding more data to our small dataset, chaning the activation function (and its derivative!), etc.

Good luck exeprimenting!
