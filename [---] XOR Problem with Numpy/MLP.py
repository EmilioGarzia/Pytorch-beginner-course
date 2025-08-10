# MLP implementation with Numpy
#
# @author Emilio Garzia, 2024

import numpy as np
import matplotlib.pyplot as plt

class MultilayerPerceptron:

    """MLP Classifier

    Parameters
    -------------------
    input_size: Number of the neurons into the input layer
    hidde_layer: Number of the hidden layers
    output_size: Number of the neurons into the output layer
    
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # init the weights randomly
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        
        # init bias
        self.bias_hidden = np.ones((1, self.hidden_size))
        self.bias_output = np.ones((1, self.output_size))

    """Logisitic Sigmoid function
    
    Parameters
    -----------------        
    x: input variable for the sigmoid function
    
    """
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    """Logistic function derivate

    Parameters
    -----------------

    x: input value for the derivation
    
    """

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    """Push the information forward in the net

    Parameters
    -----------------

    inputs: input data to forward 

    """

    def forward(self, inputs):
        # Compute the output in the hidden layers
        hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        
        # Compute the output in the output layer
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output = self.sigmoid(output_input)
        
        return output, hidden_output
    
    """Backpropagation of the informations
    
    Parameters
    -----------------

    inputs: the data arrived by the previous layer
    y: target of the samples
    output: prediction of the model on that sample
    hidden_output: the output of the current hidden layer
    learning_rate: value of the eta variable
    
    """
    
    def backward(self, inputs, y, output, hidden_output, learning_rate):
        # compute the error and delta error in the output layer
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        
        # compute the error and delta error in the hidden layer
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)
        
        # update weights
        self.weights_hidden_output += np.dot(hidden_output.T, output_delta) * learning_rate
        self.weights_input_hidden += np.dot(inputs.T, hidden_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate
        
    def fit(self, inputs, y, epochs=10000, learning_rate=0.1):
        for epoch in range(epochs):
            # Forward pass
            output, hidden_output = self.forward(inputs)
            
            # Backpropagation
            self.backward(inputs, y, output, hidden_output, learning_rate)
            
            # Compute the delta on the error (loss)
            loss = np.mean(np.square(y - output))
            if epoch % 100 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    """Predict values with the trained model
    
    Parameters
    ---------------------

    inputs: data to classify with the pre trained MLP model

    """         
    
    def predict(self, inputs):
        output, _ = self.forward(inputs)
        return output

    """Save the weights model with the savez() method of numpy

    Parameters
    -------------------

    filename: name of the model file to save on fs
    
    """

    def save_weights(self, filename):
        weights = {
            "weights_input_hidden": self.weights_input_hidden,
            "weights_hidden_output": self.weights_hidden_output,
            "bias_hidden": self.bias_hidden,
            "bias_output": self.bias_output
        }

        if filename != None:
            filename = f'{filename}.npz'
        else:
            filename = "my_model.npz"

        np.savez(filename, **weights)
        print("Weights saved successfully!")

    """Load weights model from a npz file

    Parameters
    -----------------------

    filename: npz file that contain the weights of your model
    
    """ 
    
    def load_model(self, filename):
        if filename is None:
            filename = "my_model.npz"
        
        weights = np.load(filename)
        
        self.weights_input_hidden = weights["weights_input_hidden"]
        self.weights_hidden_output = weights["weights_hidden_output"]
        self.bias_hidden = weights["bias_hidden"]
        self.bias_output = weights["bias_output"]

        print("Model loaded successfully!")

    """Decision Boundary Plotting

    Parameters
    -------------------
    X: samples from dataset
    y: targets of the data

    """

    def plot_decision_boundary(self, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
        
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Decision Boundary')
        plt.show()