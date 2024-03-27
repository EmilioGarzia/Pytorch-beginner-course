# Multilayer Perceptron (MLP): XOR Problem 
#
# @description In this program we will use a keras framework to implement a MLP to solve a XOR logic gate problem
# @author Emilio Garzia

import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential, layers
from keras.models import load_model
import tensorflow as tf

# Function that plot the boundary precision of our model
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Print boundary decision
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    # Print all points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu)
    plt.xlabel('A')
    plt.ylabel('B')
    plt.title('Decision Boundary')
    plt.show()

# If h5 model not exists start traning
def train_model():
    neurons = 10

    # Dataset
    X_train = ([[0,0],[0,1],[1,0],[1,1]])   # input data
    y_train = ([[0],[1],[1],[0]])           # output label

    # Init model
    model = Sequential()

    model.add(layers.Dense(neurons, activation="relu", input_dim=2)) # hidden layer
    model.add(layers.Dense(1, activation="sigmoid", input_dim=2))    # output layer

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x=X_train, y=y_train, epochs=1000, batch_size=16)

    loss, accuracy = model.evaluate(X_train, y_train)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    predictions = model.predict(X_train)
    print("Predictions: ", predictions)

    plot_decision_boundary(np.array(X_train),np.array(y_train), model)

    model.save("MLP_XOR.h5")


# Main
if __name__ == "__main__":
    try:
        model = load_model("MLP_XOR.h5")
        A = None
        B = None    

        while(A!='q' or B!='q'):
            A = input("Insert input A: ")
            B = input("Insert input B: ")

            try:
                predictions = model.predict(([[int(A),int(B)]]))
                output = 1 if predictions[0] > 0.5 else 0 
                print(A, " XOR ", B, " = ", output)
                print("Prediction: ", predictions)
            except:
                print("Insert correctly input: 0 or 1")
    except:
        train_model()