# Multilayer percpetron
#
# @author Emilio Garzia, 2024

import argparse as ap
import numpy as np
from MLP import MultilayerPerceptron

# input arguments
parser = ap.ArgumentParser()
parser.add_argument("-l", "--load_model", action="store_true", help="Use a pre trained npz file model")
parser.add_argument("-x", "--input_x", help="Value of the X logic port")
parser.add_argument("-y", "--input_y", help="Value of the Y logic port")
parser.add_argument("-t", "--train_model", action="store_true", help="Learn a new model")
parser.add_argument("-f", "--file_name", help="Specify the file name of the new npz trained model")
parser.add_argument("-e", "--epochs", help="Specify the number of iterations for the training")
parser.add_argument("-lr", "--learning_rate", help="eta value, learning rate")
parser.add_argument("-db", "--decision_boundary", action="store_true", help="plot the decision boundary of your model")
args = parser.parse_args()

# Hyper parameters
epochs = int(args.epochs) if args.epochs is not None else 10000
learning_rate = float(args.learning_rate) if args.learning_rate is not None else 0.1

# Driver code
if __name__ == "__main__":

    ########## TRAINING MODE ##########

    if args.train_model:
        # Dataset
        X = np.array([[0,0],[0,1],[1,0],[1,1]]) # input
        y = np.array([[0],[1],[1],[0]]) # target

        # Initialize the MLP model
        mlp = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1)
        
        # Train the model
        mlp.fit(X, y, epochs=epochs, learning_rate=learning_rate)

        # Save the model on the file system
        mlp.save_weights(args.file_name)

        # Test the model outputs
        predictions = mlp.predict(X)
        print("Output test:")
        print(predictions)
        
    ######## PREDICTION MODE ########

    if args.load_model:
        model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1)
        model.load_model(args.file_name)

        X = int(args.input_x)
        Y = int(args.input_y)
        input = np.array([[X,Y]])

        predictions = model.predict(input)
        
        Z = 1 if predictions>0.5 else 0

        print(f'{X} XOR {Y}: {Z}')
        print(f'predicted weight: {predictions}')

    ########## DECISION BOUNDARY PLOT ########
        
    if args.decision_boundary:
        # Dataset
        X = np.array([[0,0],[0,1],[1,0],[1,1]]) # input
        y = np.array([[0],[1],[1],[0]]) # target
        
        model = MultilayerPerceptron(input_size=2, hidden_size=4, output_size=1)
        model.load_model(args.file_name)

        model.plot_decision_boundary(X=X, y=y)