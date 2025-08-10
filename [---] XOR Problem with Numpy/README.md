# Multilayer perceptron with `numpy` module

- [Description](#description)
- [How to execute](#how-to-execute)
- [Dependencies](#dependencies)
- [Author](#author)

## Description

In this project I have implemented two source code written in python:

1. The source code that contain the class of my MLP model with all properties and methods

1. The second python file is the `main.py` and we can use it to try the main three operations of this project
    - Train an MLP model to solve the non linear XOR problem
    - Use a pre trained model to predict our input
    - Plot the decision boundary of our saved model

## How to execute

As we can see our `main.py` can be used for three different features:

1. Training
1. Prediction
1. Decision boundary plotting

You can use a `-h` option to see all available settings before start the program

```
usage: main.py [-h] [-l] [-x INPUT_X] [-y INPUT_Y] [-t] [-f FILE_NAME] [-e EPOCHS]
               [-lr LEARNING_RATE] [-db]

options:
  -h, --help            show this help message and exit
  -l, --load_model      Use a pre trained npz file model
  -x INPUT_X, --input_x INPUT_X
                        Value of the X logic port
  -y INPUT_Y, --input_y INPUT_Y
                        Value of the Y logic port
  -t, --train_model     Learn a new model
  -f FILE_NAME, --file_name FILE_NAME
                        Specify the file name of the new npz trained model
  -e EPOCHS, --epochs EPOCHS
                        Specify the number of iterations for the training
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        eta value, learning rate
  -db, --decision_boundary
                        plot the decision boundary of your model
```

If you train a new MLP model you can launch the command at below:

```bash
python main.py -t
```

After this command the program save a model on file system named `my_model.npz` and we can try this model with the command:

```bash
python main.py -l -x 0 -y 1
```

If you want to plot a decision boundary of your saved model, launch this command:

```bash
python main.py -db
```

## Dependencies

- [`numpy`](https://numpy.org/)
- [`matplotlib`](https://matplotlib.org/)

## Author

Emilio Garzia, 2024-
