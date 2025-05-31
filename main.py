# File: main.py
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

from layers import Layer_Dense, Layer_Dropout
from activations import Activation_ReLU, Activation_Softmax,Loss_CategoricalCrossentropy,Activation_softmax_loss_categoricalcrossentropy
#from losses import Loss_CategoricalCrossentropy
from optimizers import (
    Optimizer_Adam,
    Optimizer_GD,
    Optimizer_Adagrad,
    Optimizer_RMSProp
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network on spiral data using the selected optimizer.")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'gd', 'adagrad', 'rmsprop'], help='Optimizer to use for training.')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Set the logging level.')
    return parser.parse_args()

def select_optimizer(name):
    if name == 'adam':
        return Optimizer_Adam(learning_rate=0.05, decay=5e-5)
    elif name == 'gd':
        return Optimizer_GD(learning_rate=0.05, decay=5e-5, momentum=0.9)
    elif name == 'adagrad':
        return Optimizer_Adagrad(learning_rate=0.05, decay=5e-5)
    elif name == 'rmsprop':
        return Optimizer_RMSProp(learning_rate=0.05, decay=5e-5)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

def build_model():
    return (
        Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4),
        Activation_ReLU(),
        Layer_Dropout(0.1),
        Layer_Dense(64, 3),
        Activation_softmax_loss_categoricalcrossentropy()
        #Optimizer_Adam(learning_rate=0.05,decay=5e-5)
    )

def train_model(X, y, optimizer, layers, loss_activation, epochs=10001):
    dense1, activation1, dropout1, dense2, activation2 = layers
    for epoch in range(10001):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dropout1.forward(activation1.output)
        dense2.forward(dropout1.output)
        data_loss = loss_activation.forward(dense2.output,y)
        regularization_loss = (loss_activation.loss.regularization_loss(dense1) +
                            loss_activation.loss.regularization_loss(dense2)
        )
        loss = data_loss+regularization_loss
        
        #print(loss_activation.output[:5])
        #print(f'Loss:{loss}')

        #calculate accuracy
        predictions = np.argmax(loss_activation.output,axis=1)
        if len(y.shape)==2:
            y = np.argmax(y,axis=1)
            
        accuracy = np.mean(predictions==y)
        
        if not epoch%100:
            print(f'epoch: {epoch},'+
                f'acc: {accuracy:.3f},'+
                f'loss: {loss:.3f},'+
                f'data_loss: {data_loss:.3f}'+
                f'reg loss: {regularization_loss:.3f}'+
                f'Learning Rate: {optimizer.current_learning_rate:.3f}')
        #print(f'accurcay: {accuracy}')

        #backward pass
        loss_activation.backward(loss_activation.output,y)
        dense2.backward(loss_activation.dinputs)
        dropout1.backward(dense2.dinputs)
        activation1.backward(dropout1.dinputs)
        dense1.backward(activation1.dinputs)
        
        optimizer.pre_update_param()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_param()
'''
    for epoch in range(epochs):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dropout1.forward(activation1.output)
        dense2.forward(dropout1.output)
        activation2.forward(dense2.output)

        data_loss = loss_function.calculate(activation2.output, y)
        reg_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)
        total_loss = data_loss + reg_loss

        predictions = np.argmax(activation2.output, axis=1)
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        if epoch % 100 == 0:
            logging.info(
                f"Epoch: {epoch}, Accuracy: {accuracy:.3f}, Loss: {total_loss:.3f}, Data Loss: {data_loss:.3f}, Reg Loss: {reg_loss:.3f}, LR: {optimizer.current_learning_rate:.5f}"
            )

        loss_function.backward(activation2.output, y)
        dense2.backward(loss_function.dinputs)
        dropout1.backward(dense2.dinputs)
        activation1.backward(dropout1.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.pre_update_param()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_param()
'''
def evaluate_model(X_test, y_test, layers, loss_function):
    dense1, activation1, _, dense2, activation2 = layers

    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)

    loss = loss_function.forward(dense2.output,y_test)
    predictions = np.argmax(activation2.output, axis=1)
    if y_test.ndim == 2:
        y_test = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test)

    logging.info(f"Validation - Accuracy: {accuracy:.3f}, Loss: {loss:.3f}")

def main():
    nnfs.init()
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    X, y = spiral_data(samples=100, classes=3)
    X_test, y_test = spiral_data(samples=100, classes=3)

    optimizer = select_optimizer(args.optimizer)
    dense1, activation1, dropout1, dense2,loss_activation = build_model()
    layers = (dense1, activation1, dropout1, dense2, loss_activation)

    train_model(X, y, optimizer, layers, loss_activation)
    evaluate_model(X_test, y_test, layers, loss_activation)

if __name__ == '__main__':
    main()
