import argparse


parser = argparse.ArgumentParser(description = 'Define hyperparameters for RNN/BRNN')

parser.add_argument('--model', type=str, default = 'BRNN', help = 'The type of model to run - "RNN"/"BRNN"')
parser.add_argument('--result_file', type = str, default = 'test_file', help = 'Name of test results file. Will be stored in the results folder')
parser.add_argument('--hidden_dim', type = int, default =5, help = 'Dimension of the hidden layer')
parser.add_argument('--epochs', type = int, default =100, help = 'Number of epochs to run')
parser.add_argument('--samples', type = int, default =100, help = 'Number of times the weight is sampled from the posterior distribution')
parser.add_argument('--lr', type = float, default =0.01, help = 'Learning rate of the model')


