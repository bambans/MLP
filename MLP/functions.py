from numpy import exp

def sigmoid_funtion(x):
    return 1 / (1 + exp(-x))

def sigmoid_derivative_function(x):
    return sigmoid_funtion(x) * (1 - sigmoid_funtion(x))