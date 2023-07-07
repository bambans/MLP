from numpy import argmax, amax, zeros_like, array, where, arange, unique, zeros
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, colorbar, savefig

def one_hot_encoding_max(input):
    result = zeros_like(input)
    result[input == amax(input, axis=1, keepdims=True)] = 1
    return result

def bipolar(x, theta):
    return where(x < theta, 0, 1)

def interpretation(input_hot_encoded, dictionary):
    labels = array(dictionary)

    if len(dictionary) == 2:
        try:
            return where(input_hot_encoded == 1, True, False)
        except Exception as error:
            print(f'Não foi possível interpretar a entrada. Erro:\n{error}\n')
    else:
        return labels[argmax(input_hot_encoded, axis = 1)]
        
def confusion(expected, encoded_test, file_path, matrix_model):
    # Best path to save file: f'confusion_matrix_{file_path.split("/")[-1].split(".")[0]}.png'

    # matrix_model: if true, Characters, if false, logic ports
    return None