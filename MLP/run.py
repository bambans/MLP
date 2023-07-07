from numpy import dot

from .functions import sigmoid_funtion as activation
from .result_interpretation import one_hot_encoding_max, bipolar, interpretation

def MLP_run(characters_list, dumps, dictionary):
    hidden_weights, hidden_bias, output_weights, output_bias = dumps

    input_layer = characters_list
    hidden_layer_in = dot(input_layer, hidden_weights) + hidden_bias
    hidden_layer = activation(hidden_layer_in)
    output_layer_in = dot(hidden_layer, output_weights) + output_bias
    output_layer = activation(output_layer_in)

    if len(dictionary) != 2:
        encoded = one_hot_encoding_max(output_layer)
    else:
        encoded = bipolar(output_layer, 0.1)

    result = f"""Predited:\t{interpretation(encoded, dictionary).reshape(-1)}"""

    print(result)
    print()
