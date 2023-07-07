from numpy import dot, all, inf, set_printoptions

set_printoptions(linewidth=inf)

from .functions import sigmoid_funtion as activation
from .result_interpretation import one_hot_encoding_max, bipolar, interpretation, confusion as confusion_matrix

def MLP_test(characters_list, expected_list, dumps, dictionary, file_path):
    hidden_weights, hidden_bias, output_weights, output_bias = dumps

    expected = expected_list

    input_layer = characters_list
    hidden_layer = activation(dot(input_layer, hidden_weights) + hidden_bias)
    output_layer = activation(dot(hidden_layer, output_weights) + output_bias)

    if len(dictionary) != 2:
        encoded_test = one_hot_encoding_max(output_layer)
        matrix_one = True
    else:
        encoded_test = bipolar(output_layer, 0.1)
        matrix_one = False

    matrix = confusion_matrix(expected, encoded_test, file_path, matrix_one)

    result = f"""Predicted:\t{interpretation(encoded_test, dictionary).reshape(-1)}\nExpected:\t{interpretation(expected, dictionary).reshape(-1)}\nEvaluation:\t{all(encoded_test == expected, axis=1)}\nPass all:\t{all(encoded_test == expected)}\nConfusion Matrix:\n{matrix}"""

    print(result)
