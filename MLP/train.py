from numpy import dot, sum, any
from numpy.random import randn as normal_random_samples

from .functions import sigmoid_funtion as activation, sigmoid_derivative_function as activation_derivative
from .files import dump as save_weights

def MLP_train(MLP, characters_list, expected_list):
    epoch = 0

    hidden_weights = normal_random_samples(MLP.INPUT_LAYER_SIZE, MLP.HIDDEN_LAYER_SIZE)
    hidden_bias = normal_random_samples(1, MLP.HIDDEN_LAYER_SIZE)
    output_weights = normal_random_samples(MLP.HIDDEN_LAYER_SIZE, MLP.OUTPUT_LAYER_SIZE)
    output_bias = normal_random_samples(1, MLP.OUTPUT_LAYER_SIZE)

    keep_training = True

    ### Input and expected result
    input_layer = characters_list
    expected = expected_list

    while keep_training:
            
        ### Feedforward input propagation
        hidden_layer_in = dot(input_layer, hidden_weights) + hidden_bias
        hidden_layer = activation(hidden_layer_in)
        
        output_layer_in = dot(hidden_layer, output_weights) + output_bias
        output_layer = activation(output_layer_in)
        
        ### Error calculation
        output_error = expected - output_layer

        ### Error back propagation
        output_error_information = output_error * activation_derivative(output_layer_in)
        d_output_correction_term = MLP.LEARNING_RATE * dot(output_error_information.T, hidden_layer)

        hidden_error = dot(output_error_information, output_weights.T)
        hidden_error_information = hidden_error * activation_derivative(hidden_layer_in)
        d_hidden_corretion_term = MLP.LEARNING_RATE * dot(hidden_error_information.T, input_layer)

        ### Weights update
        output_weights += d_output_correction_term.T
        output_bias += MLP.LEARNING_RATE * sum(output_error_information, axis = 0)
        hidden_weights += d_hidden_corretion_term.T
        hidden_bias += MLP.LEARNING_RATE * sum(hidden_error_information, axis = 0)
 
        average_errors = sum(abs(output_error)) / len(expected_list)

        keep_training = True if epoch < MLP.MAX_EPOCHS and any(average_errors > MLP.MAXIMUM_ERROR) else False

        pretty_print = f"""- Epoque: {epoch}\n- Average Errors: {average_errors}\n- Keep Training: {keep_training}\n\n"""

        epoch += 1

    print(pretty_print, end='')

    # save_weights('dumps/hidden_weights.dump', hidden_weights)
    # save_weights('dumps/hidden_bias.dump', hidden_bias)
    # save_weights('dumps/output_weights.dump', output_weights)
    # save_weights('dumps/output_bias.dump', output_bias)

    return hidden_weights, hidden_bias, output_weights, output_bias