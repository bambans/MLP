from pickle import dump as pd, load as pl
from numpy import array, where
from csv import reader, QUOTE_NONNUMERIC

def dump(file_path, weights):
    try:
        with open(file_path, 'wb') as weights_file:
            pd(weights, weights_file)
    except Exception as error:
        print(f'Erro ao salvar os pesos no arquivo: {error}')

def load(file_path):
    try:
        with open(file_path, 'rb') as weights_file:
            weights = pl(weights_file)
            return weights
    except Exception as error:
        print(f'Não foi possível abrir o arquivo {file_path}: {error}')


def read_input(MLP, file_path):
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as test:
                train_file = reader(test, quoting=QUOTE_NONNUMERIC)

                characters_input = []
                expected_output = []

                for row in train_file:
                    characters_list = []
                    expected_list = []
                    for index, value in enumerate(row):
                        if index < MLP.INPUT_LAYER_SIZE:
                            characters_list.append(value)
                        else:
                            expected_list.append(value)

                    ### Converting Python's list to NumPy's array format
                    characters_list = array(characters_list)
                    expected_list = array(expected_list)

                    ### Converting -1 values to 0, because of the Sigmoid Function limits
                    characters_list = where(characters_list == -1, -1, 1)
                    expected_list = where(expected_list == -1, 0, 1) 

                    ### Appending inputs to inputs lists
                    characters_input.append(characters_list)
                    expected_output.append(expected_list)

                ### Converting the list of inputs to NumPy's array format
                characters_input = array(characters_input)
                expected_output = array(expected_output)

                return characters_input, expected_output
        
    except Exception as error:
        print(f'Não foi possível ler o arquivo {file_path}: {error}')

    