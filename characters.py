from MLP.mlp import MLP

from dictionary import letters as alphabet

if __name__ == "__main__":

    characters = MLP(
        INPUT_LAYER_SIZE = 63,
        HIDDEN_LAYER_SIZE = 35,
        OUTPUT_LAYER_SIZE = 7,
        MAXIMUM_ERROR = 0.0625,
        LEARNING_RATE = 0.01,
        MAX_EPOCHS = 100000,
        ALPHABET = alphabet
        )
    
    print(characters)

    tests = ['caracteres-Fausett/caracteres-limpo.csv', 'caracteres-Fausett/caracteres-ruido.csv', 'caracteres-Fausett/caracteres-ruido20.csv']

    # characters.weights_loader('dumpsChar/hidden_weights.dump', 'dumpsChar/hidden_bias.dump', 'dumpsChar/output_weights.dump', 'dumpsChar/output_bias.dump')

    for test in tests:
        print(f'<<<<<<<<<< {test} >>>>>>>>>>')
        characters.train(test)
        characters.weights_dumper('dumpsChar/hidden_weights.dump', 'dumpsChar/hidden_bias.dump', 'dumpsChar/output_weights.dump', 'dumpsChar/output_bias.dump')
        # characters.weights_loader('dumpsChar/hidden_weights.dump', 'dumpsChar/hidden_bias.dump', 'dumpsChar/output_weights.dump', 'dumpsChar/output_bias.dump')
        characters.test(test)
        characters.run(test)        
