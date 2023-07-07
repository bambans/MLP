from MLP.mlp import MLP

from dictionary import boolean as alphabet

if __name__ == "__main__":

    boolean = MLP(
        INPUT_LAYER_SIZE = 2,
        HIDDEN_LAYER_SIZE = 3,
        OUTPUT_LAYER_SIZE = 1,
        MAXIMUM_ERROR = 0.012125,
        LEARNING_RATE = 0.005,
        MAX_EPOCHS = 100000,
        ALPHABET = alphabet
        )
    
    print(boolean)

    tests = ['portas_logicas/problemAND.csv', 'portas_logicas/problemOR.csv', 'portas_logicas/problemXOR.csv']

    # boolean.weights_loader('dumpsLogic/hidden_weights.dump', 'dumpsLogic/hidden_bias.dump', 'dumpsLogic/output_weights.dump', 'dumpsLogic/output_bias.dump')

    for test in tests:
        print(f'<<<<<<<<<< {test} >>>>>>>>>>')
        boolean.train(test)
        boolean.weights_dumper('dumpsLogic/hidden_weights.dump', 'dumpsLogic/hidden_bias.dump', 'dumpsLogic/output_weights.dump', 'dumpsLogic/output_bias.dump')
        # boolean.weights_loader('dumpsLogic/hidden_weights.dump', 'dumpsLogic/hidden_bias.dump', 'dumpsLogic/output_weights.dump', 'dumpsLogic/output_bias.dump')
        boolean.test(test)
        boolean.run(test)
