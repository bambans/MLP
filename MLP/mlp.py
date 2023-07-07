from .files import read_input, load, dump as save_weights

from .train import MLP_train
from .test import MLP_test
from .run import MLP_run

class MLP:
    """
    A classe MLP cria uma intância passível de treino, teste e execução de uma rede neural implementada com neurônios do tipo Perceptron e função Sigmóide.
    """
    def __init__(self, INPUT_LAYER_SIZE = 63, HIDDEN_LAYER_SIZE = 35, OUTPUT_LAYER_SIZE = 7, MAXIMUM_ERROR = 0.0625, LEARNING_RATE = 0.01, MAX_EPOCHS = 100000, ALPHABET = None, OUTFOLDER = 'dumps/'):
        ### Neural network hyperparameters:
        self.INPUT_LAYER_SIZE = INPUT_LAYER_SIZE
        self.HIDDEN_LAYER_SIZE = HIDDEN_LAYER_SIZE
        self.OUTPUT_LAYER_SIZE = OUTPUT_LAYER_SIZE

        ### Training parameters:
        self.MAXIMUM_ERROR = MAXIMUM_ERROR
        self.LEARNING_RATE = LEARNING_RATE
        self.MAX_EPOCHS = MAX_EPOCHS

        ### MLP weights
        self.__HIDDEN_WEIGHTS__ = None
        self.__HIDDEN_BIAS___ = None
        self.__OUTPUT_WEIGHTS__ = None
        self.__OUTPUT_BIAS__ = None
        
        ### AUX
        self.__OUTFOLDER__ = OUTFOLDER

        ### MLP problem domain definition
        self.ALPHABET = ALPHABET

    def __str__(self) -> str:
        return f"""

\tMLP Hyperparameters
--------------------------------------------
\tINPUT LAYER SIZE: \t{self.INPUT_LAYER_SIZE}
\tHIDDEN LAYER SIZE: \t{self.HIDDEN_LAYER_SIZE}
\tOUTPUT LAYER SIZE: \t{self.OUTPUT_LAYER_SIZE}
\tMAXIMUM ERROR: \t\t{self.MAXIMUM_ERROR}
\tLEARNING RATE: \t\t{self.LEARNING_RATE}
\tMAXIMUM EPOCHS: \t{self.MAX_EPOCHS}
--------------------------------------------
"""

    def train(self, train_file_path):
        """
        Executa o treinamento com os dados de entrada e a saída esperada para cada caso.
        Ao final do treinamento, são gerados os arquivos de dump com os pesos da rede treinada, onde a ordem é tal que:
        - (1) Arquivo com os pesos da camada escondida;
        - (2) Arquivo com os viéses da camada escondida;
        - (3) Arquivo com os pesos da camada de saída;
        - (4) Arquivo com os viéses da camada de saída.
        """
        print("\nMLP training...\n")

        characters_input, expected_output = read_input(self, train_file_path)
        self.__HIDDEN_WEIGHTS__, self.__HIDDEN_BIAS___, self.__OUTPUT_WEIGHTS__, self.__OUTPUT_BIAS__ = MLP_train(self, characters_input, expected_output)

    def test(self, test_file_path = None, dumps = None):
        """
        Executa um feed forward com os dados de entrada, dados os dumps de treinamento (i.e., os pesos da rede treinada) e os compara com a saída esperada.
        O arquivo de entrada DEVE conter as saídas espedadas.
        Os dumps são dados em uma lista, onde a ordem é tal que:
        - (1) Arquivo com os pesos da camada escondida;
        - (2) Arquivo com os viéses da camada escondida;
        - (3) Arquivo com os pesos da camada de saída;
        - (4) Arquivo com os viéses da camada de saída.
        """
        print('\nMLP testinng...\n')

        if test_file_path is not None:
            dumps_loading = []
            if dumps is not None:
                dumps_loading = [load(dumps[0]), load(dumps[1]), load(dumps[2]), load(dumps[3])]
            elif (
                self.__HIDDEN_WEIGHTS__ is not None and
                self.__HIDDEN_BIAS___ is not None and
                self.__OUTPUT_WEIGHTS__ is not None and
                self.__OUTPUT_BIAS__ is not None
            ):
                dumps_loading = [self.__HIDDEN_WEIGHTS__, self.__HIDDEN_BIAS___, self.__OUTPUT_WEIGHTS__, self.__OUTPUT_BIAS__]
            else:
                print(f'Os pesos da rede são necessários!')

            characters_input, expected_output = read_input(self, test_file_path)
            MLP_test(characters_input, expected_output, dumps_loading, self.ALPHABET, test_file_path)

        else:
            print(f'Um arquivo de entradas é necessário!')

    def run(self, input_file_path = None, dumps = None):
        """
        Executa um feed forward com os dados de entrada, dados os dumps de treinamento (i.e., os pesos da rede treinada).
        Os dumps são dados em uma lista, onde a ordem é tal que:
        - (1) Arquivo com os pesos da camada escondida;
        - (2) Arquivo com os viéses da camada escondida;
        - (3) Arquivo com os pesos da camada de saída;
        - (4) Arquivo com os viéses da camada de saída.
        """
        print('\nMLP running...\n')

        if input_file_path is not None:
            dumps_loading = []
            if dumps is not None:
                dumps_loading = [load(dumps[0]), load(dumps[1]), load(dumps[2]), load(dumps[3])]
            elif (
                self.__HIDDEN_WEIGHTS__ is not None and
                self.__HIDDEN_BIAS___ is not None and
                self.__OUTPUT_WEIGHTS__ is not None and
                self.__OUTPUT_BIAS__ is not None
            ):
                dumps_loading = [self.__HIDDEN_WEIGHTS__, self.__HIDDEN_BIAS___, self.__OUTPUT_WEIGHTS__, self.__OUTPUT_BIAS__]
            else:
                print(f'Os pesos da rede são necessários!')

            characters_input, _ = read_input(self, input_file_path)

            MLP_run(characters_input, dumps_loading, self.ALPHABET)

        else:
            print(f'Um arquivo de entradas é necessário!')

    def weights_loader(self, hidden_weights_path = 'dumps/hidden_weights.dump', hidden_biases_path = 'dumps/hidden_bias.dump', output_weights_path = 'dumps/output_weights.dump', output_biases_path =  'dumps/output_bias.dump'):
        """
        Lê os arquivos de Dump dos pesos extraídos da rede em treinamento.
        A ordem é tal que:
        - (1) Arquivo com os pesos da camada escondida;
        - (2) Arquivo com os viéses da camada escondida;
        - (3) Arquivo com os pesos da camada de saída;
        - (4) Arquivo com os viéses da camada de saída.
        """
        try:
            self.__HIDDEN_WEIGHTS__ = load(hidden_weights_path)
            self.__HIDDEN_BIAS___ = load(hidden_biases_path)
            self.__OUTPUT_WEIGHTS__ = load(output_weights_path)
            self.__OUTPUT_BIAS__ = load(output_biases_path)
            return True
        except Exception as error:
            print(f'Não foi possível ler os dumps! Erro: {error}')
            return False

    def weights_dumper(self, hidden_weights_path = 'dumps/hidden_weights.dump', hidden_biases_path = 'dumps/hidden_bias.dump', output_weights_path = 'dumps/output_weights.dump', output_biases_path =  'dumps/output_bias.dump'):
        """
        Exporta os arquivos com o Dump dos pesos extraídos da rede em treinamento.
        A ordem é tal que:
        - (1) Arquivo com os pesos da camada escondida;
        - (2) Arquivo com os viéses da camada escondida;
        - (3) Arquivo com os pesos da camada de saída;
        - (4) Arquivo com os viéses da camada de saída.
        """
        try:
            save_weights(hidden_weights_path, self.__HIDDEN_WEIGHTS__)
            save_weights(hidden_biases_path, self.__HIDDEN_BIAS___)
            save_weights(output_weights_path, self.__OUTPUT_WEIGHTS__)
            save_weights(output_biases_path, self.__OUTPUT_BIAS__)
        except Exception as error:
            print(f'Não foi possível salvar os pesos da rede nos arquivos! Erro: {error}')

