import keras
from keras.models import Sequential
from keras.layers import Dense
import re
import numpy as np


class NeuralNet:
    def __init__(self, weights_dir):
        self.weights = weights_dir
        self.model = None
        self.load_model()
        self.to_move = {'w': 0, 'b': 1}
        self.notation_dict = self.get_notation()

    @staticmethod
    def get_notation():
        result = {}
        notation = list('eprnbqkPRNBQK')
        i = 0
        for piece in notation:
            result[piece] = i
            i += 1
        return result

    @staticmethod
    def _build_ann(input_dim, num_classes):
        model = Sequential()
        model.add(Dense(units=80, activation="relu", input_dim=input_dim, kernel_initializer="normal"))
        model.add(Dense(units=120, activation="relu", kernel_initializer='normal'))
        model.add(Dense(units=80, activation="relu", kernel_initializer='normal'))
        model.add(Dense(units=num_classes, activation="softmax", kernel_initializer='normal'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        return model

    def load_model(self):
        model = self._build_ann(65, 3)
        model.load_weights(self.weights)
        self.model = model

    def parse_fen(self, board_string):
        fen_string = board_string.split(' ')
        # Get the next player to move
        to_move = self.to_move[fen_string[1]]
        board_string = fen_string[0]
        numbers = list(set(re.findall('\d', board_string)))
        for number in numbers:
            board_string = board_string.replace(str(number), 'e' * int(number))
        board_string = board_string.replace('/', '')
        final_string = [self.notation_dict[x] for x in list(board_string)]
        final_string.append(to_move)
        numpy_final = np.array(final_string)
        return numpy_final

    def predict(self, obs):
        formatted_fen = self.parse_fen(obs)
        predictions = self.model.predict_classes(formatted_fen)
        return predictions[0]
