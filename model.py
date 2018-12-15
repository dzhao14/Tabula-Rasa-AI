import keras
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Flatten
from keras.models import Model
from keras.regularizers import l2
from keras.losses import mean_squared_error, binary_crossentropy
from keras import optimizers

from math import exp
import numpy
import board
import game

import ipdb

def createModel(config=None):

    board_input = Input(shape=(9,), name="main_input", dtype = 'float32')

    hidden1 = Dense(81, activation="relu")(board_input)
    bn1 = BatchNormalization(axis=1)(hidden1)

    hidden2 = Dense(81, activation="relu")(bn1)
    bn2 = BatchNormalization(axis=1)(hidden2)

    hidden3 = Dense(18, activation="relu")(bn2)
    hidden4 = Dense(18, activation="relu")(bn2)
    bn3 = BatchNormalization(axis=1)(hidden3)
    bn4 = BatchNormalization(axis=1)(hidden4)

    policy_vector = Dense(9, activation="relu")(bn3)
    softmax = Activation('softmax')(policy_vector)

    e = Dense(1)(bn4)
    e = Activation('tanh')(e)

    opt = keras.optimizers.Adam(lr = 0.2)
    model = Model(inputs=[board_input], outputs=[softmax, e])
    model.compile(
            opt,
            loss = ['categorical_crossentropy', 'mean_squared_error'],
            metrics=['accuracy'],
            )

    return model

class NN:
    def __init__(self, existing=None, config=None):
        if existing:
            self.nn = keras.models.load_model(existing)
        else:
            self.nn = createModel(config=config)

    def predict_policy(self, board):
        """ Takes in a Board object and returns p """
        board_ = board.get_board()
        board_ = board_.reshape((1, 9))
        policy, _ = self.nn.predict(board_)
        policy = policy[0].tolist()
        return policy

    def validate_policy(self, board, policy):
        if board.status() != 2:
            return []
        valid_indexes = board.get_valid_move_index()
        valid_policy = [policy[move_idx] for move_idx in valid_indexes]
        valid_policy = list(map(lambda x: exp(x), valid_policy))
        policy_sum = sum(valid_policy)
        valid_policy = list(map(lambda x: x/policy_sum, valid_policy))
        return valid_policy

    def predict_score(self, board):
        """ Takes in a Board object and returns v """
        board_ = board.get_board()
        board_ = board_.reshape((1,9))
        _, score = self.nn.predict(board_)
        return score[0][0]

    def train(self, inputs, policies, scores):
        """
        Inputs are a numpy array with the shape (x, 9)
        Policies are a numpy array with the shape (x, 9)
        Scores are a numpy array with the shape (x)
        """
        tb_cb = keras.callbacks.TensorBoard(
                log_dir='./Graph',
                histogram_freq=0,
                write_graph=True,
                write_images=True,
                )

        inputs, policies, scores = self.reshape_datas(inputs, policies, scores)
        self.nn.fit(
                inputs,
                [policies, scores],
                epochs = 1,
                batch_size = 256,
                )
                #callbacks = [tb_cb],
                #)

    def reshape_datas(self, inputs, policies, scores):
        return (inputs.reshape((len(inputs), 9)),
                policies.reshape((len(policies), 9)),
                scores.reshape((len(scores),)),
                )

    def save(self, filename):
        self.nn.save('models/{}'.format(filename))

