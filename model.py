import keras
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Flatten
from keras.models import Model
from keras.regularizers import l2
from keras.losses import mean_squared_error, binary_crossentropy
from keras import optimizers

import numpy

import ipdb

def softmax(x):
    """Calucate the softmax of a vector"""
    e_x = numpy.exp(x - numpy.max(x))
    out = e_x / e_x.sum()
    return out

def possible_move_indexes(board):
    board_ = board.reshape(9)
    return numpy.where(board_==0)[0].tolist()
    
def createModel(config=None):

    board_input = Input(shape=(9,), name="main_input", dtype = 'float32')

    hidden1 = Dense(81, activation="relu")(board_input)
    bn = BatchNormalization(axis=1)(hidden1)

    hidden2 = Dense(81, activation="relu")(bn)
    bn = BatchNormalization(axis=1)(hidden2)

    hidden3 = Dense(18, activation="relu")(bn)
    hidden4 = Dense(18, activation="relu")(bn)
    bn1 = BatchNormalization(axis=1)(hidden3)
    bn2 = BatchNormalization(axis=1)(hidden4)

    policy_vector = Dense(9, activation="relu")(bn1)
    softmax = Activation('softmax')(policy_vector)

    e = Dense(1)(bn2)
    e = Activation('tanh')(e)

    opt = keras.optimizers.Adam()
    model = Model(inputs=[board_input], outputs=[softmax, e])
    model.compile(opt,
            loss = ['binary_crossentropy', 'mean_squared_error'],
            metrics=['accuracy'])
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
        inputs, policies, scores = self.reshape_datas(inputs, policies, scores)
        self.nn.fit(
                inputs,
                [policies, scores],
                epochs = 1,
                batch_size = 256,
                )

    def reshape_datas(self, inputs, policies, scores):
        return (inputs.reshape((len(inputs), 9)),
                policies.reshape((len(policies), 9)),
                scores.reshape((len(scores),)),
                )

    def save(self, filename):
        self.nn.save(filename)

