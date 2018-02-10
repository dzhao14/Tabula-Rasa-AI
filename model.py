import keras
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Flatten
from keras.models import Model
from keras.regularizers import l2
from keras.losses import mean_squared_error, categorical_crossentropy

import numpy

def softmax(x):
    e_x = numpy.exp(x - numpy.max(x))
    out = e_x / e_x.sum()
    return out

def possible_move_indexes(board):
    if board.shape == (3,3):
        board_ = board.reshape((9,))
    else:
        board_ = board

    idx = numpy.argwhere(board_)
    return idx.reshape(len(idx)).tolist()
    
def alphazero_loss(y_true, y_pred):
    policy_vector_true = y_true[0]
    policy_vector_pred = y_pred[0]
    evaluation_true = y_true[1]
    evaluation_pred = y_pred[1]
    return sum((mean_squared_error(evaluation_true, evaluation_pred),
            categorical_crossentropy(policy_vector_true, policy_vector_pred)))

def createModel():
    board_input = Input(shape=(1,3,3), name="main_input", dtype = 'float32')

    conv1 = Conv2D(1,
            kernel_size=(3,3),
            data_format="channels_first",
            padding="same",
            kernel_regularizer=l2(0.2))(board_input)

    bn = BatchNormalization(axis=1)(conv1)

    activation = Activation('relu')(bn)

    conv2 = Conv2D(1,
            kernel_size=(3,3),
            data_format="channels_first",
            padding="same",
            kernel_regularizer=l2(0.2))(activation)

    bn = BatchNormalization(axis=1)(conv2)

    activation = Activation('relu')(bn)

    conv1a = Conv2D(1,
            kernel_size=(3,3),
            data_format="channels_first",
            padding="same",
            kernel_regularizer=l2(0.2))(activation)

    conv1b = Conv2D(1,
            kernel_size=(3,3),
            data_format="channels_first",
            padding="same",
            kernel_regularizer=l2(0.2))(activation)

    bn = BatchNormalization(axis=1)(conv1a)
    activation = Activation('relu')(bn)

    bn = BatchNormalization(axis=1)(conv1b)
    activation = Activation('relu')(bn)

    conv2a = Conv2D(1,
            kernel_size=(3,3),
            data_format="channels_first",
            padding="same",
            kernel_regularizer=l2(0.2))(activation)

    conv2b = Conv2D(1,
            kernel_size=(3,3),
            data_format="channels_first",
            padding="same",
            kernel_regularizer=l2(0.2))(activation)

    bn = BatchNormalization(axis=1)(conv2a)
    activation = Activation('relu')(bn)
    flat1 = Flatten()(activation)

    bn = BatchNormalization(axis=1)(conv2b)
    activation = Activation('relu')(bn)
    flat2 = Flatten()(activation)

    policy_vector = Dense(9, activation="relu")(flat1)
    e = Dense(1)(flat2)
    model = Model(inputs=[board_input], outputs=[policy_vector, e])
    model.compile(optimizer='sgd',
            loss = alphazero_loss)
    return model

class NN:
    def valid_policy(self, policy, board):
        board_ = board.reshape(9)
        p_moves = possible_move_indexes(board)
        new_policy = []
        for idx in p_moves:
            new_policy.append(board[idx])
        new_policy = numpy.array(new_policy)
        new_policy.dtype = "float32"
        return softmax(new_policy)


    def __init__(self):
        self.nn = createModel()


#policy = model.predict(numpy.array([0,0,0,0,0,0,0,0,0]).reshape((1,1,3,3)))[0]
#print(policy)
#print(valid_policy(policy, numpy.array([1,1,1,0,0,0,0,0,0])))








