import keras
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Flatten
from keras.models import Model
from keras.regularizers import l2
from keras.losses import mean_squared_error, binary_crossentropy
from keras import optimizers

import numpy

def softmax(x):
    """Calucate the softmax of a vector"""
    e_x = numpy.exp(x - numpy.max(x))
    out = e_x / e_x.sum()
    return out

def possible_move_indexes(board):
    board_ = board.reshape(9)
    return numpy.where(board_==0)[0].tolist()
    
def alphazero_loss(y_true, y_pred):
    #TODO: is it actually categorical_crossentropy?
    policy_vector_true = y_true[0]
    policy_vector_pred = y_pred[0]
    evaluation_true = y_true[1]
    evaluation_pred = y_pred[1]
    return sum((mean_squared_error(evaluation_true, evaluation_pred),
            mean_squared_error(policy_vector_true, policy_vector_pred)))

def createModel(config=None):
    #TODO: add config to make changing hyper params easier
    kernel_reg = 0.0
    filters = 100
    kernel = (3,3)
    board_input = Input(shape=(1,3,3), name="main_input", dtype = 'float32')

    conv1 = Conv2D(filters,
            kernel_size=kernel,
            data_format="channels_first",
            padding="same",
            kernel_regularizer=l2(kernel_reg))(board_input)

    bn = BatchNormalization(axis=1)(conv1)

    activation = Activation('relu')(bn)

    conv2 = Conv2D(filters,
            kernel_size=kernel,
            data_format="channels_first",
            padding="same",
            kernel_regularizer=l2(kernel_reg))(activation)

    bn = BatchNormalization(axis=1)(conv2)

    activation = Activation('relu')(bn)

    conv1a = Conv2D(filters,
            kernel_size=kernel,
            data_format="channels_first",
            padding="same",
            kernel_regularizer=l2(kernel_reg))(activation)

    conv1b = Conv2D(filters,
            kernel_size=kernel,
            data_format="channels_first",
            padding="same",
            kernel_regularizer=l2(kernel_reg))(activation)

    bn = BatchNormalization(axis=1)(conv1a)
    activation = Activation('relu')(bn)

    bn = BatchNormalization(axis=1)(conv1b)
    activation = Activation('relu')(bn)

    conv2a = Conv2D(filters,
            kernel_size=kernel,
            data_format="channels_first",
            padding="same",
            kernel_regularizer=l2(kernel_reg))(activation)

    conv2b = Conv2D(filters,
            kernel_size=kernel,
            data_format="channels_first",
            padding="same",
            kernel_regularizer=l2(kernel_reg))(activation)

    bn = BatchNormalization(axis=1)(conv2a)
    activation = Activation('relu')(bn)
    flat1 = Flatten()(activation)

    bn = BatchNormalization(axis=1)(conv2b)
    activation = Activation('relu')(bn)
    flat2 = Flatten()(activation)

    policy_vector = Dense(9, activation="relu")(flat1)
    softmax = Activation('softmax')(policy_vector)
    e = Dense(1)(flat2)
    e = Activation('tanh')(e)

    opt = keras.optimizers.Adam(lr=0.00001)
    model = Model(inputs=[board_input], outputs=[softmax, e, policy_vector])
    model.compile(opt,
            loss = alphazero_loss,
            metrics=['accuracy'])
    return model

class NN:
    def __init__(self, existing=None, config=None):
        if existing:
            self.nn = keras.models.load_model(existing)
        else:
            self.nn = createModel(config=config)

    def valid_policy(self, policy, board):
        board_ = board.reshape(9)
        p_moves = possible_move_indexes(board_)
        if len(p_moves) == 0:
            return numpy.array([])
        new_policy = []
        for idx in p_moves:
            new_policy.append(policy[idx])
        new_policy = numpy.array(new_policy)
        new_policy = new_policy.astype("float32")
        return softmax(new_policy)

    def predict_policy(self, board):
        board_ = board.reshape((1,1,3,3))
        _, __, policy = self.nn.predict(board_)
        policy = policy[0].tolist()
        return self.valid_policy(policy, board_)

    def predict_score(self, board):
        board_ = board.reshape((1,1,3,3))
        _, score, __ = self.nn.predict(board_)
        return score[0][0]

    def train(self, inputs, policies, scores):
        """
        Inputs are a numpy array with the shape (x, 1, 3, 3)
        Policies are a numpy array with the shape (x, 9)
        Scores are a numpy array with the shape (x)
        """
        self.nn.fit(inputs.reshape((len(inputs), 1, 3, 3)), [policies, scores,
            policies], epochs = 10, batch_size = 128)

    def save(self, filename):
        self.nn.save(filename)

