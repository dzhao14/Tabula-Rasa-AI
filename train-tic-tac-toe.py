import keras
import ipdb
from keras import Sequential
from keras.layers import Dense, Activation
from keras import initializers
from keras import regularizers

import numpy

import ttt_util

TRAINING_GAMES = 10000
BATCH_SIZE = 32
LEARNING_RATE = 0.002

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(9,), 
    kernel_regularizer=regularizers.l2(0.1)))
model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
model.add(Dense(1))
opt = keras.optimizers.SGD(lr=LEARNING_RATE)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

for i in range(TRAINING_GAMES):
    seen_boards1 = []
    seen_boards2 = []
    c = 0
    board = numpy.zeros(9)
    while ttt_util.win(board) == "ongoing":
        if c%2 == 0:
            seen_boards1.append(board)
            board, move = ttt_util.make_move(board, model)
        else:
            board = numpy.negative(board)
            seen_boards2.append(board)
            board, move = ttt_util.make_move(board, model)
            board = numpy.negative(board)
        c += 1

    result = ttt_util.win(board)
    data1 = numpy.array(seen_boards1)
    data2 = numpy.array(seen_boards2)
    if ttt_util.win(board) == 1:
        labels1 = numpy.ones(len(seen_boards1))
        labels2 = numpy.negative(numpy.ones(len(seen_boards2)))
    elif ttt_util.win(board) == 0:
        labels1 = numpy.zeros(len(seen_boards1))
        labels2 = numpy.zeros(len(seen_boards2))
    else:
        labels1 = numpy.negative(numpy.ones(len(seen_boards2)))
        labels2 = numpy.ones(len(seen_boards1))

    data = numpy.vstack((data1, data2))
    labels = numpy.hstack((labels1, labels2))

    model.fit(data, labels, batch_size=BATCH_SIZE)

ipdb.set_trace()
#model.save("lol")

