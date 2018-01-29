import keras
import train_ttt
import numpy


def random_move(board):
    """
    Given the board state as a (1,9) numpy array,
    return a (1,9) numpy array that contians the next move played
    """

    possible_moves = numpy.where(board==0)[0]
    move = numpy.random.randint(0, len(possible_moves))
    board[move] = -1
    return board


def make_move(board, AI):
    """
    Given the board state as a (1,9) numpy array,
    return a (1,9) numpy array that contains the next move played
    """

    possible_moves = numpy.where(board==0)[0]
    posibilities = numpy.zeros((len(possible_moves, 9)))
    for i, move_ind in enumerate(possible_moves):
        new_board = numpy.copy(board)
        new_board[move_ind] = 1
        posibilities[i] = new_board

    outcomes = AI.predict(posibilities, batch_size = len(posibilities))
    outcomes = numpy.negative(outcomes)
    best_board = numpy.where(outcomes=numpy.max(outcomes))
    return posibilities[best_board]


def main():

    AI = keras.models.load_model("")
    board = numpy.zeros(9)
    coin_flip = numpy.random.randint(0,2)

    if coin_flip == 0:
        

def test_knows_it_has_won(model):
    """
    Given a WON position see if the AI knows it has won
    """
    won1 = numpy.array([0,-1,1,-1,1,1,-1,0,1])
    won2 = numpy.array([1,1,1,-1,1,-1,-1,-1,0])
    won3 = numpy.array([1,-1,-1,-1,1,1,-1-1,1])

     
