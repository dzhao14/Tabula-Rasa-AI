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


def test_knows_it_has_won(model):
    """
    Given a WON position see if the AI knows it has won
    """
    won1 = numpy.array([0,-1,1,-1,1,1,-1,0,1])
    won2 = numpy.array([1,1,1,-1,1,-1,-1,-1,0])
    won3 = numpy.array([1,-1,-1,-1,1,1,-1-1,1])

     
