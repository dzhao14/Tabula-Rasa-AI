import keras
import numpy

import ttt_util
import train_ttt


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
    print("Check if the AI knows the outcome of a finished game")
    won1 = numpy.array([0,-1,1,-1,1,1,-1,0,1])
    won2 = numpy.array([1,1,1,-1,1,-1,-1,-1,0])
    won3 = numpy.array([1,-1,-1,-1,1,1,-1-1,1])

    won = [won1, won2, won3]
   
    for done_board in won:
        ttt_util.print_pretty(done_board)
        print(model.predict(done_board))
        ttt_util.print_pretty(ttt_util.rotate_180(done_board))
        print(model.predict(ttt_util.rotate_180(done_board)))
        ttt_util.print_pretty(ttt_util.rotate_cc90(done_board))
        print(model.predict(ttt_util.rotate_cc90(done_board)))
        ttt_util.print_pretty(ttt_util.rotate_ccw90(done_board))
        print(model.redict(ttt_util.rotate_ccw90(done_board)))

