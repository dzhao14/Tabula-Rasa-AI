import numpy

def print_pretty(board):
    """
    Print the tic tac toe board where the 1s are Xs and the -1s are Os
    """
    board = board.reshape((3,3))
    print(str(board).replace('-1', " O").replace('1', "X").replace('0', ' '))

def make_pretty(board):
    """
    Return the tic tac toe board where the 1s are Xs and the -1s are Os
    """
    board = board.reshape((3,3))
    return str(board).replace('-1', " O").replace('1', "X").replace('0', ' ')

def make_move(board, AI, flipped=False):
    """
    Given the board state as a (1,9) numpy array, where a 1 represents the
    AI's piece, a -1 represents the opponent's piece, and a 0 represents an
    open space return a (1,9) numpy array that contains the next move played

    We create all boards where the next move is played (for all possible next
    moves) and then we evaluate the positions from the opponent's point of view.
    We then choose the move that results in the worst evaluation for the
    opponent.
    """
    possible_moves = numpy.where(board==0)[0]
    posibilities = numpy.zeros((len(possible_moves), 9))
    for i, move_ind in enumerate(possible_moves):
        new_board = numpy.copy(board)
        new_board[move_ind] = 1
        new_board = numpy.negative(new_board)
        posibilities[i] = new_board

    outcomes = AI.predict(posibilities, batch_size = len(posibilities))
    best_board = numpy.where(outcomes==numpy.min(outcomes))[0][0]
    return (numpy.negative(posibilities[best_board]), possible_moves[best_board])


def win(board):
    """
    Given a board as a (9,) numpy array determine who won
    """
    if (1 == board[0] and 1 == board[1] and 1 == board[2]
            or 1 == board[3] and 1 == board[4] and 1 == board[5]
            or 1 == board[6] and 1 == board[7] and 1 == board[8]
            or 1 == board[0] and 1 == board[3] and 1 == board[6]
            or 1 == board[1] and 1 == board[4] and 1 == board[7]
            or 1 == board[2] and 1 == board[5] and 1 == board[8]
            or 1 == board[0] and 1 == board[4] and 1 == board[8]
            or 1 == board[2] and 1 == board[4] and 1 == board[6]):
        return 1;
    if (-1 == board[0] and -1 == board[1] and -1 == board[2]
            or -1 == board[3] and -1 == board[4] and -1 == board[5]
            or -1 == board[6] and -1 == board[7] and -1 == board[8]
            or -1 == board[0] and -1 == board[3] and -1 == board[6]
            or -1 == board[1] and -1 == board[4] and -1 == board[7]
            or -1 == board[2] and -1 == board[5] and -1 == board[8]
            or -1 == board[0] and -1 == board[4] and -1 == board[8]
            or -1 == board[2] and -1 == board[4] and -1 == board[6]):
        return -1;
    for i in range(0,9) :
        if (board[i] == 0):
            return "ongoing"
    return 0;

def flip_board(board):
    """
    Given a board flip it such that it is viewed with respect to the other
    player
    """
    return numpy.negative(board)

def rotate_180(board):
    """
    Given a board flip it 180 degrees
    """
    return board[::-1]

def rotate_cc90(board):
    """
    Returns the board rotated 90 degrees clockwise
    """
    board = board.reshape((3,3))
    rboard = numpy.array([board[:,0][::-1],board[:,1][::-1],board[:,2][::-1]])
    return rboard.reshape((9,))

def rotate_ccw90(board):
    """
    Returns the board rotated 90 degrees counter clockwise
    """
    board = board.reshape((3,3))
    rboard = numpy.array([board[:,2], board[:,1], board[:,0]])
    return rboard.reshape((9,))
