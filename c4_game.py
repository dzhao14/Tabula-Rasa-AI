import numpy

def print_pretty(board):
    """
    Print the tic tac toe board where the 1s are Xs and the -1s are Os
    """
    print(str(board).replace('-1', " O").replace('1', "X").replace('0', ' '))

def getStartBoard():
    return numpy.zeros((6,7))

def possible_move_indices(board):
    possible_moves = numpy.where(board[0,:]==0)[0]

def possible_moves(board, turn):
    """
    Given the board state as a (1,9) numpy array, where a 1 represents the
    AI's piece, a -1 represents the opponent's piece, and a 0 represents an
    open space return a (1,9) numpy array that contains the next move played

    We create all boards where the next move is played (for all possible next
    moves) and then we evaluate the positions from the opponent's point of view.
    We then choose the move that results in the worst evaluation for the
    opponent.

    turn = 1 indicates our turn, and turn = -1 indicates the opponent's turn
    """
    possible_moves = numpy.where(board[0,:]==0)[0]
    posibilities = numpy.zeros((len(possible_moves),6,7))
    for i, move_ind in enumerate(possible_moves):
        new_board = numpy.copy(board)

        depth = 0
        while depth < 5 and new_board[depth + 1, move_ind] == 0:
            depth = depth + 1

        new_board[depth, move_ind] = turn
        posibilities[i] = new_board

    return posibilities

def hasFourInRow(row):
    count = 0
    opponent_count = 0

    for i in range(0, len(row)):
        if row[i] == 1:
            count += 1
            opponent_count = 0
        elif row[i] == -1:
            opponent_count += 1
            count = 0
        else:
            opponent_count = 0
            count = 0

        if count > 3:
            return 1
        elif opponent_count > 3:
            return -1
    return 0

def getDiagonal(board, coord, direction):
    row = coord[0]
    col = coord[1]
    diag = []
    while row in range(0,6) and col in range(0,7):
        diag.append(board[row,col])
        row = row + direction[0]
        col = col + direction[1]
    return diag

def status(board):
    lines = []
    lines = lines + list(map(lambda r: board[r,:].tolist(), range(0,6)))
    lines = lines + list(map(lambda c: board[:,c].tolist(), range(0,7)))
    lines = lines + list(map(lambda c: getDiagonal(board, (0,c), (1,-1)),range(0,6)))
    lines = lines + list(map(lambda c: getDiagonal(board, (0,c), (1,1)),range(0,6)))
    lines = lines + list(map(lambda c: getDiagonal(board, (5,c), (-1,1)),range(0,6)))
    lines = lines + list(map(lambda c: getDiagonal(board, (5,c), (-1,-1)),range(0,6)))

    winner = 2
    for ln in lines:
        winner = hasFourInRow(list(ln))
        if winner != 0:
            return winner
    if boardFull(board):
        return 0
    return 2

def boardFull(board):
    return all([v != 0 for v in board.reshape(42)])

def permutations(board):
    return[ board, mirror_board(board), ]

def expandExample(board, policy):
    return (permutations(board),
            list(map(lambda x: x.reshape(9),
                permutations(policy.reshape((3,3))))))


if __name__ == "__main__":
    diagwin = numpy.array([[ 1.,  0.,  1.,  1.,  0.,  0.,  0.],
                      [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 0.,  0.,  0.,  1.,  0.,  0.,  0.],
                      [ 0.,  0.,  1.,  0.,  0.,  0.,  0.],
                      [ 0.,  1.,  0.,  0.,  0.,  0.,  0.],
                      [ 1.,  0.,  0.,  0.,  0.,  0.,  0.]])

    full = numpy.array([[ -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.],
                      [ -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.],
                      [ -1.,  -1.,  -1.,  1.,  -1.,  -1.,  -1.],
                      [ -1.,  -1.,  1.,  -1.,  -1.,  -1.,  -1.],
                      [ -1.,  1.,  -1.,  -1.,  -1.,  -1.,  -1.],
                      [ 1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.]])

    nowin = numpy.array([[  0,   0,   0,   0,   0,   0,   0,],
                         [  0,   0,   0,   0,   0,   0,   0,],
                         [  0,   0,   0,   0,   0,   0,   0,],
                         [ -1,   0,   0,   0,   0,   0,   0,],
                         [  1,   0,   1,   0,   1,   1,   0,],
                         [ -1,  -1,  -1,   1,  -1,   1,  -1,]])


    assert status(nowin) == 2
    assert status(diagwin) == 1
    assert boardFull(full)
