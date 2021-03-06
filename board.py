import numpy as np

class Board(object):
    """Represents a tic-tac-toe board """

    def __init__(self, starting_pos = None):
        if starting_pos is not None:
            self.board = starting_pos
        else:
            self.board = Board.getStartBoard()

    def get_board(self):
        return np.copy(self.board)

    def get_flipped_board(self):
        board = np.copy(self.board)
        board = np.negative(board)
        return board

    def print_pretty(self):
        """
        Print the tic tac toe board where the 1s are Xs and the -1s are Os
        """
        board = self.board.reshape((3,3))
        print(str(board).replace('-1', " O").replace('1', "X").replace('0', ' '))

    @staticmethod
    def getStartBoard():
        return np.array([0,0,0,0,0,0,0,0,0])

    def possible_moves_board(self, turn = True):
        """
        Given the board state as a (1,9) np array, where a 1 represents the
        AI's piece, a -1 represents the opponent's piece, and a 0 represents an
        open space return a (1,9) np array that contains the next move played

        We create all boards where the next move is played (for all possible next
        moves) and then we evaluate the positions from the opponent's point of view.
        We then choose the move that results in the worst evaluation for the
        opponent.

        turn = True indicates the next move will place an X, and turn = False 
        indicates the next move will place an O.
        """
        possible_moves = np.where(self.board==0)[0]
        posibilities = np.zeros((len(possible_moves), 9))
        for i, move_ind in enumerate(possible_moves):
            new_board = np.copy(self.board)
            new_board[move_ind] = turn
            posibilities[i] = new_board

        return posibilities

    def get_valid_move_index(self):
        return np.where(self.board==0)[0]

    @staticmethod
    def get_move_index(board, nextboard):
        """
        Given two boards, finds the first index where their values differ
        """
        for i in range(9):
            if board[i] != nextboard[i]:
                return i

    def make_move_index(self, index, x = True):
        """
        Place an X or O on to the specified square. If x = True an X is placed.
        If x = False an O is placed.

        Throws: An error if you try to move on a square that already is filled.
        """
        if self.board[index] == 1 or self.board[index] == -1:
            raise Exception("Trying to move on a square that already is filled")

        if x:
            self.board[index] = 1
        else:
            self.board[index] = -1

    def make_move_board(self, newboard):
        """
        Given the resulting board from the next move, set the board to the given
        board. Assumes the next move is placing an X.

        Throws: An error if the next board isn't valid from the current board.
        """
        if len(np.where(newboard==0)[0]) - len(np.where(self.board==0)[0]) != -1:
            raise Exception("Didn't make exactly one move")
        if sum(newboard) - sum(self.board) != 1:
            raise Exception("Next move either isn't an X")
        self.board = newboard

    def status(self):
        """
        Given a board as a (9,) np array determine who won
        1 for win, -1 for loss, 0 for tie, 2 for ongoing
        """
        board = np.copy(self.board)
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
        for i in range(0,9):
            if (board[i] == 0):
                return 2
        return 0;

    def flip_board(self):
        """
        Given a board flip it such that it is viewed with respect to the other
        player
        """
        board = np.copy(self.board)
        board = np.negative(board)
        self.board = board



def permute(board):
    return[
        board,
        rotate_180(board), rotate_cc90(board), 
        rotate_ccw90(board), 
        mirror_board(board),
        mirror_board(rotate_180(board)),
        mirror_board(rotate_cc90(board)),
        mirror_board(rotate_ccw90(board)),
        ]

def mirror_board(board):
    """
    Given a board, find the mirror images
    """
    board = board.reshape((3,3))
    return np.flip(board, 1).reshape((9,))

def rotate_180(board):
    """
    Given a board rotate it 180 degrees
    """
    board = board.reshape((3,3))
    return board[::-1].reshape((9,))

def rotate_cc90(board):
    """
    Returns the board rotated 90 degrees clockwise
    """
    board = board.reshape((3,3))
    rboard = np.array([board[:,0][::-1],board[:,1][::-1],board[:,2][::-1]])
    return rboard.reshape((9,))

def rotate_ccw90(board):
    """
    Returns the board rotated 90 degrees counter clockwise
    """
    board = board.reshape((3,3))
    rboard = np.array([board[:,2], board[:,1], board[:,0]])
    return rboard.reshape((9,))


class C4Board(object):

    def __init__(self, starting_pos = None):
        if starting_pos is not None:
            self.board = starting_pos
        else:
            self.board = Board.getStartBoard()

    @staticmethod
    def getStartBoard():
        return np.zeros(shape=(6,7))


