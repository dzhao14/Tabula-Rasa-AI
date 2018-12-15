from board import Board
import numpy as np

class Game(object):
    """
    Represents a game of tic-tac-toe.

    This is composed of a board and two players.
    Player 1 plays with X's (1s).
    Player -1 plays with O's (0s).

    At each point of the game, the board is flipped such that the player who 
    acts is always playing an X.
    """

    def __init__(self, starting_pos = None, player = 1):
        self.player = player
        self.board = Board(starting_pos = starting_pos)
        self.result = None

        self.check_result()

    def get_possible_moves(self):
        return self.board.possible_moves_board()

    def get_possible_move_indexes(self):
        return self.board.get_valid_move_index()

    def make_move_index(self, index):
        if not self.result:
            self.board.make_move_index(index)
            self.check_result()
            self.player = self.player * -1
            self.board.flip_board()
        else:
            raise Exception("Can't make move game is already over")

    def make_move(self, newboard):
        if not self.result:
            self.board.make_move_board(newboard)
            self.check_result()
            self.player = self.player * -1
            self.board.flip_board()
        else:
            raise Exception("Can't make move game is already over")

    def make_move_and_copy(self, newboard):
        newboard = np.negative(newboard)
        player = self.player * -1
        if not self.result:
            game = Game(starting_pos = newboard, player = player)
            return game
        else:
            raise Exception("Can't make move game is already over")

    def make_copy(self):
        return Game(starting_pos = self.board.get_board(), player = self.player)

    def check_result(self):
        """
        Check if the game is over.
        If the game is over set self.result to z.
        -1 if player -1  won
         0 if it was a   draw
         1 if player  1  won

        If the game is still on going do not mutate self.result.
        """
        if self.result is None:
            res = self.board.status()
            if res != 2:
                self.result = self.player * res
        else:
            raise Exception("Can't make move game is already over")

    def game_over(self):
        if self.result is not None:
            return True
        else:
            return False

    def get_result(self):
        return self.result if self.result else 2

