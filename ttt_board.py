import numpy
class Board(object):
    def start(self):
        # Returns a representation of the starting state of the game.
        return (1,np.array([0,0,0,0,0,0,0,0,0]))
        pass

    def current_player(self, state):
        # Takes the game state and returns the current player's
        # number.
        return state[0]
        pass

    def next_state(self, state, play):
        # Takes the game state, and the move to be applied.
        # Returns the new game state.
	board = state[1]
	player = state[0]
        board[play[1]] = play[0]
	return (3 - player, board)
	
        pass

    def legal_plays(self, state_history):
        # Takes a sequence of game states representing the full
        # game history, and returns the full list of moves that
        # are legal plays for the current player.
        board = state_history[-1][1]
        player = state_history[-1][0]

        possible_moves = numpy.where(board==0)[0]
        posibilities = numpy.zeros((len(possible_moves), 2))
        for i, move_ind in enumerate(possible_moves):
            posibilities[i] = (3 - player, move_ind)
        pass
	return posibilities

    def winner(self, state_history):
        # Takes a sequence of game states representing the full
        # game history.  If the game is now won, return the player
        # number.  If the game is still ongoing, return zero.  If
        # the game is tied, return a different distinct value, e.g. -1.
	board = state_history[-1][1]

	if (1 == board[0] and 1 == board[1] and 1 == board[2]
		or 1 == board[3] and 1 == board[4] and 1 == board[5]
		or 1 == board[6] and 1 == board[7] and 1 == board[8]
		or 1 == board[0] and 1 == board[3] and 1 == board[6]
		or 1 == board[1] and 1 == board[4] and 1 == board[7]
		or 1 == board[2] and 1 == board[5] and 1 == board[8]
		or 1 == board[0] and 1 == board[4] and 1 == board[8]
		or 1 == board[2] and 1 == board[4] and 1 == board[6]):
	    return 1;
	if (2 == board[0] and 2 == board[1] and 2 == board[2]
		or 2 == board[3] and 2 == board[4] and 2 == board[5]
		or 2 == board[6] and 2 == board[7] and 2 == board[8]
		or 2 == board[0] and 2 == board[3] and 2 == board[6]
		or 2 == board[1] and 2 == board[4] and 2 == board[7]
		or 2 == board[2] and 2 == board[5] and 2 == board[8]
		or 2 == board[0] and 2 == board[4] and 2 == board[8]
		or 2 == board[2] and 2 == board[4] and 2 == board[6]):
	    return 2;
	for i in range(0,9) :
	    if (board[i] == 0):
		return -1
	return 0;

        pass
