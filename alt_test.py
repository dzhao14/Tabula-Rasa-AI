import alt_mcts
import numpy as np

board = np.array([0,0,0,0,0,0,0,0,0])
mcts = alt_mcts.MonteCarlo(board)
mcts.get_play()
