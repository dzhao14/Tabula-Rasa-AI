import mcts
import numpy as np
import ttt_game
import sys
import random

searcher = mcts.MonteCarloTreeSearch(1,2,3)
board = np.array([0,0,0,0,0,0,0,0,0])
"""
print(ttt_game.print_pretty(board))
while ttt_game.status(board == 2):
  move = input("where to move?")
  board[move] = -1
  if ttt_game.status(board) != 2:
      ttt_game.print_pretty(board)
      break
  board = searcher.findNextMove(board,1)
  print(ttt_game.print_pretty(board))
  if ttt_game.status(board) != 2:
      ttt_game.print_pretty(board)
      break
"""

wins = 0
losses = 0
ties = 0
for i in range(0,20):
    board = np.array([0,0,0,0,0,0,0,0,0])
    while ttt_game.status(board == 2):
        moves = ttt_game.possible_moves(board, -1)
        board = random.choice(moves)
        if ttt_game.status(board) != 2:
            ttt_game.print_pretty(board)
            break
        board = searcher.findNextMove(board,1)
        if ttt_game.status(board) != 2:
            ttt_game.print_pretty(board)
            break
    print("winner: " + str(ttt_game.status(board)))
    if ttt_game.status(board) == 1:
        wins = wins + 1
    if ttt_game.status(board) == -1:
        losses = losses + 1
    if ttt_game.status(board) == 0:
        ties = ties + 1
print( "wins" + str(wins), "losses" + str(losses), "ties" + str(ties))

import pdb; pdb.set_trace()
