from mcts import MonteCarloTreeSearch
import game
from board import Board
import numpy as np
import random


print ("How many iterations can the mcts AI think for?")
iters = int(input())
print ("Hit 1 if you want to go first or 2 if you want to go second")
turn = int(input())


user = True if turn == 1 else False
g = game.Game(player = -1) if turn == 1 else game.Game()

while not g.game_over():
    if user:
        g.board.print_pretty()
        ind = int(input())
        g.make_move_index(ind)
    else:
        mcts = MonteCarloTreeSearch(g, simulations = iters)
        pi = mcts.mcts()
        best_prob = 0
        best_moves = []
        for i, prob in enumerate(pi):
            if prob > best_prob:
                best_prob = prob
                best_moves = [i]
            elif prob == best_prob:
                best_moves.append(i)
            else:
                continue

        g.make_move_index(random.choice(best_moves))
    user = not user

g.board.print_pretty()
if g.result == 0:
    print("Draw!")

if user and g.result != 0:
    print("AI won...")

if not user and g.result != 0:
    print("You won!")


