from mcts import MonteCarloTreeSearch
import game
from board import Board
import numpy as np
import random

import ipdb

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
        mcts = MonteCarloTreeSearch(g)
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
        p1_images, p1_pi, p2_images, p2_pi = mcts.get_training_data()
        for i in range(len(p2_images)):
            b = Board(starting_pos = p2_images[i])
            b.print_pretty()
            print (p2_pi[i])
        g.make_move_index(random.choice(best_moves))
    user = not user

g.board.print_pretty()
if g.result == 0:
    print("Draw!")

if user and g.result != 0:
    print("AI won...")

if not user and g.result != 0:
    print("You won!")


