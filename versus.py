from mcts import MonteCarloTreeSearch
import game
from board import Board
import numpy as np
import random
import model

import ipdb

print ("type the filename of the model you want to use")
model_name = input()
m = model.NN(existing = model_name) 
print ("type the number of games each AI should play going first")
games = int(input())
print ("number of iterations alphazero can search for")
alpha_iters = int(input())
print("number of iterations the vanilla mcts can search for")
vanilla_iters = int(input())

vanilla = 0
alphazero = 0
total = 0

for i in range(2*games):
    if i >= games:
        v = False
    else:
        v = True
    g = game.Game()
    while not g.game_over():
        if v:
            mcts = MonteCarloTreeSearch(g, simulations = vanilla_iters)
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
        else:
            mcts = MonteCarloTreeSearch(g, simulations = alpha_iters, model = m)
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

        v = not v
        g.player = 1

    g.board.print_pretty()
    if v and g.result != 0:
        alphazero += 1

    if not v and g.result != 0:
        vanilla += 1

    total+=1

print("---Stats---")
print("alpha zero winrate: {}".format(alphazero / total))
print("vanilla winrate: {}".format(vanilla / total))
print("draw rate: {}".format((total - alphazero - vanilla) / total)) 

