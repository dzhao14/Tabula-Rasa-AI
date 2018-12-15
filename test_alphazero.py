from mcts import MonteCarloTreeSearch
import game
from board import Board
import numpy as np
import random
import model

import ipdb

model_name = 'models/m1'
m = model.NN(existing = model_name) 
alpha_iters = 100
perfect_iters = 1000

alphazero = MonteCarloTreeSearch(simulations = alpha_iters, model = m)
perfect = MonteCarloTreeSearch(simulations = perfect_iters)
vanilla = MonteCarloTreeSearch(simulations = alpha_iters)

def test_alphazero(g):
    g.board.print_pretty()
    alpha_pi = np.array(alphazero.mcts(g))
    perfect_pi = np.array(perfect.mcts(g))
    vanilla_pi = np.array(vanilla.mcts(g))
    print ("alpha_pi =\n {}".format(np.array(alpha_pi).reshape((3,3))))
    print("p =\n {}".format(np.array(m.predict_policy(g.board)).reshape((3,3))))
    print("v = {}".format(m.predict_score(g.board)))


print("Testing obvious draw")
draw_pos = []
draw_pos.append(np.array([0, 1, -1, -1, -1, 1, 1, 1, -1]))
draw_pos.append(np.array([1, 0, -1, -1, -1, 1, 1, -1, 1]))
draw_pos.append(np.array([1, -1, -1, -1, 0, 1, 1, 1, -1]))

for pos in draw_pos:
    g = game.Game(starting_pos = pos)
    test_alphazero(g)

print("Testing obvious win")
win_pos = []
win_pos.append(np.array([0, 1, 1, -1, -1, 1, -1, 1, -1]))
win_pos.append(np.array([1, 0, 1, -1, -1, 1, -1, 1, -1]))
win_pos.append(np.array([1, -1, 1, -1, 0, -1, -1, 1, 1]))

for pos in win_pos:
    g = game.Game(starting_pos = pos)
    test_alphazero(g)

print("Testing obvious loss")
loss_pos = []
loss_pos.append(np.array([-1, -1, 0, -1, 1, 1, 0, 1, -1]))
loss_pos.append(np.array([-1, 0, -1, 1, -1, 1, 0, 0, 0]))
loss_pos.append(np.array([-1, 1, -1, 1, -1, 1, 0, 0, 0]))

for pos in loss_pos:
    g = game.Game(starting_pos = pos)
    test_alphazero(g)
