from mcts import MonteCarloTreeSearch
import game
from board import Board
import numpy as np
import random
import model

import ipdb

print ("type the filename of the model you want to use")
model_name = 'third'
m = model.NN(existing = model_name) 
alpha_iters = 100
perfect_iters = 100000

print("Game 1")
g = game.Game()
alphazero = MonteCarloTreeSearch(g, simulations = alpha_iters, model = m)
perfect = MonteCarloTreeSearch(g, simulations = perfect_iters)
vanilla = MonteCarloTreeSearch(g, simulations = alpha_iters)
alpha_pi = np.array(alphazero.mcts())
perfect_pi = np.array(perfect.mcts())
vanilla_pi = np.array(vanilla.mcts())
print ("alpha-vanilla distance: {}".format(np.linalg.norm(vanilla_pi - alpha_pi)))
print ("vanilla distance: {}".format(np.linalg.norm(vanilla_pi - perfect_pi)))
print ("alpha distance: {}".format(np.linalg.norm(alpha_pi - perfect_pi)))

