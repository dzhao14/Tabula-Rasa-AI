from mcts import MonteCarloTreeSearch
import game
from board import Board
import numpy as np
import random
import model

import ipdb

model_name = 'n4'
m = model.NN(existing = model_name) 
alpha_iters = 100
perfect_iters = 100

alphazero = MonteCarloTreeSearch(simulations = alpha_iters, model = m)
perfect = MonteCarloTreeSearch(simulations = perfect_iters)
vanilla = MonteCarloTreeSearch(simulations = alpha_iters)

print("Game 1")
g = game.Game()
g.board.print_pretty()
alpha_pi = np.array(alphazero.mcts(g))
perfect_pi = np.array(perfect.mcts(g))
vanilla_pi = np.array(vanilla.mcts(g))
print ("alpha_pi =\n {}".format(np.array(alpha_pi).reshape((3,3))))
print ("mcts_pi =\n {}".format(np.array(vanilla_pi).reshape((3,3))))
print("p =\n {}".format(np.array(m.predict_policy(g.board)).reshape((3,3))))
print("v = {}".format(m.predict_score(g.board)))

print("Game 2")
g = game.Game(starting_pos = np.array([0,0,0,0,-1,0,0,0,0]))
g.board.print_pretty()
alpha_pi = np.array(alphazero.mcts(g))
perfect_pi = np.array(perfect.mcts(g))
vanilla_pi = np.array(vanilla.mcts(g))
print ("alpha_pi =\n {}".format(np.array(alpha_pi).reshape((3,3))))
print ("mcts_pi =\n {}".format(np.array(vanilla_pi).reshape((3,3))))
print("p =\n {}".format(np.array(m.predict_policy(g.board)).reshape((3,3))))
print("v = {}".format(m.predict_score(g.board)))

print("Game 3")
g = game.Game(starting_pos = np.array([0,0,-1,0,1,0,1,-1,0]))
g.board.print_pretty()
alpha_pi = np.array(alphazero.mcts(g))
perfect_pi = np.array(perfect.mcts(g))
vanilla_pi = np.array(vanilla.mcts(g))
print ("alpha_pi =\n {}".format(np.array(alpha_pi).reshape((3,3))))
print ("mcts_pi =\n {}".format(np.array(vanilla_pi).reshape((3,3))))
print("p =\n {}".format(np.array(m.predict_policy(g.board)).reshape((3,3))))
print("v = {}".format(m.predict_score(g.board)))

print("Game 4")
g = game.Game(starting_pos = np.array([-1,0,-1,0,-1,1,0,1,0]))
g.board.print_pretty()
alpha_pi = np.array(alphazero.mcts(g))
perfect_pi = np.array(perfect.mcts(g))
vanilla_pi = np.array(vanilla.mcts(g))
print ("alpha_pi =\n {}".format(np.array(alpha_pi).reshape((3,3))))
print ("mcts_pi =\n {}".format(np.array(vanilla_pi).reshape((3,3))))
print("p =\n {}".format(np.array(m.predict_policy(g.board)).reshape((3,3))))
print("v = {}".format(m.predict_score(g.board)))


print("Game 5")
g = game.Game(starting_pos = np.array([-1,1,-1,1,1,-1,-1,-1,1]))
g.board.print_pretty()
alpha_pi = np.array(alphazero.mcts(g))
perfect_pi = np.array(perfect.mcts(g))
vanilla_pi = np.array(vanilla.mcts(g))
print ("alpha_pi =\n {}".format(np.array(alpha_pi).reshape((3,3))))
print ("mcts_pi =\n {}".format(np.array(vanilla_pi).reshape((3,3))))
print("p =\n {}".format(np.array(m.predict_policy(g.board)).reshape((3,3))))
print("v = {}".format(m.predict_score(g.board)))
