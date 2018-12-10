import numpy as np
import game
import mcts
import ipdb

print("Test game 1")
g = game.Game(np.array([1,1,0,-1,-1,0,-1,1,-1]))
pi, root = mcts.mcts_debug(g, 10000)
assert pi[2] == 1 and pi[5] == 0
root.game.board.print_pretty()
print(pi)


print("Test game 2")
g = game.Game(np.array([1,1,0,-1,-1,0,1,-1,-1]))
pi, root = mcts.mcts_debug(g, 10000)
root.game.board.print_pretty()
print(pi)
assert pi[2] > pi[5]

print("Test game 3")
g = game.Game(np.array([1,0,0,-1,1,0,1,-1,-1]))
pi, root = mcts.mcts_debug(g, 10000)
root.game.board.print_pretty()
print(pi)

print("Test game 4")
g = game.Game(np.array([0,1,-1,-1,-1,0,1,0,0]))
pi, root = mcts.mcts_debug(g, 10000)
root.game.board.print_pretty()
print(pi)

print("Test game 5")
g = game.Game(np.array([0,1,-1,0,1,0,1,-1,0]))
pi, root = mcts.mcts_debug(g, 10000)
root.game.board.print_pretty()
print(pi)

print("Test game 6")
g = game.Game(np.array([0,0,-1,0,1,0,1,-1,0]))
pi, root = mcts.mcts_debug(g, 10000)
root.game.board.print_pretty()
print(pi)

print("Test game 7")
g = game.Game(np.array([0,0,0,0,-1,0,0,0,0]))
pi, root = mcts.mcts_debug(g, 10000)
root.game.board.print_pretty()
print(pi)
