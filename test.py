import mcts
import numpy as np
import c4_game as game
import sys
import random
import model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

AI1 = model.NN()
AI2 = model.NN()
winrate = []
searcher = mcts.MonteCarloTreeSearch(1, 2, AI1)

board = game.getStartBoard()
print(game.print_pretty(board))
while game.status(board == 2):
  move = input("where to move?")
  board = game.possible_moves(board, -1)[int(move)]
  game.print_pretty(board)
  if game.status(board) != 2:
      game.print_pretty(board)
      break
  board = searcher.findNextMove(board,1)
  print(game.print_pretty(board))
  if game.status(board) != 2:
      game.print_pretty(board)
      break
print("winner: " + str(game.status(board)))
print(game.print_pretty(board))
import pdb; pdb.set_trace()

training_rounds = 1
games_per_round = 40

for i in range(0,training_rounds):
    wins = 0
    losses = 0
    ties = 0
    boards = []
    policies = []
    outcomes = []
    for i in range(0,games_per_round):
        board = game.getStartBoard()
        turn = -1
        if random.random() > 0.5:
            turn = -1

        status = game.status(board)
        while status == 2:

            if turn == 1:
                board_after_move = searcher.findNextMove(board,1)
                board = board_after_move
                turn = -1
            else:
                moves = game.possible_moves(board, -1)
                board = random.choice(list(moves))
                turn = 1

            if game.status(board) != 2:
                print("winner: " + str(game.status(board)))
                game.print_pretty(board)
                print("   ")
                break

        status = game.status(board)

        if game.status(board) == 1:
            wins = wins + 1
        if game.status(board) == -1:
            losses = losses + 1
        if game.status(board) == 0:
            ties = ties + 1
    winrate.append(wins / (wins + losses + ties))

    boards = np.array(boards)
    policies = np.array(policies)
    outcomes = np.array(outcomes)

    print(boards.shape)
    print(policies.shape)
    print(outcomes.shape)

    AI1.train(boards, policies, outcomes)
    # AI2.train(boards, policies, outcomes)

    print( "wins" + str(wins), "losses" + str(losses), "ties" + str(ties))
    plt.plot(winrate, 'b')
    plt.savefig('graph.png');
    plt.clf()

wins = 0
losses = 0
ties = 0
for i in range(0,games_per_round):
    board = game.getStartBoard()
    turn = -1
    if random.random() > 0.5:
        turn = -1
    
    status = game.status(board)
    while status == 2:
        if turn == 1:
            board_after_move = searcher.findNextMove(board,1)
            policy = searcher.getTrainingData(board)
            board = board_after_move
            turn = -1
        else:
            board_after_move = searcher.findNextMove(board,1)
            moves = game.possible_moves(board, -1)
            board = random.choice(list(moves))
            turn = 1
    
        if game.status(board) != 2:
            game.print_pretty(board)
            break
        status = game.status(board)
    
    status = game.status(board)
    while len(outcomes) < len(boards):
        outcomes.append(status)
    
    print("winner: " + str(game.status(board)))
    if game.status(board) == 1:
        wins = wins + 1
    if game.status(board) == -1:
        losses = losses + 1
    if game.status(board) == 0:
        ties = ties + 1
print( "wins" + str(wins), "losses" + str(losses), "ties" + str(ties))

print(boards.shape)
print(policies.shape)
print(outcomes.shape)
import pdb; pdb.set_trace()

import pdb; pdb.set_trace()
