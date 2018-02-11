import mcts
import numpy as np
import ttt_game
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

# board = ttt_game.getStartBoard()
# print(ttt_game.print_pretty(board))
# while ttt_game.status(board == 2):
#   move = input("where to move?")
#   board[int(move)] = -1
#   if ttt_game.status(board) != 2:
#       ttt_game.print_pretty(board)
#       break
#   board = searcher.findNextMove(board,1)
#   print(ttt_game.print_pretty(board))
#   if ttt_game.status(board) != 2:
#       ttt_game.print_pretty(board)
#       break

training_rounds = 1
games_per_round = 30

for i in range(0,training_rounds):
    wins = 0
    losses = 0
    ties = 0
    boards = []
    policies = []
    outcomes = []
    for i in range(0,games_per_round):
        board = ttt_game.getStartBoard()
        turn = -1
        if random.random() > 0.5:
            turn = -1

        status = ttt_game.status(board)
        while status == 2:

            if turn == 1:
                board_after_move = searcher.findNextMove(board,1)
                policy = searcher.getTrainingData(board)
                examples = ttt_game.expandExample(board, policy)
                boards = boards + examples[0]
                policies = policies + examples[1]
                board = board_after_move
                turn = -1
            else:
                moves = ttt_game.possible_moves(board, -1)
                board = random.choice(list(moves))
                turn = 1

            if ttt_game.status(board) != 2:
                #ttt_game.print_pretty(board)
                break
            status = ttt_game.status(board)

        status = ttt_game.status(board)
        while len(outcomes) < len(boards):
            outcomes.append(status)

        print("winner: " + str(ttt_game.status(board)))
        if ttt_game.status(board) == 1:
            wins = wins + 1
        if ttt_game.status(board) == -1:
            losses = losses + 1
        if ttt_game.status(board) == 0:
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
    board = ttt_game.getStartBoard()
    turn = -1
    if random.random() > 0.5:
        turn = -1
    
    status = ttt_game.status(board)
    while status == 2:
    
        if turn == 1:
            moveind = np.argmax(AI1.predict_policy(board))
            board = random.choice(list(moves))
            board[np.where(board==0)[0][moveind]] = 1
            turn = -1
        else:
            moves = ttt_game.possible_moves(board, -1)
            board = random.choice(list(moves))
            turn = 1
    
        if ttt_game.status(board) != 2:
            #ttt_game.print_pretty(board)
            break
        status = ttt_game.status(board)
    
    status = ttt_game.status(board)
    while len(outcomes) < len(boards):
        outcomes.append(status)
    
    print("winner: " + str(ttt_game.status(board)))
    if ttt_game.status(board) == 1:
        wins = wins + 1
    if ttt_game.status(board) == -1:
        losses = losses + 1
    if ttt_game.status(board) == 0:
        ties = ties + 1
print( "wins" + str(wins), "losses" + str(losses), "ties" + str(ties))

print(boards.shape)
print(policies.shape)
print(outcomes.shape)
import pdb; pdb.set_trace()

import pdb; pdb.set_trace()
