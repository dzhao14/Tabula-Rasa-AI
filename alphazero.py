import model
import numpy as np
import game
from mcts import MonteCarloTreeSearch
from board import Board
import random

import ipdb


class AlphaZeroAI(object):
    """
    The AlphaZero inspired AI for tic-tac-toe
    """

    def __init__(self, nn = None):
        if nn is None:
            self.model = model.NN()
        else:
            self.model = model.NN(existing=nn)

        self.mcts_simulations = 100

    def train(self):
        """
        Perform a singular self play game and then update the neural network
        """
        p1_images = []
        p2_images = []
        p1_pi = []
        p2_pi = []

        g = game.Game()
        mcts = MonteCarloTreeSearch(
                g,
                simulations = self.mcts_simulations,
                model = self.model,
                )
        player = 1
        while not g.game_over():
            pi = list(mcts.mcts())
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

            if player == 1:
                p1_images_, p1_pi_, p2_images_, p2_pi_ = mcts.get_training_data()
            else:
                p2_images_, p2_pi_, p1_images_, p1_pi_ = mcts.get_training_data()
            assert len(p1_images_) == len(p1_pi_) and len(p2_images_) == len(p2_pi_)

            p1_images += p1_images_
            p1_pi += p1_pi_
            p2_images += p2_images_
            p2_pi += p2_pi_

            g.make_move_index(random.choice(best_moves))
            g = game.Game(starting_pos = g.board.get_board())
            mcts = MonteCarloTreeSearch(
                    g,
                    simulations = self.mcts_simulations,
                    model = self.model,
                    )
            player *= -1

        
        if g.result == 0:
            labels = [0 for _ in range(len(p1_pi) + len(p2_pi))]
        elif g.result == -1:
            if player == 1:
                labels = [-1. for _ in range(len(p1_images))]
                labels += [1. for _ in range(len(p2_images))]
            else:
                labels = [1. for _ in range(len(p1_images))]
                labels += [-1. for _ in range(len(p2_images))]
        else:
            ipdb.set_trace()

        p1_images = np.array(p1_images)
        p2_images = np.array(p2_images)
        data = np.vstack((p1_images, p2_images))
        p1_pi = np.array(p1_pi)
        p2_pi = np.array(p2_pi)
        policy = np.vstack((p1_pi, p2_pi))
        labels = np.array(labels)

        self.model.train(data, policy, labels)

    def train_batched(self, number_games):
        """
        Perform self play for the given number of games and then update the 
        neural network
        """
        all_images = []
        all_pi = []
        all_labels = []

        for i in range(number_games):
            p1_images = []
            p2_images = []
            p1_pi = []
            p2_pi = []
            g = game.Game()
            mcts = MonteCarloTreeSearch(
                    g,
                    simulations = self.mcts_simulations,
                    model = self.model,
                    )
            player = 1
            while not g.game_over():
                pi = list(mcts.mcts())
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

                if player == 1:
                    p1_images_, p1_pi_, p2_images_, p2_pi_ = mcts.get_training_data()
                else:
                    p2_images_, p2_pi_, p1_images_, p1_pi_ = mcts.get_training_data()

                p1_images += p1_images_
                p1_pi += p1_pi_
                p2_images += p2_images_
                p2_pi += p2_pi_

                g.make_move_index(random.choice(best_moves))
                g = game.Game(starting_pos = g.board.get_board())
                mcts = MonteCarloTreeSearch(
                        g,
                        simulations = self.mcts_simulations,
                        model = self.model,
                        )
                player *= -1

            
            if g.result == 0:
                labels = [0 for _ in range(len(p1_pi) + len(p2_pi))]
            elif g.result == -1:
                if player == 1:
                    labels = [-1. for _ in range(len(p1_images))]
                    labels += [1. for _ in range(len(p2_images))]
                else:
                    labels = [1. for _ in range(len(p1_images))]
                    labels += [-1. for _ in range(len(p2_images))]
            else:
                ipdb.set_trace()
            
            combined_images = p1_images + p2_images
            combined_pi = p1_pi + p2_pi
            all_labels += labels
            all_images += combined_images
            all_pi += combined_pi


        data = np.array(all_images)
        policy = np.array(all_pi)
        labels = np.array(all_labels)

        self.model.train(data, policy, labels)
