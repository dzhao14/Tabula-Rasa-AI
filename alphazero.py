from board import Board
import game
from mcts import MonteCarloTreeSearch
import model
import numpy as np
import random

import ipdb


class AlphaZeroAI(object):
    """
    The AlphaZero inspired AI for tic-tac-toe
    """

    def __init__(self, simulations, nn = None):
        if nn is None:
            self.model = model.NN()
        else:
            self.model = model.NN(existing=nn)

        self.mcts_simulations = simulations

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
                simulations = self.mcts_simulations,
                model = self.model,
                )
        player = 1
        while not g.game_over():
            pi = list(mcts.mcts(g))
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

            images, pi = mcts.get_training_data()
            assert len(images) == len(pi)

            if player == 1:
                p1_images += images
                p1_pi += pi
            else:
                p2_images += images
                p2_pi += pi

            g.make_move_index(random.choice(best_moves))
            player *= -1

        
        if g.result == 0:
            labels = [0 for _ in range(len(p1_pi) + len(p2_pi))]
        else:
            if player == 1:
                labels = [-1. for _ in range(len(p1_images))]
                labels += [1. for _ in range(len(p2_images))]
            else:
                labels = [1. for _ in range(len(p1_images))]
                labels += [-1. for _ in range(len(p2_images))]

        p1_images = np.array(p1_images)
        p2_images = np.array(p2_images)
        data = np.vstack((p1_images, p2_images))
        p1_pi = np.array(p1_pi)
        p2_pi = np.array(p2_pi)
        policy = np.vstack((p1_pi, p2_pi))
        labels = np.array(labels)
        
        self.model.train(data, policy, labels)

