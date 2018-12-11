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

    def __init__(self, model):
        self.model = model
        self.mcts_simulations = 10000

    def train(self):
        """
        Perform a singular self play game and then update the neural network
        """
        p1_images = []
        p2_images = []
        p1_labels = []
        p2_labels = []

        g = game.Game()
        mcts = MonteCarloTreeSearch(g, self.mcts_simulations)
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

            if player = 1:
                p1_images, p1_pi, p2_images, p2_pi mcts.get_training_data()
            else:
                p2_images, p2_pi, p1_images, p1_pi mcts.get_training_data()

            self.p1_images += p1_images
            self.p1_pi += p1_pi
            self.p2_images += p1_images
            self.p2_pi += p2_pi

            g.make_move_index(random.choice(best_moves))
            g = game.Game(starting_pos = g.board.get_board())
            mcts = MonteCarloTreeSearch(g, self.mcts_simulations)
            player *= -1



