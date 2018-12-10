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

        p1 = True
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

            g.make_move_index(random.choice(best_moves))




        

