from board import Board, permute
from game import Game
from math import log, sqrt, exp
import numpy as np
import random

import ipdb

UNSEEN_SCORE = 999999

class Node(object):
    """
    A state is represented by:
     - The game which contains the player and board info
     - The parent to this node
     - The resulting board position from taking a valid action at this node is
     a child node
    """
    def __init__(self, parent, game, model = None):
        self.game = game
        self.is_terminal = self.game.game_over()

        self.parent = parent
        self.children = None

        self.visits = 0
        self.score = 0
        self.model = model
        self.prior = 1.

        self.noise = False

    def set_prior(self, prior):
        self.prior = prior

    def use_noise(self):
        self.noise = True

    def compute_g(self):
        """
        Calculate the evaluation for going to this node from the parent node
        """
        if self.visits == 0:
            return UNSEEN_SCORE

        if self.model is not None:
            cbase = 19652
            cinit = 1.25
            c = log((1 + self.parent.visits + cbase) / cbase) + cinit
        else:
            c = 2
        Q = self.score / self.visits
        U = c * self.prior * sqrt(self.parent.visits) / (1 + self.visits)
        if self.parent.game.player == -1 and self.parent is not None:
            return -Q + U
        else:
            return Q + U

    def rollout(self):
        """
        Playout the game randomly and return the outcome of the game wrt to the
        player that started this game
        """
        if self.model is None:
            game = self.game.make_copy()
            while not game.game_over():
                possible_moves = game.get_possible_moves()
                random_choice = random.randint(0, len(possible_moves)-1)
                game.make_move(possible_moves[random_choice])

            return game.result

        else:
            score = self.model.predict_score(self.game.board)
            return self.game.player * score

    def create_children(self):
        """
        Create the children nodes from this node
        """
        children = []
        if self.is_terminal:
            self.children = children
            return

        possible_moves = self.game.get_possible_moves()
        move_indexes = self.game.get_possible_move_indexes()
        self.children_index = move_indexes

        if self.model is not None:
            p = self.model.predict_policy(self.game.board)
            p = self.model.validate_policy(self.game.board, p)
            if self.noise:
                noise = np.random.gamma(0.3, 1, len(p))
                for i, prob in enumerate(p):
                    p[i] = 0.75 * prob + 0.25 + noise[i]

        for i, move in enumerate(possible_moves):
            child = Node(
                    self,
                    self.game.make_move_and_copy(move),
                    model=self.model,
                    )
            if self.model is not None:
                child.set_prior(p[i])
            children.append(child)

        self.children = children

    def back_prop(self, value):
        """
        Update the score and visit stats with the given rollout value
        """
        self.visits += 1
        self.score += value
        if self.parent:
            self.parent.back_prop(value)

    def get_next_state(self):
        if self.children is None:
            self.create_children()

        g_vals = list(map(lambda x : x.compute_g(), self.children))
        if self.game.player == -1:
            best_val = float('-inf')
        else:
            best_val = -1

        best = []
        for i, gval in enumerate(g_vals):
            if gval > best_val:
                best_val = gval
                best = [i]
            elif gval == best_val:
                best.append(i)
            else:
                continue

        return self.children[random.choice(best)]

    def create_policy(self):
        avg_scores = []
        for child in self.children:
            avg_scores.append(child.score / child.visits)
        avg_scores = list(map(lambda x : exp(x), avg_scores))
        avg_sum = sum(avg_scores)
        avg_scores = list(map(lambda x : x / avg_sum, avg_scores))

        pi = [0 for _ in range(9)]
        for i, move_indx in enumerate(self.children_index):
            pi[move_indx] = avg_scores[i]

        return pi

    def get_avg_score(self):
        if self.visits == 0:
            return UNSEEN_SCORE
        else:
            return self.score / self.visits


class MonteCarloTreeSearch:

    def __init__(self, simulations = None, model = None):
        if simulations is None:
            self.max_simulations = 10000
        else:
            self.max_simulations = simulations
        self.model = model

        self.images = []
        self.pi = []

    def mcts(self, game):
        game.player = 1
        root = Node(None, game.make_copy(), model = self.model)
        if root.is_terminal:
            return [0. for _ in range(9)]

        simulations = 1
        while simulations <= self.max_simulations:
            node = root
            while node.visits != 0 or node == root:
                node = node.get_next_state()
                if node.is_terminal:
                    break
            value = node.rollout()
            node.back_prop(value)
            simulations += 1

        self.clear_training_data()
        self.store_training_data(root)

        return root.create_policy()

    def clear_training_data(self):
        self.images = []
        self.pi = []

    def store_training_data(self, node):
        """
        For each node with more than X visits store the board state and computed
        policy vector
        """
        images = permute(node.game.board.get_board())
        pi = permute(np.array(node.create_policy()))

        self.images += images
        self.pi += pi

    def get_training_data(self):
        return self.images, self.pi,

