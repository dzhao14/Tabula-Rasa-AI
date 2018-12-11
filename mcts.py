from board import Board
from game import Game
from math import log, sqrt
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

    def set_prior(self, prior):
        self.prior = prior

    def compute_g(self):
        """
        Calculate the evaluation for going to this node from the parent node
        """
        if self.visits == 0:
            return UNSEEN_SCORE
        
        if self.model is not None:
            cbase = 2
            cinit = 2
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
        game = self.game.make_copy()
        while not game.game_over():
            possible_moves = game.get_possible_moves()
            random_choice = random.randint(0, len(possible_moves)-1)
            game.make_move(possible_moves[random_choice])

        return game.result

    def create_children(self):
        """
        Create the children nodes from this node
        """
        children = []
        if self.is_terminal:
            self.children = children
            return
        
        p = None
        nn_prob = []
        if self.model is not None:
            p = self.model.predict_policy(self.game.board)

        board = self.game.board.get_flipped_board()
        possible_moves = self.game.get_possible_moves()
        for move in possible_moves:
            move_indx = Board.get_move_index(board, move)
            child = Node(self, self.game.make_move_and_copy(move))
            if p is not None:
                child.set_prior(p[move_indx])
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
            best = []
            for i, gval in enumerate(g_vals):
                if gval > best_val:
                    best_val = gval
                    best = [i]
                elif gval == best_val:
                    best.append(i)
                else:
                    continue

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
        if self.children is None:
            self.create_children()

        board = self.game.board.get_flipped_board()
        avg_scores = [-1 for _ in range(9)]
        for child in self.children:
            next_board = child.game.board.get_board()
            move_indx = Board.get_move_index(board, next_board)
            avg_scores[move_indx] = child.score / child.visits
            if self.game.player == -1:
                avg_scores[move_indx] *= -1

        min_val = min(avg_scores)
        avg_scores_shifted = list(map(lambda v : v - min_val, avg_scores))
        total = sum(avg_scores_shifted)
        if total == 0:
            return self.create_uniform_policy()
        pi = list(map(lambda x : x / total, avg_scores_shifted)) 
        return pi

    def create_uniform_policy(self):
        pi = [0 for i in range(9)]
        board = self.game.board.get_flipped_board()
        for child in self.children:
            next_board = child.game.board.get_board()
            move_indx = Board.get_move_index(board, next_board)
            pi[move_indx] = 1. / len(self.children)

        return pi

    def get_avg_score(self):
        if self.visits == 0:
            return UNSEEN_SCORE
        else:
            return self.score / self.visits


class MonteCarloTreeSearch:

    def __init__(self, game, simulations = None, model = None):
        self.game = game
        if simulations is None:
            self.max_simulations = 10000
        else:
            self.max_simulations = simulations
        self.model = model
        self.p1_images = []
        self.p2_images = []
        self.p1_pi = []
        self.p2_pi = []

    def mcts(self):
        root = Node(None, self.game.make_copy(), model = self.model)
        assert root.game.player == 1
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
        self.p1_images = []
        self.p2_images = []
        self.p1_pi = []
        self.p2_pi = []

    def store_training_data(self, node):
        """
        For each node with more than X visits store the board state and computed
        policy vector
        """
        if node.visits >= self.max_simulations // 10 and not node.game.game_over():
            pi = node.create_policy()
            if node.game.player == 1:
                self.p1_images.append(node.game.board.get_board())
                self.p1_pi.append(np.array(pi))
            else:
                self.p2_images.append(node.game.board.get_board())
                self.p2_pi.append(np.array(pi))
            for child in node.children:
                self.store_training_data(child)

    def get_training_data(self):
        return self.p1_images, self.p1_pi, self.p2_images, self.p2_pi
         

def mcts_debug(game, max_simulations):
    root = Node(None, game)
    assert root.game.player == 1
    if root.is_terminal:
        return [0. for _ in range(9)]

    simulations = 1
    while simulations <= max_simulations:
        node = root
        while node.visits != 0 or node == root:
            node = node.get_next_state()
            if node.is_terminal:
                break
        value = node.rollout()
        node.back_prop(value)
        simulations += 1

    return root.create_policy(), root
