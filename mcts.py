from board import Board
from game import Game
from math import log, sqrt
import numpy
import random

import ipdb


class Node(object):
    """
    A state is represented by:
     - The game which contains the player and board info
     - The parent to this node
     - The resulting board position from taking a valid action at this node is 
     a child node
    """
    def __init__(self, parent, game):
        self.game = game
        self.is_terminal = self.game.game_over()
         
        self.parent = parent
        self.children = None

        self.visits = 0
        self.score = 0
        self.prior = 1. #TODO add in the cnn

    def compute_g(self):
        """
        Calculate the evaluation for going to this node from the parent node
        """
        if self.visits == 0:
            return 999999

        #c = log((1 + self.parent.visits + cbase) / cbase) + cinit
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
        
        possible_moves = self.game.get_possible_moves()
        for move in possible_moves:
            children.append(Node(self, self.game.make_move_and_copy(move)))
        
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

        min_val = min(avg_scores)
        avg_scores_shifted = list(map(lambda v : v - min_val, avg_scores))
        total = sum(avg_scores_shifted)
        pi = list(map(lambda x : x / total, avg_scores_shifted)) 
        return pi

    def get_avg_score(self):
        if self.visits == 0:
            return 999999
        else:
            return self.score / self.visits


class MonteCarloTreeSearch:

    def __init__(self, game):
        self.game = game
        self.max_simulations = 100000

    def mcts(self):
        root = Node(None, self.game.make_copy())
        assert root.game.player == 1
        if root.is_terminal:
            return [0. for _ in range(9)]
        simulations = 1
        while simulations <= self.max_simulations:
            node = root
            while node.visits != 0 or node == self.root:
                node = node.get_next_state()
                if node.is_terminal:
                    break
            value = node.rollout()
            node.back_prop(value)
            simulations += 1

        return self.root.create_policy()

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
