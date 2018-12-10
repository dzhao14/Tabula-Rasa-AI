from math import log, sqrt
import numpy
import random
import model


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
        self.childArray = []

        self.visits = 0
        self.score = 0
        #self.priorProb = apply f to this board

    def calculate_g(self):
        """
        Calculate the evaluation for going to this node from the parent node
        """
        if self.visits == 0:
            return 999999
        #c = log((1 + self.parent.visits + cbase) / cbase) + cinit
        c = 2
        Q = self.score / self.visits
        U = c * self.prior * sqrt(self.parent.visits) / (1 + self.visits)
        return Q + U


class MonteCarloTreeSearch:

    def __init__(self, game):
        self.root = Node(None, game)

    def selectNode(self, root):
        node = root
        while len(node.childArray) != 0:
            node = self.findBestNode(node)
        return node

    def findBestNode(self, node):
        if node.state.playerNo == -1:
            if len(node.childArray) == 0:
                self.expandNode(node)
            node = random.choice(node.childArray)
            return node
        parentVisit = node.state.visitCount
        vals = list(map(lambda x: self.uct(parentVisit, x.state.winScore, x.state.visitCount),node.childArray));
        return node.childArray[numpy.argmax(vals)]

    def expandNode(self, node):
        possibleStates = node.state.getAllPossibleStates();
        node.childArray = list(map(lambda x: Node(x,node), possibleStates))

    def backPropogation(self, leaf, win):
        node = leaf
        while(leaf != "no parent"):
            leaf.state.visitCount = leaf.state.visitCount + 1
            if win == 1:
                leaf.state.winScore = leaf.state.winScore + 1
            if win == -1:
                leaf.state.winScore = leaf.state.winScore - 1
            leaf = leaf.parent

    def simulateRandomPlayout(self,start):
        node = start

        while game.status(node.state.board) == 2:
            if len(node.childArray) == 0:
                self.expandNode(node)
            node = random.choice(node.childArray)
        return game.status(node.state.board)


    def findNextMove(self, board, playerNo):
        tree = Tree(Node(State(board, playerNo, 0, 0), "no parent"))
        self.rootNode = tree.root

        self.simulations = 0
        while self.simulations < 1000:
            node = self.selectNode(self.rootNode);
            if game.status(node.state.board) == 2:
                self.expandNode(node);
            exploreNode = node
            if len(exploreNode.childArray) > 0:
                exploreNode = node.getRandomChildNode()
            playoutResult = self.simulateRandomPlayout(exploreNode)
            self.backPropogation(exploreNode, playoutResult)
            self.simulations = self.simulations + 1

        vals = list(map(lambda x: self.uct(self.simulations, x.state.winScore,
            x.state.visitCount),self.rootNode.childArray));
        bestNode = self.findBestNode(self.rootNode)
        return bestNode.state.board

