import numpy
import random
import math
import ttt_game as game

class Node:
    """
    A node has a state, a parent, and a possible empty array of children
    """
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent
        self.childArray = []

    def getRandomChildNode(self):
        return random.choice(self.childArray)

class Tree:
    def __init__(self, root):
        self.root = root

class State:
    def __init__(self, board, playerNo, visitCount, winScore):
        self.board = board
        self.playerNo = playerNo
        self.visitCount = visitCount
        self.winScore = winScore

    def getAllPossibleStates(self):
        return map(lambda bd: State(bd, -self.playerNo, 0, 0),
                game.possible_moves(self.board, self.playerNo));
        
class MonteCarloTreeSearch:
    def __init__(self, WIN_SCORE, level, opponent):
        self.WIN_SCORE = WIN_SCORE
        self.level = level
        self.opponent = opponent

    def selectNode(self, root):
        node = root
        while len(node.childArray) != 0:
            node = self.findBestNode(node)
        return node

    def uct(self, totalVisit, wins, nodeVisit):
        if nodeVisit == 0:
            return 999999
        return wins / nodeVisit + 1.41 * math.sqrt(math.log(totalVisit) / nodeVisit)

    def findBestNode(self, node):
        if node.state.playerNo == -1:
            return node.getRandomChildNode()
        parentVisit = node.state.visitCount
        vals = map(lambda x: self.uct(parentVisit, x.state.winScore, x.state.visitCount),node.childArray);
        return node.childArray[numpy.argmax(vals)]

    def expandNode(self, node):
        possibleStates = node.state.getAllPossibleStates();
        node.childArray = map(lambda x: Node(x,node), possibleStates)

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
            possibleStates = node.state.getAllPossibleStates();
            node = Node(random.choice(possibleStates),node)
        return game.status(node.state.board)


    def findNextMove(self, board, playerNo):
        tree = Tree(Node(State(board, playerNo, 0, 0), "no parent"))
        rootNode = tree.root

        simulations = 0
        while simulations < 7000:
            node = self.selectNode(rootNode);
            if game.status(node.state.board) == 2:
                self.expandNode(node);
            exploreNode = node
            if len(exploreNode.childArray) > 0:
                exploreNode = node.getRandomChildNode()
            playoutResult = self.simulateRandomPlayout(exploreNode)
            self.backPropogation(exploreNode, playoutResult)
            simulations = simulations + 1


        bestNode = self.findBestNode(rootNode)
        return bestNode.state.board
