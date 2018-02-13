import numpy
import random
import math
import c4_game as game
import model

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
        return list(map(lambda bd: State(bd, -self.playerNo, 0, 0),
                game.possible_moves(self.board, self.playerNo)));
        
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
            if len(node.childArray) == 0:
                self.expandNode(node)
            node = random.choice(node.childArray)
            return node
        parentVisit = node.state.visitCount
        vals = list(map(lambda x: self.uct(parentVisit, x.state.winScore, x.state.visitCount),node.childArray));
        return node.childArray[numpy.argmax(vals)]
            # if len(node.childArray) == 0:
            #     self.expandNode(node)
            # moveind = numpy.argmax(self.opponent.predict_policy(node.state.board))
            # return node.childArray[moveind]


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

    def getTrainingData(self, board):
        #possible_moves = numpy.where(board==0)[0]
        #mov_vals = model.softmax(list(map(lambda x: self.uct(self.simulations,
        #    x.state.winScore, x.state.visitCount), self.rootNode.childArray)))
        #policy = numpy.zeros(9)
        #for i in range(0, len(possible_moves)):
        #    policy[possible_moves[i]] = mov_vals[i]
        #return policy
        return []
