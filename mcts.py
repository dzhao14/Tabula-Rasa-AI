import numpy
import random
import math
import ttt_game as game
import model

class Node:
    """
    A node has a state, a parent, and a possible empty array of children
    """
    def __init__(self, state, parent, prob, score):
        self.state = state
        self.parent = parent
        self.childArray = []
        self.prob = prob
        self.score = score

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
        return map(lambda bd: State(bd, -self.playerNo, 1, 1),
                game.possible_moves(self.board, self.playerNo));
        
class MonteCarloTreeSearch:
    def __init__(self, WIN_SCORE, level, mod, opmod):
        self.WIN_SCORE = WIN_SCORE
        self.level = level
        self.mod = mod
        self.opmod = opmod

    def selectNode(self, root):
        node = root
        while len(node.childArray) != 0:
            node = self.findBestNode(node)
        return node

    def uct(self, node):
        assert node.score > -1
        result = self.vanillaUct(node)
        result = result * node.prob * (node.score + 1)
        return  result

    def vanillaUct(self, node):
        result = node.state.winScore / node.state.visitCount + 1.41 * math.sqrt(math.log(self.rootNode.state.visitCount) / node.state.visitCount)
        return  result

    def evaluateMove(self, board):
        newboard = game.flip_board(board)
        move_inds = self.opmod.predict_policy(newboard)
        if len(move_inds) == 0:
            return self.mod.predict_score(game.flip_board(board))
        move_ind = numpy.argmax(move_inds)
        newboard = game.possible_moves(newboard, 1)[move_ind]
        newboard = game.flip_board(newboard)
        return self.mod.predict_score(newboard)

    def findBestNode(self, node):
        if node.state.playerNo == -1:
            newboard = game.flip_board(node.state.board)
            op_policy = self.opmod.predict_policy(newboard)
            assert len(op_policy) == len(node.childArray)
            move_ind = numpy.argmax(self.opmod.predict_policy(newboard))
            return node.childArray[move_ind]
        vals = map(lambda nd: self.uct(nd), node.childArray);
        return node.childArray[numpy.argmax(vals)]

    def expandNode(self, node):
        """Populate this node's children states"""
        possibleStates = node.state.getAllPossibleStates()
        policy = self.mod.predict_policy(node.state.board)

        assert numpy.isclose(policy.sum(), 1)
        if len(possibleStates) > 0 and possibleStates[0].playerNo == 1:
            scores = list(map(lambda x: self.mod.predict_score(x.board), possibleStates))
        else:
            scores = list(map(lambda x: self.evaluateMove(x.board), possibleStates))
        tups = zip(policy, possibleStates, scores)
        node.childArray = list(map(lambda x: Node(x[1],node, x[0], x[2]), tups))

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
        return self.mod.predict_score(start.state.board)

        #node = start
        #while game.status(node.state.board) == 2:
        #    possibleStates = node.state.getAllPossibleStates();
        #    node = Node(random.choice(possibleStates),node, 0)
        #return game.status(node.state.board)


    def findNextMove(self, board, playerNo):
        tree = Tree(Node(State(board, playerNo, 0, 0), "no parent", 0, 0))
        self.rootNode = tree.root

        simulations = 0
        while simulations < 10:
            node = self.selectNode(self.rootNode);
            if game.status(node.state.board) == 2:
                self.expandNode(node);
            exploreNode = node
            if len(exploreNode.childArray) > 0:
                exploreNode = node.getRandomChildNode()
            playoutResult = self.simulateRandomPlayout(exploreNode)
            self.backPropogation(exploreNode, playoutResult)
            simulations = simulations + 1


        bestNode = self.findBestNode(self.rootNode)
        return bestNode.state.board

    def getTrainingData(self, board):
        possible_moves = numpy.where(board==0)[0]
        mov_vals = model.softmax(list(map(self.vanillaUct, self.rootNode.childArray)))
        policy = numpy.zeros(9)
        for i in range(0, len(possible_moves)):
            policy[possible_moves[i]] = mov_vals[i]
        return policy

