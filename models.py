from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def classify_win(self):
        raise NotImplementedError("AI must be able to predict the outcome of " \
                "the game")

    @abstractmethod
    def classify_moves(self):
        raise NotImplementedError("AI should be able to return a policy " \
                "vector containing the probability of taking each move")



class FNN(Model):
    def __init__(self, model=None):
        self.model = model

    def classify_win(self):
        pass

    def classify_moves(self):
        pass
