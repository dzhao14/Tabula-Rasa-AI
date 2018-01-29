import models

class AI():

    def __init__(board, model=None):
        if model is None:
            self.model = models.FNN() 
        self.model = model
        self.board = board

    def update_board(self, board):
        self.board = board



    
