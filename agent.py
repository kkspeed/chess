import encoder
from model import create_model
from chess_types import GameState, Player

BOARD_WIDTH = 9
BOARD_HEIGHT = 10

# Assumes it's playing red
class Agent:
    def __init__(self):
        self.encoder = encoder.SimpleEncoder()
        self.model = create_model((BOARD_WIDTH, BOARD_HEIGHT, 1), (encoder.TOTAL_MOVES, 1))

    def select_move(self, game_state: GameState):
        board = self.encoder.encode(game_state)
        # predict and return move:
        # 1. get mask, soft max
        # 2. decode and return move
        pass