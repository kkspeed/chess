import numpy as np
import copy
import encoder
from model import create_model
from chess_types import GameState, Player, Move

BOARD_WIDTH = 9
BOARD_HEIGHT = 10

class ExpCollector:
    def __init__(self):
        self.inputs = []
        self.actions = []
        self.rewards = []

    def record(self, input, action):
        self.inputs.append(input)
        self.actions.append(action)
        
    def assign_reward(self, reward):
        self.rewards = [reward] * len(self.inputs)

    def save(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('inputs', data=self.inputs)
        h5file['experience'].create_dataset('predictions', data=self.predictions)
        h5file['experience'].create_dataset('rewards', data=self.rewards)

class Agent:
    def __init__(self, player: Player, collector: ExpCollector):
        self.player = player
        self.encoder = encoder.SimpleEncoder()
        self.model = create_model((BOARD_WIDTH, BOARD_HEIGHT, 1), (encoder.TOTAL_MOVES, 1))
        self.collector = collector

    def select_move(self, game_state: GameState):
        if self.player == Player.black:
            assert game_state.player == Player.black
            game_state.player = Player.red
            top = game_state.board.height - 1
            for piece in game_state.board.pieces:
                piece.color = Player.black if piece.color == Player.red else Player.red
                piece.pos.row = top - piece.pos.row
            encoded = self.encoder.encode(game_state.board)
            predicted = self.model.predict(encoded)
            move = self.choose(predicted, game_state)
            if move is None:
                return None
            m, idx = move
            self.collector.record(encoded, idx)
            new_board = m.apply_move(game_state.board)
            for piece in new_board.pieces:
                piece.color = Player.black if piece.color == Player.red else Player.red
                piece.pos.row = top - piece.pos.row
            state = GameState(new_board, Player.red, game_state.steps + 1)
            return state
        else:
            encoded = self.encoder.encode(game_state.board)
            predicted = self.model.predict(encoded)
            move = self.choose(predicted, game_state)
            if move is None:
                return None
            m, idx = move
            self.collector.record(encoded, idx)
            new_board = m.apply_move(game_state.board)
            state = GameState(new_board, Player.black, game_state.steps + 1)
            return state

    def finish(self, reward):
        self.collector.assign_reward(reward)

    def choose(self, move_probs, state) -> Move:
        candidates = np.arange(0, encoder.TOTAL_MOVES)
        ranked_moves = np.random.choice(candidates, 
            len(candidates), replace=False, p=clip_probs(move_probs))
        for idx in ranked_moves:
            move = self.encoder.decode_move(state, idx)
            if move is not None and move.pos.row >= 0 and move.pos.row < state.board.height \
                and move.pos.col < state.board.width and move.pos.col >= 0:
                return move, idx
        return None

def clip_probs(original_probs):
    min_p = 1e-5
    max_p = 1 - min_p
    clipped_probs = np.clip(original_probs, min_p, max_p)
    clipped_probs = clipped_probs / np.sum(clipped_probs)
    return clipped_probs 