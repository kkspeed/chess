import numpy as np
import copy
import encoder
import textwrap
import h5py
from model import create_model
from chess_types import GameState, Player, Move, Board, Point

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
        h5file['experience'].create_dataset('inputs', data=np.array(self.inputs))
        h5file['experience'].create_dataset('actions', data=np.array(self.actions))
        h5file['experience'].create_dataset('rewards', data=np.array(self.rewards))

    def load(self, h5file):
        experience = h5file.get('experience')
        self.inputs = experience['inputs']
        self.actions = experience['actions']
        self.rewards = experience['rewards']

class Agent:
    def __init__(self, player: Player, collector: ExpCollector):
        self.player = player
        self.encoder = encoder.SimpleEncoder()
        self.model = create_model((1, BOARD_HEIGHT, BOARD_WIDTH), (encoder.TOTAL_MOVES, 1))
        self.collector = collector

    def select_move(self, game_state: GameState):
        if self.player == Player.black:
            assert game_state.player == Player.black
            game_state.player = Player.red
            top = game_state.board.height - 1
            for piece in game_state.board.pieces:
                piece.color = Player.black if piece.color == Player.red else Player.red
                piece.pos = Point(top - piece.pos.row, piece.pos.col)
            encoded = self.encoder.encode(game_state.board)
            predicted = self.model.predict(np.array([encoded]))[0]
            move = self.choose(predicted, game_state)
            if move is None:
                return None
            m, idx = move
            self.collector.record(encoded, idx)
            new_board = m.apply_move(game_state.board)
            for piece in new_board.pieces:
                piece.color = Player.black if piece.color == Player.red else Player.red
                piece.pos = Point(top - piece.pos.row, piece.pos.col)
            state = GameState(new_board, Player.red, game_state.steps + 1)
            return state
        else:
            encoded = self.encoder.encode(game_state.board)
            predicted = self.model.predict(np.array([encoded]))[0]
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
            if move is not None and move.piece in state.board.pieces and move.target.row >= 0 and move.target.row < state.board.height \
                and move.target.col < state.board.width and move.target.col >= 0:
                return move, idx
        return None

def clip_probs(original_probs):
    min_p = 1e-5
    max_p = 1 - min_p
    clipped_probs = np.clip(original_probs, min_p, max_p)
    clipped_probs = clipped_probs / np.sum(clipped_probs)
    return clipped_probs 

def self_play():
    collector1 = ExpCollector()
    collector2 = ExpCollector()
    agent1 = Agent(Player.red, collector1)
    agent2 = Agent(Player.black, collector2)
    board = Board()
    board.parse_from_string(textwrap.dedent("""\
        車馬象仕将仕象馬車
        .........
        .包.....包.
        卒.卒.卒.卒.卒
        .........
        .........
        兵.兵.兵.兵.兵
        .炮.....炮.
        .........
        车马相士帅士相马车"""))
    game = GameState(board, Player.red)
    winner = None
    
    while game.winner() is None:
        print("Playing: ", game.steps)
        game = agent1.select_move(game)
        if game is None:
            winner = Player.black
            break
        game = agent2.select_move(game)
        if game is None:
            winner = Player.red
            break
    if winner is None:
        winner = game.winner()
    if winner == Player.black:
        agent1.finish(-1)
        agent2.finish(1)
    if winner == Player.red:
        agent1.finish(1)
        agent2.finish(-1)
    if winner != -1:
        h51 = h5py.File('play_1.h5', 'w')
        collector1.save(h51)
        h51.close()
        h52 = h5py.File('play_2.h5', 'w')
        collector2.save(h52)
        h52.close()

if __name__ == "__main__":
    self_play()