import sys
import os
import numpy as np
import copy
import encoder
import textwrap
import h5py
from model import create_model
from chess_types import GameState, Player, Move, Board, Point
from keras.optimizers import SGD, Adam

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
        self.encountered = set()

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
            self.encountered.add(new_board)
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
            self.encountered.add(new_board)
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
                result_board = move.apply_move(state.board)
                if result_board in self.encountered:
                    continue
                ps = [p for p in result_board.pieces if str(p) == '帅' or str(p) == '将']
                if len(ps) == 2:
                    k1, k2 = ps
                    if k1.pos.col == k2.pos.col:
                        face = True
                        for r in range(min(k1.pos.row + 1, k2.pos.row + 1), max(k1.pos.row, k2.pos.row)):
                            if result_board.piece_at(Point(r, k1.pos.col)):
                                face = False
                                break
                        if face:
                            continue
                return move, idx
        return None

    def train(self, exp: ExpCollector):
        self.model.compile(optimizer=Adam(), loss=['categorical_crossentropy'])
        target_vectors = prepare_experience_data(exp)
        self.model.fit(exp.inputs, target_vectors, batch_size=20, epochs=1, shuffle='batch')

def clip_probs(original_probs):
    min_p = 0.0001
    max_p = 1 - min_p
    clipped_probs = np.clip(original_probs, min_p, max_p)
    clipped_probs = clipped_probs / np.sum(clipped_probs)
    return clipped_probs

def prepare_experience_data(experience: ExpCollector):
    experience_size = len(experience.rewards)
    target_vectors = np.zeros((experience_size, encoder.TOTAL_MOVES))
    for i in range(experience_size):
        action = experience.actions[i]
        reward = experience.rewards[i]
        target_vectors[i][action] = reward
    return target_vectors

def game_play(agent1, agent2):
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
        # print("Playing: ", game.steps)
        game = agent1.select_move(game)
        if game is None:
            winner = Player.black
            break
        game = agent2.select_move(game)
        if game is None:
            winner = Player.red
            break
    if winner is None:
        return game.winner()
    return winner

def self_play(episode, round, agent1, agent2):
    if not os.path.exists(episode):
        os.mkdir(episode)
    winner = game_play(agent1, agent2)
    score = 600 - len(collector1.inputs)
    if winner == Player.black:
        print("Black win")
        agent1.finish(-score)
        agent2.finish(score)
        collector2.rewards[-1] = 2 * score
    if winner == Player.red:
        print("Red win")
        agent1.finish(score)
        collector1.rewards[-1] = 2 * score
        agent2.finish(-score)
    if winner == -1:
        agent1.finish(-600000)
        agent2.finish(-600000)
        print("It's draw %s - %d" % (episode, round))
    file_path = os.path.join(episode, "%s_1.h5" % round)
    h51 = h5py.File(file_path, 'w')
    collector1.save(h51)
    h51.close()
    file_path = os.path.join(episode, "%s_2.h5" % round)
    h52 = h5py.File(file_path, 'w')
    collector2.save(h52)
    h52.close()

if __name__ == "__main__":
    episode = sys.argv[1]
    for round in range(100):
        self_play(episode, str(round))
