import sys
import os
import numpy as np
import copy
import encoder
import textwrap
import h5py
from model import create_model
from chess_types import GameState, Player, Move, KillMove, Board, Point
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
        h5file['experience'].create_dataset(
            'inputs', data=np.array(self.inputs))
        h5file['experience'].create_dataset(
            'actions', data=np.array(self.actions))
        h5file['experience'].create_dataset(
            'rewards', data=np.array(self.rewards))

    def load(self, h5file):
        experience = h5file.get('experience')
        self.inputs = experience['inputs'][:]
        self.actions = experience['actions'][:]
        self.rewards = experience['rewards'][:]


class Agent:
    def __init__(self, player: Player, collector: ExpCollector):
        self.player = player
        self.encoder = encoder.SimpleEncoder()
        self.model = create_model(
            (1, BOARD_HEIGHT, BOARD_WIDTH), (encoder.TOTAL_MOVES, 1))
        self.collector = collector
        self.encountered = set()

    def select_move(self, game_state: GameState):
        if self.player == Player.black:
            assert game_state.player == Player.black
            game_state.player = Player.red
            top = game_state.board.height - 1
            for piece in game_state.board.pieces:
                piece.color = piece.color.other()
                piece.pos = Point(top - piece.pos.row, piece.pos.col)
            encoded = self.encoder.encode(game_state.board)
            predicted = self.model.predict(np.array([encoded]))[0]
            move = self.choose(predicted, game_state)
            if move is None:
                return None
            m, idx = move
            if self.collector is not None:
                self.collector.record(encoded, idx)
            new_board = m.apply_move(game_state.board)
            self.encountered.add(str(new_board))
            for piece in new_board.pieces:
                piece.color = piece.color.other()
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
            if self.collector is not None:
                self.collector.record(encoded, idx)
            new_board = m.apply_move(game_state.board)
            state = GameState(new_board, Player.black, game_state.steps + 1)
            self.encountered.add(str(new_board))
            return state

    def finish(self, reward):
        for i in range(len(self.collector.inputs)):
            input = self.collector.inputs[i]
            board = self.encoder.decode(input)
            state = GameState(board, Player.red)
            move = self.encoder.decode_move(state, self.collector.actions[i])
            if type(move) is KillMove:
                piece = board.piece_at(move.target)
                name = piece.name()
                if name == '帅':
                    self.collector.rewards.append(reward + 50)
                elif name == '车':
                    self.collector.rewards.append(reward + 10)
                elif name == '炮':
                    self.collector.rewards.append(reward + 6)
                elif name == '马':
                    self.collector.rewards.append(reward + 5)
                elif name == '兵':
                    self.collector.rewards.append(reward + 2)
                elif name == '士' or name == '相':
                    self.collector.rewards.append(reward + 1)
                else:
                    self.collector.rewards.append(reward)
            else:
                self.collector.rewards.append(reward)
        assert len(self.collector.rewards) == len(self.collector.inputs)

    def choose(self, move_probs, state) -> Move:
        explore_probs = 0.01
        candidates = np.arange(0, encoder.TOTAL_MOVES)
        weighted_moves = np.random.choice(
            candidates, len(candidates), replace=False, p=clip_probs(move_probs))
        uniform_moves = np.random.choice(
            candidates, len(candidates), replace=False)
        ranked_moves = weighted_moves if np.random.uniform(
        ) >= explore_probs else uniform_moves
        valid_move = None
        for idx in reversed(ranked_moves):
            move = self.encoder.decode_move(state, idx)
            if move is not None:
                with state.board.mutable() as result_board:
                    result_board.move_piece(move)
                    if str(result_board) in self.encountered:
                        continue
                    if isinstance(move, KillMove) and not result_board.pieces_by_strs('将'):
                        return move, idx
                valid_move = (move, idx)
        return valid_move

    def train_batch(self, inputs, target_vectors):
        self.model.compile(optimizer=Adam(lr=0.001), loss=[
                           'categorical_crossentropy'])
        self.model.fit(inputs, target_vectors, batch_size=4000,
                       epochs=10, shuffle='batch')

    def train(self, exp: ExpCollector):
        self.model.compile(optimizer=Adam(lr=0.02), loss=[
                           'categorical_crossentropy'])
        target_vectors = prepare_experience_data(exp)
        self.model.fit(exp.inputs, target_vectors,
                       batch_size=128, epochs=6, shuffle='batch')


def clip_probs(original_probs):
    min_p = 0.001
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
    agent1.encountered = set()
    agent2.encountered = set()
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
    collector1 = ExpCollector()
    collector2 = ExpCollector()

    agent2.collector = collector2
    agent1.collector = collector1

    winner = game_play(agent1, agent2)
    if winner == Player.black:
        print("Black win")
        agent1.finish(-20)
        agent2.finish(20)
        collector1.rewards[-1] = -100
    if winner == Player.red:
        print("Red win")
        agent1.finish(20)
        agent2.finish(-20)
        collector2.rewards[-1] = -100
    if winner == -1:
        agent1.finish(0)
        agent2.finish(0)
        # for i in range(300, len(collector1.rewards)):
        #     collector1.rewards[i] = -1000
        # for i in range(300, len(collector2.rewards)):
        #     collector2.rewards[i] = -1000
        print("It's draw %s - %d" % (episode, round))
    file_path = os.path.join(episode, "%s_1.h5" % round)
    with h5py.File(file_path, 'w') as h51:
        collector1.save(h51)
    file_path = os.path.join(episode, "%s_2.h5" % round)
    with h5py.File(file_path, 'w') as h52:
        collector2.save(h52)


class HumanAgent:
    def __init__(self, player: Player):
        self.player = player

    def select_move(self, game_state: GameState):
        assert game_state.player == self.player
        move = input("Your move: ")
        row, col = ord(move[0]) - ord('A'), int(move[1])
        tr, tc = ord(move[3]) - ord('A'), int(move[4])
        piece = game_state.board.piece_at(Point(row, col))
        if piece.color != self.player:
            return self.select_move(game_state)
        m = piece.calc_move(game_state.board, Point(tr, tc))
        if m is None:
            return self.select_move(game_state)
        return GameState(m.apply_move(game_state.board),
                         self.player.other(),
                         game_state.steps + 1)
