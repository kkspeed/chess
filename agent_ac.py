import numpy as np
from keras.optimizers import Adam
import os
import h5py
import textwrap

from chess_types import Player, GameState, Point, Move, KillMove, Board
import encoder
import model_ac


class AcExpCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.estimated_rewards = []
        self.advantages = []
        self.rewards = []

    def record(self, state, action, estimated_reward):
        self.states.append(state)
        self.actions.append(action)
        self.estimated_rewards.append(estimated_reward)

    def assign_reward(self, reward):
        self.rewards = [reward] * len(self.actions)
        self.advantages = [reward - e for e in self.estimated_rewards]

    def save(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset(
            'inputs', data=np.array(self.states))
        h5file['experience'].create_dataset(
            'actions', data=np.array(self.actions))
        h5file['experience'].create_dataset(
            'rewards', data=np.array(self.rewards))
        h5file['experience'].create_dataset(
            'estimated_rewards', data=np.array(self.estimated_rewards))
        h5file['experience'].create_dataset(
            'advantages', data=np.array(self.advantages))

    def load(self, h5file):
        experience = h5file.get('experience')
        self.states = experience['inputs'][:]
        self.actions = experience['actions'][:]
        self.estimated_rewards = experience['estimated_rewards'][:]
        self.rewards = experience['rewards'][:]
        self.advantages = experience['advantages'][:]


class AcAgent:
    def __init__(self, player: Player, collector: AcExpCollector):
        self.player = player
        self.encoder = encoder.SimpleEncoder()
        self.model = model_ac.create_model()
        self.collector = collector
        self.encountered = set()

    def select_move(self, game_state: GameState) -> GameState:
        if self.player == Player.black:
            assert game_state.player == Player.black
            game_state.player = Player.red
            top = game_state.board.height - 1
            for piece in game_state.board.pieces:
                piece.color = piece.color.other()
                piece.pos = Point(top - piece.pos.row, piece.pos.col)
            encoded = self.encoder.encode(game_state.board)
            predicted, value = self.model.predict(np.array([encoded]))
            predicted = predicted[0]
            estimated_value = value[0][0]
            move = self.choose(predicted, game_state)
            if move is None:
                return None
            m, idx = move
            if self.collector is not None:
                self.collector.record(encoded, idx, estimated_value)
            new_board = m.apply_move(game_state.board)
            self.encountered.add(str(new_board))
            for piece in new_board.pieces:
                piece.color = piece.color.other()
                piece.pos = Point(top - piece.pos.row, piece.pos.col)
            state = GameState(new_board, Player.red, game_state.steps + 1)
            return state
        else:
            encoded = self.encoder.encode(game_state.board)
            predicted, value = self.model.predict(np.array([encoded]))
            predicted = predicted[0]
            estimated_value = value[0][0]
            move = self.choose(predicted, game_state)
            if move is None:
                return None
            m, idx = move
            if self.collector is not None:
                self.collector.record(encoded, idx, estimated_value)
            new_board = m.apply_move(game_state.board)
            state = GameState(new_board, Player.black, game_state.steps + 1)
            self.encountered.add(str(new_board))
            return state

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

    def train_batch(self, states, policy_targets, value_targets):
        self.model.compile(optimizer=Adam(lr=0.001), loss=[
                           'categorical_crossentropy', 'mse'])
        self.model.fit(
            states, [policy_targets, value_targets], batch_size=4000, epochs=10, shuffle='batch')

    def finish(self, reward):
        self.collector.assign_reward(reward)


def clip_probs(original_probs):
    min_p = 0.001
    max_p = 1 - min_p
    clipped_probs = np.clip(original_probs, min_p, max_p)
    clipped_probs = clipped_probs / np.sum(clipped_probs)
    return clipped_probs


def prepare_experience_data(experience: AcExpCollector):
    n = experience.states.shape[0]
    num_moves = encoder.TOTAL_MOVES
    policy_target = np.zeros((n, num_moves))
    value_target = np.zeros((n,))
    for i in range(n):
        action = experience.actions[i]
        reward = experience.rewards[i]
        policy_target[i][action] = experience.advantages[i]
        value_target[i] = reward
    return policy_target, value_target


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
    collector1 = AcExpCollector()
    collector2 = AcExpCollector()

    agent2.collector = collector2
    agent1.collector = collector1

    winner = game_play(agent1, agent2)
    if winner == Player.black:
        print("Black win")
        agent1.finish(-1)
        agent2.finish(1)
    if winner == Player.red:
        print("Red win")
        agent1.finish(1)
        agent2.finish(-1)
    if winner == -1:
        agent1.finish(0)
        agent2.finish(0)
        print("It's draw %s - %d" % (episode, round))
    file_path = os.path.join(episode, "ac_%s_1.h5" % round)
    with h5py.File(file_path, 'w') as h51:
        collector1.save(h51)
    file_path = os.path.join(episode, "ac_%s_2.h5" % round)
    with h5py.File(file_path, 'w') as h52:
        collector2.save(h52)
