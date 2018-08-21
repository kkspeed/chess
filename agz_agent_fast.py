import os
import numpy as np
import textwrap
import h5py

from keras.optimizers import SGD
import copy

from chess_types import GameState, Move, Player, Board, Point
from encoder import SimpleEncoder
import encoder
from typing import List, Dict
from model_ac import create_model


class AgzExpCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def record(self, state, action):
        self.states.append(state)
        self.actions.append(action)

    def assign_reward(self, reward):
        self.rewards = [reward] * len(self.actions)

    def save(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset(
            'inputs', data=np.array(self.states))
        h5file['experience'].create_dataset(
            'actions', data=np.array(self.actions))
        h5file['experience'].create_dataset(
            'rewards', data=np.array(self.rewards))

    def load(self, h5file):
        experience = h5file.get('experience')
        self.states = experience['inputs'][:]
        self.actions = experience['actions'][:]
        self.rewards = experience['rewards'][:]

class Branch:
    def __init__(self, prior, idx):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.idx = idx

class ZeroTreeNode:
    def __init__(self, value, priors: Dict[Move, (float, int)],
                 parent: 'ZeroTreeNode',
                 last_move: Move):
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches: Dict[Move, Branch] = {}
        for move, p in priors.items():
            self.branches[move] = Branch(p[0], p[1])
        self.children = {}

    def has_move(self) -> bool:
        return len(self.moves()) > 0

    def moves(self) -> List[Move]:
        return list(self.branches.keys())

    def add_child(self, move, child_node):
        self.children[move] = child_node

    def has_child(self, move):
        return move in self.children

    def get_child(self, move) -> 'ZeroTreeNode':
        return self.children[move]

    def expected_value(self, move: Move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, move: Move):
        return self.branches[move].prior

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0

    def record_visit(self, move: Move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value


class ZeroAgent:
    def __init__(self, player: Player, collector: AgzExpCollector=None):
        self.player = player
        self.c = 0.3
        self.num_rounds = 160
        self.encoder = SimpleEncoder()
        self.model = create_model()
        self.collector = collector
        self.encountered = set()

    def select_branch(self, node: ZeroTreeNode) -> Move:
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return q + self.c * p * np.sqrt(total_n) / (n + 1)
        return max(node.moves(), key=score_branch)

    def select_move(self, game_state: GameState) -> GameState:
        print("select move: ", game_state.player, game_state.steps)
        print(str(game_state.board))
        root = self.create_node(game_state.board, game_state.player)
        # TODO: flip black / red sides when selecting move, revisiting exp collector

        for _ in range(self.num_rounds):
            with game_state.board.mutable() as board:
                node = root
                next_move = self.select_branch(node)
                board.move_piece(next_move)
                player = game_state.player.other()

                while node.has_child(next_move):
                    node = node.get_child(next_move)
                    next_move = self.select_branch(node)
                    board.move_piece(next_move)
                    player = player.other()
                board.move_piece(next_move) 
                player = player.other()
                child_node = self.create_node(board, player, parent=node, move=next_move)
                move = next_move
                value = -1 * child_node.value
                while node is not None:
                    node.record_visit(move, value)
                    move = node.last_move
                    node = node.parent
                    value = -1 * value
        
        if not root.has_move():
            return None

        if self.collector is not None:
            with game_state.board.flipped(game_state.player == Player.black) as board:
                result = [0.000001] * encoder.TOTAL_MOVES
                encoded_board = self.encoder.encode(board)
                for move in root.priors.keys():
                    _, idx = root.priors[move]
                    result[idx] = root.visit_count(move)
                result = np.array(result) / sum(result)
                self.collector.record(encoded_board, result)

        exploration_prob = 0.5
        if np.random.uniform() < exploration_prob and root.has_move():
            idx = int(np.random.uniform(0, len(root.moves())))
            move = root.moves()[idx]
            new_board = move.apply_move(game_state.board)
            return GameState(new_board, game_state.player.other(), game_state.steps + 1)

        for move in sorted(root.moves(), key=root.visit_count, reverse=True):
            new_board = move.apply_move(game_state.board)
            if str(new_board) in self.encountered:
                continue
            self.encountered.add(str(new_board))
            return GameState(new_board, game_state.player.other(), game_state.steps + 1)
        return None

    def create_node(self, board: Board, player: Player, move=None, parent=None) -> ZeroTreeNode:
        with board.flipped(player == Player.black) as flipped:
            board_tensor = self.encoder.encode(board)
            model_input = np.array([board_tensor])
            priors, values = self.model.predict(model_input)
            priors = priors[0]
            value = values[0][0]

            move_priors = {}
            for idx, p in enumerate(priors):
                ds = GameState(flipped, Player.red, 0)
                move = self.encoder.decode_move(ds, idx)
                if move is not None:
                    move = move.flip(player == Player.black)
                    move_priors[move] = (p, idx)
            new_node = ZeroTreeNode(value, move_priors, parent, move)
            if parent is not None:
                parent.add_child(new_node)
            return new_node

    def finish(self, reward):
        self.collector.assign_reward(reward)

    def train_batch(self, states, policy_targets, value_targets):
        self.model.compile(optimizer=SGD(lr=0.01, clipvalue=0.2),
            loss=['categorical_crossentropy', 'mse'])
        self.model.fit(
            states, [policy_targets, value_targets],
            batch_size=4000, epochs=10, shuffle='batch')


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
    collector1 = AgzExpCollector()
    collector2 = AgzExpCollector()

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
    file_path = os.path.join(episode, "agz_%s_1.h5" % round)
    with h5py.File(file_path, 'w') as h51:
        collector1.save(h51)
    file_path = os.path.join(episode, "agz_%s_2.h5" % round)
    with h5py.File(file_path, 'w') as h52:
        collector2.save(h52)
