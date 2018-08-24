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
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0

class ZeroTreeNode:
    def __init__(self, state: GameState, value, priors: Dict[Move, float],
                 parent: 'ZeroTreeNode',
                 last_move: Move):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches: Dict[Move, Branch] = {}
        for move, p in priors.items():
            self.branches[move] = Branch(p)
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
        if node.has_move():
            return max(node.moves(), key=score_branch)
        return None

    def select_move(self, game_state: GameState) -> GameState:
        print("select move: ", game_state.player, game_state.steps)
        print(str(game_state.board).replace('.', '。'))
        root = self.create_node(game_state)
        # TODO: flip black / red sides when selecting move, revisiting exp collector

        for _ in range(self.num_rounds):
            node = root
            next_move = self.select_branch(node)
            if next_move is None:
                return None
            while next_move is not None and node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node)
            if next_move is None:
                value = -1 * node.value
                move = node.last_move
                node = node.parent
                while node is not None:
                    node.record_visit(move, value)
                    move = node.last_move
                    node = node.parent
                    value = -1 * value
                continue
            new_board = next_move.apply_move(node.state.board)
            new_state = GameState(
                new_board, node.state.player.other(), node.state.steps + 1)
            child_node = self.create_node(new_state, parent=node, move=next_move)
            move = next_move
            value = -1 * child_node.value
            while node is not None:
                node.record_visit(move, value)
                move = node.last_move
                node = node.parent
                value = -1 * value

        if root.has_move():
            if self.collector is not None:
                with game_state.board.flipped(game_state.player == Player.black) as board:
                    result = []
                    encoded_board = self.encoder.encode(board)
                    for idx in range(encoder.TOTAL_MOVES):
                        ds = GameState(board, Player.red)
                        move = self.encoder.decode_move(ds, idx)
                        if move is None:
                            result.append(0.00000001)
                        else:
                            move = move.flip(game_state.player == Player.black)
                            result.append(root.visit_count(move))
                    result = np.array(result) / sum(result)
                    self.collector.record(encoded_board, result)

            # if np.random.uniform() < exploration_prob and root.has_move():
            #     idx = int(np.random.uniform(0, len(root.moves())))
            #     move = root.moves()[idx]
            #     new_board = move.apply_move(game_state.board)
            #     return GameState(new_board, game_state.player.other(), game_state.steps + 1)
            has_skip = False
            for move in sorted(root.moves(), key=root.visit_count, reverse=True):
                new_board = move.apply_move(game_state.board)
                if str(new_board) in self.encountered:
                    has_skip = True
                    continue
                self.encountered.add(str(new_board))
                return GameState(new_board, game_state.player.other(), game_state.steps + 1)
            if has_skip:
                # Skipped move is the only valid move, allow it.
                for move in sorted(root.moves(), key=root.visit_count, reverse=True):
                    new_board = move.apply_move(game_state.board)
                    return GameState(new_board, game_state.player.other(), game_state.steps + 1)
        return None

    def create_node(self, game_state: GameState, move=None, parent=None) -> ZeroTreeNode:
        with game_state.board.flipped(game_state.player == Player.black) as board:
            state_tensor = self.encoder.encode(board)
            model_input = np.array([state_tensor])
            priors, values = self.model.predict(model_input)
            priors = priors[0]

            exploration_prob = 0.3
            if np.random.uniform() < exploration_prob:
                priors = np.random.dirichlet(priors)

            value = values[0][0]

            move_priors = {}
            for idx, p in enumerate(priors):
                ds = GameState(board, Player.red, 0)
                mv = self.encoder.decode_move(ds, idx)
                if mv is not None:
                    mv = mv.flip(game_state.player == Player.black)
                    move_priors[mv] = p

        new_node = ZeroTreeNode(game_state, value, move_priors, parent, move)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node

    def finish(self, reward):
        self.collector.assign_reward(reward)

    def train_batch(self, states, policy_targets, value_targets):
        self.model.compile(optimizer=SGD(lr=0.001, clipvalue=0.2),
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
            print("move is none for red")
            winner = Player.black
            break
        game = agent2.select_move(game)
        if game is None:
            print("move is none for black")
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
        agent1.finish(-1)
        agent2.finish(-1)
        print("It's draw %s - %d" % (episode, round))
    file_path = os.path.join(episode, "agz_%s_1.h5" % round)
    with h5py.File(file_path, 'w') as h51:
        collector1.save(h51)
    file_path = os.path.join(episode, "agz_%s_2.h5" % round)
    with h5py.File(file_path, 'w') as h52:
        collector2.save(h52)
