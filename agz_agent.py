import numpy as np

from chess_types import GameState, Move
from encoder import SimpleEncoder
from typing import List, Dict
from model_ac import create_model

class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0

class ZeroTreeNode:
    def __init__(self, state: GameState, value, priors: Dict[Move, float], parent: ZeroTreeNode, last_move: Move):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches: Dict[Move, Branch] = {}
        for move, p in priors.items():
            self.branches[move] = Branch(p)
        self.children = {}

    def moves(self) -> List[Move]:
        return self.branches.keys()

    def add_child(self, move, child_node):
        self.children[move] = child_node

    def has_child(self, move):
        return move in self.children

    def get_child(self, move) -> ZeroTreeNode:
        return self.children[move]

    def expected_value(self, move: Move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, move: Move):
        return self.branches[move].move

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0

    def record_visit(self, move: Move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value


class ZeroAgent:
    def __init__(self):
        self.c = 0.3
        self.num_rounds = 10
        self.encoder = SimpleEncoder()
        self.model = create_model()

    def select_branch(self, node: ZeroTreeNode) -> Move:
        total_n = node.total_visit_count
        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return q + self.c * p * np.sqrt(total_n) / (n + 1)
        return max(node.moves(), key=score_branch)

    def select_move(self, game_state: GameState) -> GameState:
        root = self.create_node(game_state)
        # TODO: flip black / red sides when selecting move, revisiting exp collector

        for i in range(self.num_rounds):
            node = root
            next_move = self.select_branch(node)
            while node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node)
            new_board = next_move.apply_move(node.state.board)
            new_state = GameState(new_board, node.state.player.other(), node.state.steps + 1)
            child_node = self.create_node(new_state, parent=None)
            move = next_move
            value = -1 * child_node.value
            while node is not None:
                node.record_visit(move, value)
                move = node.last_move
                node = node.parent
                value = -1 * value
        move = max(root.moves(), key=root.visit_count)
        return move.apply_move(game_state)

    def create_node(self, game_state: GameState, move=None, parent=None) -> ZeroTreeNode:
        state_tensor = self.encoder.encode(game_state.board)
        model_input = np.array([state_tensor])
        priors, values = self.model.predict(model_input)
        priors = priors[0]
        value = values[0][0]

        move_priors = {}
        for idx, p in enumerate(priors):
            move = self.encoder.decode_move(game_state, idx)
            if move is not None:
                move_priors[move] = p

        new_node = ZeroTreeNode(game_state, value, move_priors, parent, move)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node
