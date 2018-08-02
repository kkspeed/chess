from chess_types import *

import copy
import collections
import numpy as np

BOARD = ['帅', '将', '士', '仕', '相', '象', '马', '馬', '车', '車', '兵', '卒', '炮', '包']

DEFAULT = {
    '帅': [-1],
    '将': [-1],
    '士': [-1, -1],
    '仕': [-1, -1],
    '相': [-1, -1],
    '象': [-1, -1],
    '马': [-1, -1],
    '馬': [-1, -1],
    '车': [-1, -1],
    '車': [-1, -1],
    '兵': [-1, -1, -1, -1, -1],
    '卒': [-1, -1, -1, -1, -1],
    '炮': [-1, -1],
    '包': [-1, -1]
}

MOVE = [
 ('帅', 4),
 ('士', 4),
 ('士', 4),
 ('相', 4),
 ('相', 4),
 ('马', 8),
 ('马', 8),
 ('车', 17),
 ('车', 17),
 ('炮', 17),
 ('炮', 17),
 ('兵', 3),
 ('兵', 3),
 ('兵', 3),
 ('兵', 3),
 ('兵', 3),
]

TOTAL_MOVES = sum([m[1] for m in MOVE])

class SimpleEncoder:
    def encode(self, state):
        d = copy.deepcopy(DEFAULT)
        for p in state.board.pieces:
            name = str(p)
            i = d[name].index(-1)
            d[name][i] = self.point_to_index(p.pos)
            d[name].sort()
        result = []
        for key in BOARD:
            result.extend(d[key])
        result.append(1 if state.player == Player.red else 0)
        result.append(state.steps)
        return np.array(result)

    def decode(self, array) -> GameState:
        board = Board()
        index = 0
        for name in BOARD:
            for _ in range(len(DEFAULT[name])):
                elem = array[index]
                if elem >= 0:
                    col = elem % 9
                    row = elem // 9
                    piece = Piece.from_name(Point(row, col), name)
                    board.pieces.append(piece)
                index += 1

        player = Player.red if array[-2] == 1 else Player.black
        return GameState(board, player, array[-1])

    def decode_move(self, state: GameState, index: int) -> Move:
        start = 0
        passed = collections.defaultdict(int)
        move_fig = None
        fig_num = -1
        move_dir = -1
        for fig, step in MOVE:
            if index < start + step:
                move_fig = fig
                fig_num = passed[fig]
                move_dir = index - start
                break
            else:
                start += step
                passed[fig] += 1
        assert fig_num != -1 and move_dir != -1
        return self.gen_move(state, move_fig, fig_num, move_dir)

    def gen_move(self, state: GameState, move_fig: str, fig_num: int, move_dir: int) -> Move:
        candidates = sorted(filter(lambda piece: piece.color == state.player, state.board.pieces))
        encounter = 0
        for piece in candidates:
            if str(piece.__class__) == move_fig:
                if encounter == fig_num:
                    target = piece.possible_positions[move_dir]
                    return piece.calc_move(state.board, target)
                encounter += 1
        raise ValueError("Should not reach") 

    def point_to_index(self, point):
        return point.row * 9 + point.col

    def shape(self):
        return (34, 1)
