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

    def point_to_index(self, point):
        return point.row * 9 + point.col

    def shape(self):
        return (34, 1)
