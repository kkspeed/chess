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

PIECE_VALUES = {
    '帅': 1,
    '将': -1,
    '士': 2,
    '仕': -2,
    '相': 3,
    '象': -3,
    '马': 4,
    '馬': -4,
    '车': 5,
    '車': -5,
    '兵': 6,
    '卒': -6,
    '炮': 7,
    '包': -7,
}

PIECE_VALUES_REV = {
    1: '帅',
    -1: '将',
    2: '士',
    -2: '仕',
    3: '相',
    -3: '象',
    4: '马',
    -4: '馬',
    5: '车',
    -5: '車',
    6: '兵',
    -6: '卒',
    7: '炮',
    -7: '包',
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
    def encode(self, board: Board):
        result = np.zeros((1, board.height, board.width))
        for piece in board.pieces:
            result[0][piece.pos.row][piece.pos.col] = PIECE_VALUES[str(piece)]
        return result

    def decode(self, array) -> Board:
        board = Board()
        for row in range(len(array[0])):
            for col in range(len(array[0][0])):
                if array[0][row][col] in PIECE_VALUES_REV:
                    ch = PIECE_VALUES_REV[array[0][row][col]]
                    board.pieces.append(Piece.from_name(Point(row, col), ch))
        return board

    def encode_move(self, state: GameState, move: Move):
        result = np.zeros(TOTAL_MOVES)
        index = self.move_to_index(state, move)
        result[index] = 1
        return result

    def move_to_index(self, state: GameState, move: Move) -> int:
        index = 0
        candidates = sorted(
            filter(lambda piece: piece.color == move.piece.color,
                   state.board.pieces))
        skip = 0
        for c in candidates:
            if c.name() == move.piece.name():
                if c.pos != move.piece.pos:
                    skip += 1
                else:
                    break

        for name, step in MOVE:
            if name != move.piece.name():
                index += step
            else:
                index += skip * step
                break

        index += move.piece.possible_positions().index(move.target)
        return index

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
        candidates = sorted(
            filter(lambda piece: piece.color == state.player, state.board.pieces))
        encounter = 0
        for piece in candidates:
            if piece.name() == move_fig:
                if encounter == fig_num:
                    target = piece.possible_positions()[move_dir]
                    return piece.calc_move(state.board, target)
                encounter += 1
        return None

    def move_mask(self, state: GameState) -> List[int]:
        result = [0] * TOTAL_MOVES
        candidates = sorted(
            filter(lambda piece: piece.color == state.player, state.board.pieces))
        d = collections.defaultdict(list)
        for piece in candidates:
            d[piece.name()].append(piece)
            d[piece.name()].sort()
        start = 0
        for name, step in MOVE:
            if name in d:
                inc = 0
                for piece in d[name]:
                    for pos in piece.possible_positions():
                        next_move = piece.calc_move(state.board, pos)
                        if pos.row >= 0 and pos.row < state.board.height and pos.col >= 0 \
                                and pos.col < state.board.width and next_move is not None:
                            result[start + inc] = 1
                        inc += 1
                del d[name]
            start += step
        return result

    def point_to_index(self, point):
        return point.row * 9 + point.col

    def shape(self):
        return (34, 1)
