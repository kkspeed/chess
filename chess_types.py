from typing import List
from collections import namedtuple
import copy
import enum


BOARD_HEIGHT = 10

class Player(enum.Enum):
    red = 1
    black = 2

    def other(self):
        return Player.red if self == Player.black else Player.black


# pylint: disable=C0103
class Point(namedtuple('Point', 'row col')):
    pass


def check_valid_move(func):
    """
    Ensure the move applied to |func| is globally valid.
    """

    def calc_move(self, board: 'Board', target: Point) -> 'Move':
        if target.row >= board.height or target.row < 0 or \
                target.col >= board.width or target.col < 0:
            return None
        move = func(self, board, target)
        if move is None:
            return None
        with board.mutable() as mut_board:
            mut_board.move_piece(move)
            ps = mut_board.pieces_by_strs('帅', '将')
            if len(ps) == 2:
                k1, k2 = ps
                if k1.pos.col == k2.pos.col:
                    not_face = any([mut_board.piece_at(Point(r, k1.pos.col))
                                    for r in range(min(k1.pos.row, k2.pos.row) + 1,
                                                   max(k1.pos.row, k2.pos.row))])
                    move = (not_face or None) and move
        return move
    calc_move.__name__ = func.__name__
    calc_move.__doc__ = func.__doc__
    return calc_move


class Piece:
    def __init__(self, pos: Point, color: Player):
        self.pos = pos
        self.color = color

    def possible_positions(self) -> List[Point]:
        """
        Return all the possible positions for this piece
        """
        raise NotImplementedError

    def possible_moves(self, board):
        result = []
        for target in self.possible_positions():
            move = self.calc_move(board, target)
            if move is not None:
                result.append(move)
        return result

    def calc_move(self, board, target):
        raise NotImplementedError

    @classmethod
    def from_name(cls, pos: Point, ch: str):
        if ch == "帅":
            return 帅(pos, Player.red)
        if ch == "将":
            return 帅(pos, Player.black)
        if ch == "士":
            return 士(pos, Player.red)
        if ch == "仕":
            return 士(pos, Player.black)
        if ch == "相":
            return 相(pos, Player.red)
        if ch == "象":
            return 相(pos, Player.black)
        if ch == "马":
            return 马(pos, Player.red)
        if ch == "馬":
            return 马(pos, Player.black)
        if ch == "车":
            return 车(pos, Player.red)
        if ch == "車":
            return 车(pos, Player.black)
        if ch == "炮":
            return 炮(pos, Player.red)
        if ch == "包":
            return 炮(pos, Player.black)
        if ch == "兵":
            return 兵(pos, Player.red)
        if ch == "卒":
            return 兵(pos, Player.black)
        raise NotImplementedError

    def name(self):
        return str(self.__class__.__name__)

    def __eq__(self, other):
        return self.color == other.color and self.pos == other.pos and\
            str(self.__class__) == str(other.__class__)

    def __lt__(self, other):
        if self.pos != other.pos:
            return self.pos < other.pos
        return str(self.__class__) < str(other.__class__)

    def __hash__(self):
        return hash((self.color, self.pos, str(self.__class__)))


class 帅(Piece):
    def possible_positions(self):
        return [Point(self.pos.row + 1, self.pos.col),
                Point(self.pos.row - 1, self.pos.col),
                Point(self.pos.row, self.pos.col + 1),
                Point(self.pos.row, self.pos.col - 1)]

    @check_valid_move
    def calc_move(self, board, target):
        if self.color == Player.red:
            if target.row <= 6 or target.col <= 2 or target.col >= 6:
                return None
        if self.color == Player.black:
            if target.row >= 3 or target.col <= 2 or target.col >= 6:
                return None

        piece = board.piece_at(target)

        if piece is None:
            return Move(self, target)
        if piece.color != self.color:
            return KillMove(self, target, piece)
        return None

    def __str__(self):
        if self.color == Player.red:
            return "帅"
        return "将"


class 士(Piece):
    def possible_positions(self):
        return [Point(self.pos.row + 1, self.pos.col + 1),
                Point(self.pos.row + 1, self.pos.col - 1),
                Point(self.pos.row - 1, self.pos.col + 1),
                Point(self.pos.row - 1, self.pos.col - 1)]

    @check_valid_move
    def calc_move(self, board, target):
        if self.color == Player.red:
            if target.row <= 6 or target.col <= 2 or target.col >= 6:
                return None
        if self.color == Player.black:
            if target.row >= 3 or target.col <= 2 or target.col >= 6:
                return None

        piece = board.piece_at(target)
        if piece is None:
            return Move(self, target)
        if piece.color != self.color:
            return KillMove(self, target, piece)
        return None

    def __str__(self):
        return "士" if self.color == Player.red else "仕"


class 相(Piece):
    def possible_positions(self):
        return [Point(self.pos.row + 2, self.pos.col + 2),
                Point(self.pos.row + 2, self.pos.col - 2),
                Point(self.pos.row - 2, self.pos.col + 2),
                Point(self.pos.row - 2, self.pos.col - 2)]

    @check_valid_move
    def calc_move(self, board, target):
        if self.color == Player.red:
            if target.row < 5:
                return None
        if self.color == Player.black:
            if target.row > 4:
                return None

        mid_r, mid_c = (
            self.pos.row + target.row) // 2, (self.pos.col + target.col) // 2
        if board.piece_at(Point(mid_r, mid_c)) is not None:
            return None

        piece = board.piece_at(target)
        if piece is None:
            return Move(self, target)
        if piece.color != self.color:
            return KillMove(self, target, piece)
        return None

    def __str__(self):
        return "相" if self.color == Player.red else "象"


class 马(Piece):
    def possible_positions(self):
        return [Point(self.pos.row + 2, self.pos.col + 1),
                Point(self.pos.row + 2, self.pos.col - 1),
                Point(self.pos.row - 2, self.pos.col + 1),
                Point(self.pos.row - 2, self.pos.col - 1),
                Point(self.pos.row + 1, self.pos.col + 2),
                Point(self.pos.row - 1, self.pos.col + 2),
                Point(self.pos.row + 1, self.pos.col - 2),
                Point(self.pos.row - 1, self.pos.col - 2)]

    @check_valid_move
    def calc_move(self, board, target):
        check = None
        if abs(target.row - self.pos.row) == 1:
            check = self.pos.row, (self.pos.col + target.col) // 2
        if abs(target.col - self.pos.col) == 1:
            check = (self.pos.row + target.row) // 2, self.pos.col
        assert check is not None
        if board.piece_at(check) is not None:
            return None

        piece = board.piece_at(target)
        if piece is None:
            return Move(self, target)
        if piece.color != self.color:
            return KillMove(self, target, piece)
        return None

    def __str__(self):
        return "马" if self.color == Player.red else "馬"


class 车(Piece):
    def possible_positions(self):
        positions = set(Point(self.pos.row, col) for col in range(0, 9)) | \
            set(Point(row, self.pos.col) for row in range(0, 10))
        positions.remove(self.pos)
        return list(sorted(positions))

    @check_valid_move
    def calc_move(self, board, target):
        if self.pos.row == target.row:
            for c in range(min(self.pos.col, target.col) + 1,
                           max(self.pos.col, target.col)):
                if board.piece_at(Point(self.pos.row, c)) is not None:
                    return None
        if self.pos.col == target.col:
            for r in range(min(self.pos.row, target.row) + 1,
                           max(self.pos.row, target.row)):
                if board.piece_at(Point(r, self.pos.col)) is not None:
                    return None

        piece = board.piece_at(target)
        if piece is None:
            return Move(self, target)
        if piece.color != self.color:
            return KillMove(self, target, piece)
        return None

    def __str__(self):
        return "车" if self.color == Player.red else "車"


class 炮(Piece):
    def possible_positions(self):
        positions = set(Point(self.pos.row, col) for col in range(0, 9)) | \
            set(Point(row, self.pos.col) for row in range(0, 10))
        positions.remove(self.pos)
        return list(sorted(positions))

    @check_valid_move
    def calc_move(self, board, target):
        piece = board.piece_at(target)
        if self.pos.row == target.row:
            count = 0
            for c in range(min(self.pos.col, target.col) + 1,
                           max(self.pos.col, target.col)):
                if board.piece_at(Point(self.pos.row, c)) is not None:
                    count += 1
            if count > 1:
                return None
            if count == 1 and piece is None:
                return None
            if count == 1 and piece is not None and piece.color == self.color:
                return None
            if count == 0 and piece is not None:
                return None
        if self.pos.col == target.col:
            count = 0
            for r in range(min(self.pos.row, target.row) + 1,
                           max(self.pos.row, target.row)):
                if board.piece_at(Point(r, self.pos.col)) is not None:
                    count += 1
            if count > 1:
                return None
            if count == 1 and piece is None:
                return None
            if count == 1 and piece is not None and piece.color == self.color:
                return None
            if count == 0 and piece is not None:
                return None
        if piece is None:
            return Move(self, target)
        if piece.color != self.color:
            return KillMove(self, target, piece)
        return None

    def __str__(self):
        return "炮" if self.color == Player.red else "包"


class 兵(Piece):
    def possible_positions(self):
        delta = 1 if self.color == Player.black else -1
        return [
            Point(self.pos.row + delta, self.pos.col),
            Point(self.pos.row, self.pos.col + 1),
            Point(self.pos.row, self.pos.col - 1)]

    def __str__(self):
        return "兵" if self.color == Player.red else "卒"

    @check_valid_move
    def calc_move(self, board, target):
        过河 = (self.pos.row >= 5 and self.color == Player.black) or\
            (self.pos.row <= 4 and self.color == Player.red)
        if not 过河 and target.col != self.pos.col:
            return None
        piece = board.piece_at(target)
        if piece is None:
            return Move(self, target)
        if piece.color != self.color:
            return KillMove(self, target, piece)
        return None


class Board:
    def __init__(self):
        self.pieces = []
        self.width = 0
        self.height = 0

    def parse_from_string(self, s: str):
        lines = s.split('\n')
        self.height = len(lines)
        self.width = len(lines[0])
        for row, line in enumerate(lines):
            for col, ch in enumerate(line):
                if ch != '.':
                    self.pieces.append(Piece.from_name(Point(row, col), ch))

    def piece_at(self, point):
        for piece in self.pieces:
            if piece.pos == point:
                return piece
        return None

    def pieces_by_strs(self, *strs):
        strs = set(strs)
        return [piece for piece in self.pieces if str(piece) in strs]

    def mutable(self) -> 'MutableBoard':
        """
        Returns a board that is internally mutable. It's supposed to be
        used in with statement.
        """
        return MutableBoard(self)

    def flipped(self, flip: bool) -> 'FlippedBoard':
        return FlippedBoard(self, flip)

    def __str__(self):
        matrix = [['.' for c in range(self.width)]
                  for r in range(self.height)]
        for p in self.pieces:
            matrix[p.pos.row][p.pos.col] = str(p)
        return '\n'.join(map(''.join, matrix))

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


class FlippedBoard(Board):
    def __init__(self, board: Board, do_flip: bool):
        self.pieces = board.pieces
        self.height = board.height
        self.width = board.width
        self.do_flip = do_flip

    def __enter__(self):
        if self.do_flip:
            for p in self.pieces:
                p.color = p.color.other()
                p.pos = Point(self.height - p.pos.row - 1, p.pos.col)
        return self

    def __exit__(self, exception_type, exception_val, trace_back):
        if self.do_flip:
            for p in self.pieces:
                p.color = p.color.other()
                p.pos = Point(self.height - p.pos.row - 1, p.pos.col)


class MutableBoard(Board):
    def __init__(self, board: Board):
        self.pieces = board.pieces
        self.height = board.height
        self.width = board.width
        self.moves = []

    def move_piece(self, move: 'Move'):
        self.moves.append((copy.copy(move.piece.pos), move))
        if type(move) is KillMove:
            self.pieces.remove(move.killed)
        piece = move.piece
        piece.pos = move.target

    def __enter__(self):
        assert len(self.moves) == 0
        return self

    def __exit__(self, exception_type, exception_val, trace_back):
        while len(self.moves) > 0:
            pos, move = self.moves.pop()
            piece = self.piece_at(move.target)
            piece.pos = pos
            if isinstance(move, KillMove):
                self.pieces.append(move.killed)


class Move:
    def __init__(self, piece: Piece, target: Point):
        self.piece = piece
        self.target = target

    def apply_move(self, board: Board) -> Board:
        result = copy.deepcopy(board)
        index = result.pieces.index(self.piece)
        result.pieces[index].pos = self.target
        return result

    def flip(self):
        self.target = Point(BOARD_HEIGHT - self.target.row - 1, self.target.col)

    def __str__(self):
        return "M"

    def __hash__(self):
        return hash((self.piece, self.target, type(self)))

    def __eq__(self, other):
        return self.piece == other.piece and self.target == other.target and type(self) == type(other)


class KillMove(Move):
    def __init__(self, piece: Piece, target: Point, killed: Piece):
        super().__init__(piece, target)
        self.killed = killed

    def apply_move(self, board: Board) -> Board:
        result = copy.deepcopy(board)
        result.pieces.remove(self.killed)
        return super().apply_move(result)

    def __str__(self):
        return "X"


def plot_move(move, plot):
    plot[move.piece.pos.row][move.piece.pos.col] = str(move.piece)
    plot[move.target.row][move.target.col] = str(move)


def visualize_moves(moves: List[Move]) -> str:
    result = [['.' for c in range(9)] for r in range(10)]
    for move in moves:
        plot_move(move, result)
    return '\n'.join(map(''.join, result))


class GameState:
    def __init__(self, board: Board, player: Player, steps=0):
        self.board = board
        self.player = player
        self.steps = steps

    def is_win(self, player):
        other = player.other()
        win = True
        for piece in self.board.pieces:
            if type(piece) is 帅 and piece.color == other:
                win = False
        return win

    def winner(self):
        # Draw when steps > 600
        if self.steps > 600:
            return -1
        for p in [Player.red, Player.black]:
            if self.is_win(p):
                return p
        return None
