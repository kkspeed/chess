import textwrap
import unittest

from chess_types import *


class TestBoard(unittest.TestCase):
    def test_serialize_and_deserialize(self):
        board = Board()
        board_string = textwrap.dedent("""\
            ..帅..
            ..士..
            .车車馬.
            马相象仕.
            炮包兵卒.""")
        board.parse_from_string(board_string)
        self.assertEqual(str(board), board_string)


class TestMove(unittest.TestCase):
    def test_帅(self):
        board = Board()
        board.parse_from_string(textwrap.dedent("""\
            .........
            .........
            .........
            .........
            .........
            .........
            .........
            .........
            ...帅士....
            ...車....."""))
        piece = board.pieces[0]
        moves = piece.possible_moves(board)
        self.assertEqual(textwrap.dedent("""\
            .........
            .........
            .........
            .........
            .........
            .........
            .........
            ...M.....
            ...帅.....
            ...X....."""), visualize_moves(moves))

    def test_士(self):
        board = Board()
        board.parse_from_string(textwrap.dedent("""\
            .........
            .........
            ...仕.....
            .........
            .........
            .........
            .........
            .........
            ....士....
            ...車.帅..."""))
        piece = board.pieces[1]
        moves = piece.possible_moves(board)
        self.assertEqual(textwrap.dedent("""\
            .........
            .........
            .........
            .........
            .........
            .........
            .........
            ...M.M...
            ....士....
            ...X....."""), visualize_moves(moves))
        piece = board.pieces[0]
        moves = piece.possible_moves(board)
        self.assertEqual(textwrap.dedent("""\
            .........
            ....M....
            ...仕.....
            .........
            .........
            .........
            .........
            .........
            .........
            ........."""), visualize_moves(moves))

    def test_相(self):
        board = Board()
        board.parse_from_string(textwrap.dedent("""\
            .........
            .........
            .........
            .........
            .........
            ..相......
            ...马.....
            .........
            .........
            ........."""))
        piece = board.pieces[0]
        moves = piece.possible_moves(board)
        self.assertEqual(textwrap.dedent("""\
            .........
            .........
            .........
            .........
            .........
            ..相......
            .........
            M........
            .........
            ........."""), visualize_moves(moves))

    def test_马(self):
        board = Board()
        board.parse_from_string(textwrap.dedent("""\
            .........
            .........
            .........
            .........
            .........
            ...车.....
            ...马兵....
            .車...炮...
            ..卒......
            ........."""))
        piece = board.pieces[1]
        moves = piece.possible_moves(board)
        self.assertEqual(textwrap.dedent("""\
            .........
            .........
            .........
            .........
            .........
            .M.......
            ...马.....
            .X.......
            ..X.M....
            ........."""), visualize_moves(moves))

    def test_车(self):
        board = Board()
        board.parse_from_string(textwrap.dedent("""\
            .........
            ...将.....
            .........
            ...兵.....
            .........
            車..车..兵..
            .........
            .車...炮...
            ..卒卒.....
            ........."""))
        piece = board.pieces[3]
        moves = piece.possible_moves(board)
        self.assertEqual(textwrap.dedent("""\
            .........
            .........
            .........
            .........
            ...M.....
            XMM车MM...
            ...M.....
            ...M.....
            ...X.....
            ........."""), visualize_moves(moves))

    def test_炮(self):
        board = Board()
        board.parse_from_string(textwrap.dedent("""\
            .........
            ...将.....
            .........
            ...兵.....
            .........
            車..炮..兵马.
            .........
            .車...炮...
            ..卒卒.....
            ........."""))
        piece = board.pieces[3]
        moves = piece.possible_moves(board)
        self.assertEqual(textwrap.dedent("""\
            .........
            ...X.....
            .........
            .........
            ...M.....
            .MM炮MM...
            ...M.....
            ...M.....
            .........
            ........."""), visualize_moves(moves))

    def test_兵(self):
        board = Board()
        board.parse_from_string(textwrap.dedent("""\
            .........
            .........
            .........
            ..卒兵.....
            .........
            .........
            ........兵
            .........
            ...卒.....
            ...车....."""))
        piece = board.pieces[1]
        moves = piece.possible_moves(board)
        self.assertEqual(textwrap.dedent("""\
            .........
            .........
            ...M.....
            ..X兵M....
            .........
            .........
            .........
            .........
            .........
            ........."""), visualize_moves(moves))
        piece = board.pieces[0]
        moves = piece.possible_moves(board)
        self.assertEqual(textwrap.dedent("""\
            .........
            .........
            .........
            ..卒......
            ..M......
            .........
            .........
            .........
            .........
            ........."""), visualize_moves(moves))
        piece = board.pieces[3]
        moves = piece.possible_moves(board)
        self.assertEqual(textwrap.dedent("""\
            .........
            .........
            .........
            .........
            .........
            .........
            .........
            .........
            ..M卒M....
            ...X....."""), visualize_moves(moves))


class TestApplyMoves(unittest.TestCase):
    def test_apply_killer_move(self):
        board = Board()
        board.parse_from_string(textwrap.dedent("""\
            .........
            ..车...車.."""))
        move = KillMove(board.pieces[0], Point(1, 6), board.pieces[1])
        new_board = move.apply_move(board)
        self.assertEqual(str(new_board), textwrap.dedent("""\
            .........
            ......车.."""))

    def test_apply_move(self):
        board = Board()
        board.parse_from_string(textwrap.dedent("""\
            .炮...
            ....."""))
        move = Move(board.pieces[0], Point(0, 3))
        new_board = move.apply_move(board)
        self.assertEqual(str(new_board), textwrap.dedent("""\
            ...炮.
            ....."""))


if __name__ == "__main__":
    unittest.main()
