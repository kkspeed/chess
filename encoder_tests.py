from encoder import *
from chess_types import *

import unittest
import textwrap


class EncoderTest(unittest.TestCase):
    def test_encoder(self):
        b = Board()
        b.parse_from_string(textwrap.dedent("""\
            ....将....
            .........
            .........
            ......車..
            .........
            .........
            兵........
            ......车..
            .........
            ....帅...."""))
        encoder = SimpleEncoder()
        mat = encoder.encode(b)
        self.assertTrue(np.shape(mat) == (10, 9))
        decoded_board = encoder.decode(mat)
        self.assertTrue(len(decoded_board.pieces) == 5)
        self.assertTrue(set(decoded_board.pieces)
                        == set(b.pieces))

    def test_move_selection(self):
        b = Board()
        b.parse_from_string(textwrap.dedent("""\
            ....将....
            .........
            .........
            ......車..
            .........
            .........
            兵........
            ......车..
            .........
            ....帅...."""))
        state = GameState(b, Player.red, 12)
        encoder = SimpleEncoder()
        self.assertTrue(encoder.decode_move(state, 3).target == Point(9, 3))
        state.player = Player.black
        self.assertTrue(encoder.decode_move(state, 3).target == Point(0, 3))

    def test_move_mask(self):
        b = Board()
        b.parse_from_string(textwrap.dedent("""\
            ....将....
            .........
            .........
            ......車..
            .........
            .........
            兵........
            ....相.车..
            ....帅....
            ..相士车...."""))
        state = GameState(b, Player.red, 12)
        encoder = SimpleEncoder()
        move_mask = encoder.move_mask(state)
        self.assertEqual(move_mask[:4], [0, 0, 1, 1])
        self.assertEqual(move_mask[4:8], [0, 0, 0, 0])
        self.assertEqual(move_mask[16:20], [0, 0, 0, 1])

if __name__ == "__main__":
    unittest.main()
