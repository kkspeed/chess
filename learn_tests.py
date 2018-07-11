from learn import *
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
        state = GameState(b, Player.red, 12)
        encoder = SimpleEncoder()
        mat = encoder.encode(state)
        self.assertTrue(len(mat) == 34)
        self.assertTrue(mat[0] == 9 * 9 + 4)
        self.assertTrue(mat[1] == 0 * 9 + 4)
        # Steps:
        self.assertTrue(mat[33] == 12)
        # Player:
        self.assertTrue(mat[32] == 1)

        decoded_state = encoder.decode(mat)
        self.assertTrue(decoded_state.player == state.player)
        self.assertTrue(len(decoded_state.board.pieces) == 5)
        self.assertTrue(decoded_state.steps == state.steps)
        self.assertTrue(set(decoded_state.board.pieces)
                        == set(state.board.pieces))


if __name__ == "__main__":
    unittest.main()
