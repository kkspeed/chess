import sys
import os
import textwrap
from typing import List, Tuple
import h5py
import agz_agent

from chess_types import GameState, Board, Player, Move, Point, KillMove
from encoder import SimpleEncoder


def simulate_game(moves: List[Tuple[int, int, int, int]]):
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

    game_states = []
    actions = []
    values = []

    encoder = SimpleEncoder()

    state = GameState(board, Player.red)
    for c0, r0, c1, r1 in moves:
        print(f"({9 - r0}, {c0}) -> ({9 - r1}, {c1})")
        piece = state.board.piece_at(Point(9 - r0, c0))
        killed = state.board.piece_at(Point(9 - r1, c1))
        original_move = None
        if killed is not None:
            original_move = KillMove(piece, Point(9 - r1, c1), killed)
        else:
            original_move = Move(piece, Point(9 - r1, c1))
        move = original_move.flip(state.player == Player.black)
        with state.board.flipped(state.player == Player.black) as flipped_board:
            game_states.append(encoder.encode(flipped_board))
            actions.append(encoder.encode_move(
                GameState(flipped_board, Player.red), move))
            # Trust professionals.
            values.append(1)
        new_board = original_move.apply_move(state.board)
        state = GameState(new_board, state.player.other())
        print(state.board)
        print("=========")

    return game_states, actions, values


if __name__ == "__main__":
    file_name = sys.argv[1]
    base, ext = file_name.split(".")
    out_dir = sys.argv[2]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(file_name, "r") as fi:
        for idx, line in enumerate(fi.read().splitlines()):
            moves = []
            for i in range(0, len(line), 4):
                moves.append(map(int, line[i:i+4]))
            collector = agz_agent.AgzExpCollector()
            states, actions, values = simulate_game(moves)
            collector.states = states
            collector.actions = actions
            collector.rewards = values
            with h5py.File(os.path.join(out_dir,  f"{base}_{idx}.h5")) as hf:
                collector.save(hf)
