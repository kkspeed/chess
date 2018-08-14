import agent_ac
import os
import sys
import h5py

from chess_types import Player


def test(model1, model2):
    agent1 = agent_ac.AcAgent(Player.red, None)
    if model1 is not None:
        agent1.model.load_weights(model1)
    agent2 = agent_ac.AcAgent(Player.black, None)
    if model2 is not None:
        agent2.model.load_weights(model2)
    red = 0
    black = 0
    for i in range(100):
        winner = agent_ac.game_play(agent1, agent2)
        if winner == Player.red:
            red += 1
        if winner == Player.black:
            black += 1
        print("Playing: ", i, "of", 100, "red:", red, "black:", black)
    print("Red win %d, Black win %d" % (red, black))


if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2])
