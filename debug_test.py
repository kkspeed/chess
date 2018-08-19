import agent
import agent_ac
import agz_agent
import os
import sys
import h5py

from chess_types import Player


def create_agent(model, player: Player):
    if model is None:
        return agent.Agent(player, None)
    if 'ac' in model:
        a = agent_ac.AcAgent(player, None)
        a.model.load_weights(model)
        return a
    if 'agz' in model:
        a = agz_agent.ZeroAgent(player, None)
        a.model.load_weights(model)
        return a
    a = agent.Agent(player, None)
    a.model.load_weights(model)
    return a


def test(model1, model2):
    agent1 = create_agent(model1, Player.red)
    agent2 = create_agent(model2, Player.black)
    red = 0
    black = 0
    for i in range(100):
        winner = agent.game_play(agent1, agent2)
        if winner == Player.red:
            red += 1
        if winner == Player.black:
            black += 1
        print("Playing: ", i, "of", 100, "red:", red, "black:", black)
    print("Red win %d, Black win %d" % (red, black))


if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2])
