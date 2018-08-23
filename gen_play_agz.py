import os
import sys
import h5py

from chess_types import Player
import agz_agent


if __name__ == "__main__":
    start_round = int(sys.argv[1])
    num_round = int(sys.argv[2])
    model = int(sys.argv[3])
    epoch = int(sys.argv[4])
    if model > 0:
        model = 'agz_model_agz_%d.h5' % model
    agent1 = agz_agent.ZeroAgent(Player.red, None)
    agent2 = agz_agent.ZeroAgent(Player.black, None)
    if model:
        agent1.model.load_weights(model)
        agent2.model.load_weights(model)
    for round in range(start_round, start_round + num_round):
        print("===\n\nPlaying epoch: %d, round %d\n\n===" % (epoch, round))
        agent1.encountered = set()
        agent2.encountered = set()
        agz_agent.self_play('agz_' + str(epoch), round, agent1, agent2)
