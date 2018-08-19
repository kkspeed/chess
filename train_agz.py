import os
import sys
import h5py

from chess_types import Player
import agz_agent

def train_batch(epoch, model=None):
    import numpy as np
    poly_agent = agz_agent.ZeroAgent(Player.red, None)
    if model:
        poly_agent.model.load_weights(model)
    inputs = []
    target = []
    rewards = []
    for f in os.listdir(epoch):
        path = os.path.join(epoch, f)
        print('train on ', path)
        with h5py.File(path, 'r') as h5:
            exp = agz_agent.AgzExpCollector()
            exp.load(h5)
            inputs.append(exp.states)
            target.append(exp.actions)
            rewards.append(exp.rewards)
    poly_agent.train_batch(np.concatenate(inputs), np.concatenate(target), np.concatenate(rewards))
    new_model = "agz_model_%s.h5" % epoch
    poly_agent.model.save_weights(new_model)
    return new_model


if __name__ == "__main__":
    last_num = None
    if len(sys.argv) >= 2:
        last_num = int(sys.argv[1])
    max_num = 20
    if len(sys.argv) >= 3:
        max_num = int(sys.argv[2])
    last_model = None
    if last_num is not None and last_num != 0:
        last_model = 'agz_model_%d.h5' % last_num
    else:
        last_num = 0
    for epoch in range(last_num + 1, last_num + max_num):
        agent1 = agz_agent.ZeroAgent(Player.red, None)
        agent2 = agz_agent.ZeroAgent(Player.black, None)
        if last_model:
            agent1.model.load_weights(last_model)
            agent2.model.load_weights(last_model)
        for round in range(1000):
            print("===\n\nPlaying epoch: %d, round %d\n\n===" % (epoch, round))
            agent1.encountered = set()
            agent2.encountered = set()
            agz_agent.self_play('agz_' + str(epoch), round, agent1, agent2)
        last_model = train_batch('agz_' + str(epoch), last_model)
