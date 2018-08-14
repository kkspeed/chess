import agent_ac
import os
import sys
import h5py

from chess_types import Player


def train_batch_ac(epoch, model=None):
    import numpy as np
    poly_agent = agent_ac.AcAgent(Player.red, agent_ac.AcExpCollector())
    if model:
        poly_agent.model.load_weights(model)
    inputs = []
    target = []
    values = []
    for f in os.listdir(epoch):
        path = os.path.join(epoch, f)
        print('train on ', path)
        with h5py.File(path, 'r') as h5:
            exp = agent_ac.AcExpCollector()
            exp.load(h5)
            inputs.append(exp.states)
            t, v = agent_ac.prepare_experience_data(exp)
            target.append(t)
            values.append(v)
    poly_agent.train_batch(np.concatenate(
        inputs), np.concatenate(target), np.concatenate(values))
    new_model = "ac_model_%s.h5" % epoch
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
        last_model = 'ac_model_ac_%d.h5' % last_num
    else:
        last_num = 0
    for epoch in range(last_num + 1, last_num + max_num):
        agent1 = agent_ac.AcAgent(Player.red, None)
        agent2 = agent_ac.AcAgent(Player.black, None)
        if last_model:
            agent1.model.load_weights(last_model)
            agent2.model.load_weights(last_model)
        for round in range(1000):
            print("===\n\nPlaying epoch: %d, round %d\n\n===" % (epoch, round))
            agent1.encountered = set()
            agent2.encountered = set()
            agent_ac.self_play('ac_' + str(epoch), round, agent1, agent2)
        last_model = train_batch_ac('ac_' + str(epoch), last_model)
