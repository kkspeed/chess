import agent
import os
import sys
import h5py

from chess_types import Player


def test(model1, model2):
    agent1 = agent.Agent(Player.red, agent.ExpCollector())
    if model1 is not None:
        agent1.model.load_weights(model1)
    agent2 = agent.Agent(Player.black, agent.ExpCollector())
    if model2 is not None:
        agent2.model.load_weights(model2)
    red = 0
    black = 0
    for i in range(20):
        winner = agent.game_play(agent1, agent2)
        if winner == Player.red:
            red += 1
        if winner == Player.black:
            black += 1
    print("Red win %d, Black win %d" % (red, black))


def debug_play(model1, model2):
    c1 = agent.ExpCollector()
    agent1 = agent.Agent(Player.red, c1)
    agent1.model.load_weights(model1)
    c2 = agent.ExpCollector()
    agent2 = agent.Agent(Player.black, c2)
    agent2.model.load_weights(model2)
    print("Winner: ", agent.game_play(agent1, agent2))
    agent1.finish(0)
    agent2.finish(0)

    file_path = "debug_1_%s.h5" % model2
    h51 = h5py.File(file_path, 'w')
    c1.save(h51)
    h51.close()
    file_path = "debug_2_%s.h5" % model2
    h52 = h5py.File(file_path, 'w')
    c2.save(h52)
    h52.close()


def train(epoch, model=None):
    poly_agent = agent.Agent(Player.red, agent.ExpCollector())
    if model:
        poly_agent.model.load_weights(model)
    for f in os.listdir(epoch):
        path = os.path.join(epoch, f)
        print('train on ', path)
        with h5py.File(path, 'r') as h5:
            exp = agent.ExpCollector()
            exp.load(h5)
            poly_agent.train(exp)
    new_model = "model_%s.h5" % epoch
    poly_agent.model.save_weights(new_model)
    return new_model


def train_batch(epoch, model=None):
    import numpy as np
    poly_agent = agent.Agent(Player.red, agent.ExpCollector())
    if model:
        poly_agent.model.load_weights(model)
    inputs = []
    target = []
    for f in os.listdir(epoch):
        path = os.path.join(epoch, f)
        print('train on ', path)
        with h5py.File(path, 'r') as h5:
            exp = agent.ExpCollector()
            exp.load(h5)
            inputs.append(exp.inputs)
            target.append(agent.prepare_experience_data(exp))
    poly_agent.train_batch(np.concatenate(inputs), np.concatenate(target))
    new_model = "model_%s.h5" % epoch
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
        last_model = 'model_%d.h5' % last_num
    else:
        last_num = 0
    for epoch in range(last_num + 1, last_num + max_num):
        agent1 = agent.Agent(Player.red, None)
        agent2 = agent.Agent(Player.black, None)
        if last_model:
            agent1.model.load_weights(last_model)
            agent2.model.load_weights(last_model)
        for round in range(1000):
            print("===\n\nPlaying epoch: %d, round %d\n\n===" % (epoch, round))
            agent1.encountered = set()
            agent2.encountered = set()
            agent.self_play(str(epoch), round, agent1, agent2)
        last_model = train_batch(str(epoch), last_model)
