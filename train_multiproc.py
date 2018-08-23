import os
import sys
import h5py
import subprocess

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
    poly_agent.train_batch(np.concatenate(inputs), np.concatenate(target),
            np.concatenate(rewards))
    new_model = "agz_model_%s.h5" % epoch
    poly_agent.model.save_weights(new_model)
    return new_model


if __name__ == "__main__":
    start_epoch = 0
    if len(sys.argv) > 1:
        start_epoch = int(sys.argv[1])

    for epoch in range(start_epoch, 10):
        proc = []
        step = 5
        end = 100
        for start in range(0, end, step):
            p = subprocess.Popen(["python", "gen_play_agz.py", str(start), str(step), str(epoch), 
                    str(epoch + 1), ">", "agz_train_%d.log" % (start // step)])
            proc.append(p)
        for p in proc:
            print(len(proc), " left, ", "epoch: ", epoch)
            p.wait()

        last_model = None
        if epoch > 0:
            last_model = 'agz_model_agz_%d.h5' % epoch
        last_model = train_batch('agz_' + str(epoch + 1), last_model)
