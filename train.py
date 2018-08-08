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


def train(epoch, model = None):
	poly_agent = agent.Agent(Player.red, agent.ExpCollector())
	if model:
		poly_agent.model.load_weights(model)
	for f in os.listdir(epoch):
		path = os.path.join(epoch, f)
		print('train on ', path)
		h5 = h5py.File(path, 'r')
		exp = agent.ExpCollector()
		exp.load(h5)
		poly_agent.train(exp)
		h5.close()
	new_model = "model_%s.h5" % epoch
	poly_agent.model.save_weights(new_model)
	return new_model

if __name__ == "__main__":
	import tensorflow as tf
	tf.device("/gpu:0")
	last_num = None
	if len(sys.argv) >= 2:
		last_num = int(sys.argv[1])
	last_model = None
	if last_num is not None:
		last_model = 'model_%d.h5' % last_num
	for epoch in range(last_num + 1, last_num + 2):
		agent1 = agent.Agent(Player.red, None)
		agent2 = agent.Agent(Player.black, None)
		if last_model:
			agent1.model.load_weights(last_model)
			agent2.model.load_weights(last_model)
		for round in range(200):
			print("===\n\nPlaying epoch: %d, round %d\n\n===" % (epoch, round))
			agent.self_play(str(epoch), round, agent1, agent2)
		last_model = train(str(epoch), last_model)
	# test('model_10.h5', 'model_9.h5')
    # for i in range(20):
    #     debug_play("model_12.h5", "model_1.h5")