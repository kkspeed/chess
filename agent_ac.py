import numpy as np
from keras.optimizers import Adam

from chess_types import Player, GameState
import encoder
import model_ac


class AcExpCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.estimated_rewards = []
        self.advantages = []

    def record(self, state, action, estimated_reward):
        self.states.append(state)
        self.actions.append(action)
        self.estimated_rewards.append(estimated_reward)

    def assign_reward(self, reward):
        self.advantages = [reward - e for e in self.estimated_rewards]

    def save(self, h5file):
        h5file['experience'].create_dataset(
            'states', data=np.array(self.states))
        h5file['experience'].create_dataset(
            'actions', data=np.array(self.actions))
        h5file['experience'].create_dataset(
            'advantages', data=np.array(self.advantages))
        h5file['estimated_rewards'].create_dataset(
            'estimated_rewards', data=np.array(self.estimated_rewards))

    def load(self, h5file):
        experience = h5file.get('experience')
        self.states = experience['states'][:]
        self.actions = experience['actions'][:]
        self.estimated_rewards = experience['estimated_rewards'][:]
        self.advantages = experience['advantages'][:]


class AcAgent:
    def __init__(self, player: Player):
        self.player = player
        self.encoder = encoder.SimpleEncoder()
        self.model = model_ac.create_model()
        self.collector = None
        self.encountered = set()

    def train_batch(self, states, policy_targets, value_targets):
        self.model.compile(optimizer=Adam(lr=0.001), loss=[
                           'categorical_crossentropy', 'mse'])
        self.model.fit(
            states, [policy_targets, value_targets], batch_size=4000, epochs=10)
