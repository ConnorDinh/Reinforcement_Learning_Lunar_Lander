from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, is_done):
        experience = (state, action, reward, next_state, is_done)
        self.deque.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.deque, k=batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        is_dones = []
        for s, a, r, n, i in experiences:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(n)
            is_dones.append(i)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        is_dones = np.array(is_dones)

        return states, actions, rewards, next_states, is_dones

    def len(self):
        return len(self.deque)
        