from src.replay_buffer import ReplayBuffer
import numpy as np
buffer = ReplayBuffer(capacity=100)

for i in range(10):
    state = np.ones(8) * i
    action = i % 4
    reward = float(i)
    next_state = state + 1
    done = False

    buffer.push(state, action, reward, next_state, done)

states, actions, rewards, next_states, dones = buffer.sample(4)

print("Buffer size:", buffer.len())
print("States shape:", states.shape)
print("Actions shape:", actions.shape)
print("Rewards shape:", rewards.shape)
print("Next states shape:", next_states.shape)
print("Dones shape:", dones.shape)