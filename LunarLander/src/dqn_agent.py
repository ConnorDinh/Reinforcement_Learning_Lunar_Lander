from src.network import QNetwork
from src.replay_buffer import ReplayBuffer
import random
import torch

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, buffer_capacity, batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.action_dim).to(self.device)

        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), learning_rate)

        self.buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim -1)
        state = torch.as_tensor(state, dtype=torch.float32)

        state = state.unsqueeze(0)
        state = state.to(self.device)

        with torch.no_grad():
            q_values = self.policy_network.forward(state)

        return torch.argmax(q_values, dim=1).item()

    def store_transition(self, state, action, reward, next_state, is_done):
        self.buffer.push(state, action, reward, next_state, is_done)

    def train_step(self):
        if self.buffer.len() < self.batch_size:
            return
        s, a, r, n, i = self.buffer.sample(self.batch_size)
        states = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(a, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(n, dtype=torch.float32, device=self.device)
        is_dones = torch.as_tensor(i, dtype=torch.float32, device=self.device)

        predicted_q = self.policy_network.forward(states)

        selected_q = predicted_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network.forward(next_states)
            next_max_q = next_q.max(dim=1).values
            targets = rewards + self.gamma*(1-is_dones)*next_max_q

        loss_fn = torch.nn.HuberLoss()

        loss = loss_fn(selected_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
        