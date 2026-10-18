import gymnasium as gym
from src.network import QNetwork
import torch
import numpy as np

def evaluation(model_path, num_episodes=20, render=False):
    env = gym.make("LunarLander-v3", render_mode= "human"  if render else None)
    network = QNetwork(8, 4)
    model_weights = torch.load(model_path, weights_only=True)
    network.load_state_dict(model_weights)
    network.eval()

    rewards = []
    success_counter = 0
    for episode in range(num_episodes):
        state, info = env.reset(seed=42 + episode)
        episode_reward = 0
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                predict_q_values = network.forward(state_tensor)
                action = torch.argmax(predict_q_values, dim=1).item()
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
            state = next_state
        rewards.append(episode_reward)
        if episode_reward >= 200:
            success_counter += 1
    print("Mean rewards: ", sum(rewards)/len(rewards))
    print("Standard deviation: ", np.std(rewards))
    print("Percentage of episodes with reward >= 200: ", success_counter/len(rewards) * 100, "%")
    env.close()

if __name__ == "__main__":
    models = [
        "results/models/checkpoint_ep799.pt",
        "results/models/checkpoint_ep899.pt",
        "results/models/dqn_lunarlander.pt"
    ]

    for model_path in models:
        print(f"\nEvaluating: {model_path}")
        evaluation(model_path, num_episodes=20)