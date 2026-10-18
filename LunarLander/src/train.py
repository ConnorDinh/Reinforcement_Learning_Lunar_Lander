import gymnasium as gym
from src.dqn_agent import DQNAgent
import torch
from pathlib import Path
import csv

def train():
    env = gym.make("LunarLander-v3")
    state, info = env.reset()
    state_dim, action_dim = state.shape[0], env.action_space.n
    agent = DQNAgent(state_dim, action_dim, 0.0005, 0.99, 100000, 64)
    total_steps = 0
    episode_rewards = []
    episode_losses = []
    model_dir = Path("results/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    for episode in range(1000):
        state, info = env.reset()
        episode_reward = 0
        lossess = []
        is_done = False
        while not is_done:
            epsilon = max(0.05, 1.0 - (1.0 - 0.05) * min(total_steps / 100000, 1))
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            is_done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, terminated)
            loss = agent.train_step()
            if loss is not None:
                lossess.append(loss)
            total_steps += 1
            if total_steps % 1000 == 0:
                agent.update_target_network()
            episode_reward += reward
            state = next_state
            if is_done:
                break
        episode_rewards.append(episode_reward)
        mean_loss = sum(lossess) / len(lossess) if lossess else float("nan")
        episode_losses.append(mean_loss)
        if (episode+1) % 100 == 0:
            print("Episode: ", episode)
            print("Reward: ", episode_reward)
            print("Epsilon: ", epsilon)
            recent_rewards = episode_rewards[-100:]
            print("Average reward: ", sum(recent_rewards)/len(recent_rewards))
            torch.save(agent.policy_network.state_dict(),model_dir / f"checkpoint_ep{episode}.pt")    

    torch.save(agent.policy_network.state_dict(), model_dir / "dqn_lunarlander.pt")
    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "training_metrics.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["episode", "reward", "mean_loss"])

        for episode, (reward, loss) in enumerate(zip(episode_rewards, episode_losses), start=1):
            writer.writerow([episode, reward, loss])
    env.close()

train()