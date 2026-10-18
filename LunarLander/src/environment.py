import gymnasium as gym

def create_environment(render_mode=None):
    env = gym.make("LunarLander-v3", render_mode= render_mode)
    return env

def inspect_environment(env):
    observation_space = env.observation_space
    action_space = env.action_space
    print("Observation space: ", observation_space)
    print("Action space: ", action_space)
    state, info = env.reset()
    print("Initial state: ", state)
    print("Number of features: ", state.shape[0])
    print("Number of possible actions: ", env.action_space.n)

def run_random_agent(env, num_episodes):
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        is_done = False
        while not is_done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            is_done = terminated or truncated
            state = next_state
        print("Episode number: ", episode)
        print("Total reward: ", total_reward)

def main():
    env = create_environment("human")
    inspect_environment(env)
    run_random_agent(env, 5)
    env.close()

main()