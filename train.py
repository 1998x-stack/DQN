from tqdm import trange
import numpy as np

def one_hot_state(state, observation_space):
    s = np.zeros(observation_space)
    s[state] = 1
    return s

def train_dqn(agent, env, config):
    
    cumulative_rewards = []
    
    for episode in trange(config.episodes):
        state, info = env.reset()
        state = one_hot_state(state, config.observation_space) if config.observation_space_discrete else state
        
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, trucated, info = env.step(action)
            next_state = one_hot_state(next_state, config.observation_space) if config.observation_space_discrete else next_state
            agent.buffer.push(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

            if done or trucated:
                break
        
        agent.model.reset_noise()
        
        cumulative_rewards.append(total_reward)
        if episode % config.print_interval == 0:
            config.logger.info(f"[ --- Episode: {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f} --- ]")
            print(f"[ --- Episode: {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f} --- ]")

    return cumulative_rewards