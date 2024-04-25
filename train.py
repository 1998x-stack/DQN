from tqdm import trange

def train_dqn(agent, env, episodes):
    
    cumulative_rewards = []
    
    for episode in trange(episodes):
        state, info = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, trucated, info = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

            if done or trucated:
                break
        
        agent.model.reset_noise()
        
        cumulative_rewards.append(total_reward)
        if episode % 3 == 0:
            print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    return cumulative_rewards