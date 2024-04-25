import gym, json
import warnings
warnings.filterwarnings('ignore')

from buffer import ReplayBuffer
from agent import DQNAgent
from models import DQN, DQN_CNN
from config import CONFIG, env_info
from train import train_dqn
from visualizer import visualize_cum_rewards


if __name__ == '__main__':
    config = CONFIG()
    env = gym.make(config.env_name)
    replay_buffer = ReplayBuffer(capacity=config.capacity)
    # Determining the sizes from the environment
    observation_space = env_info[config.env_name]['observation_space']
    action_space = env_info[config.env_name]['action_space']
    observation_space_discrete = env_info[config.env_name]['observation_space_discrete']
    action_space_discrete = env_info[config.env_name]['action_space_discrete']
    assert action_space_discrete, "Only discrete action spaces are supported"
    
    target_model = None
    # Instantiating the model
    if isinstance(observation_space, int):
        model = DQN(observation_space, action_space, config.hidden_size_list, dueling=config.dueling, is_noisy=config.is_noisy).to(config.DEVICE)
        if config.target:
            target_model = DQN(observation_space, action_space, config.hidden_size_list, dueling=config.dueling, is_noisy=config.is_noisy).to(config.DEVICE)
            target_model.load_state_dict(model.state_dict())
            target_model.eval()
    elif isinstance(observation_space, list):
        model = DQN_CNN(observation_space, action_space, config.channel_kernel_stride_list, dueling=config.dueling, is_noisy=config.is_noisy).to(config.DEVICE)
        if config.target:
            target_model = DQN_CNN(observation_space, action_space, config.channel_kernel_stride_list, dueling=config.dueling, is_noisy=config.is_noisy).to(config.DEVICE)
            target_model.load_state_dict(model.state_dict())
            target_model.eval()
    # Instantiating the agent
    agent = DQNAgent(model, replay_buffer, config, env, target_model=target_model)
    
    # Running the training function
    cumulative_rewards = train_dqn(agent, env, config.episodes, observation_space_discrete, observation_space)
    with open(f'data/{config.suffix()}_rewards.json', 'w') as f:
        json.dump({config.env_name: cumulative_rewards}, f, ensure_ascii=False, indent=4)
        
        
    visualize_cum_rewards(cumulative_rewards, additional_info=config.suffix())
    visualize_cum_rewards(cumulative_rewards, additional_info=config.suffix(), smooth_rate=2)
    visualize_cum_rewards(cumulative_rewards, additional_info=config.suffix(), smooth_rate=3)