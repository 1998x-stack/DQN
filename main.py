import concurrent.futures
import gym, json, torch, warnings
from loguru import logger
warnings.filterwarnings('ignore')

from buffer import ReplayBuffer
from agent import DQNAgent
from models import DQN, DQN_CNN
from config import CONFIG, env_info, ale_ram_envs
from train import train_dqn
from visualizer import visualize_cum_rewards


def main(env_name=None, cuda_device=None):
    config = CONFIG()
    config.random_seed(config.seed)
    config.env_name = env_name if env_name else config.env_name
    if cuda_device:
        config.DEVICE = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
    replay_buffer = ReplayBuffer(capacity=config.capacity)
    
    logger.add(f"logs/{config.env_name}.log", format="{time} - {level} - {message}", rotation="500 MB", compression="zip", enqueue=True)
    config.logger = logger
    env = gym.make(config.env_name)
    
    # Determining the sizes from the environment
    observation_space = env_info[config.env_name]['observation_space']
    action_space = env_info[config.env_name]['action_space']
    observation_space_discrete = env_info[config.env_name]['observation_space_discrete']
    action_space_discrete = env_info[config.env_name]['action_space_discrete']
    assert action_space_discrete, "Only discrete action spaces are supported"
    config.observation_space = action_space_discrete
    config.observation_space_discrete = observation_space_discrete
    config.print_all()
    
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
    cumulative_rewards = train_dqn(agent, env, config)
    with open(f'data/{config.suffix()}_rewards.json', 'w') as f:
        json.dump({config.env_name: cumulative_rewards}, f, ensure_ascii=False, indent=4)
        
        
    visualize_cum_rewards(cumulative_rewards, additional_info=config.suffix())
    visualize_cum_rewards(cumulative_rewards, additional_info=config.suffix(), smooth_rate=50)

def threaded_main(env_name, device_id):
    """线程函数，用于在指定设备上运行训练环境"""
    print(f"开始训练环境 {env_name} 在 CUDA 设备 {device_id}")
    main(env_name, cuda_device=device_id)
    print(f"训练环境 {env_name} 完成在设备 CUDA:{device_id}")

if __name__ == '__main__':
    # env_list = ['MountainCar-v0', 'FrozenLake-v1', 'FrozenLake8x8-v1', 'Breakout-ramDeterministic-v4', "ALE/Zaxxon-ram-v5", "ALE/VideoChess-ram-v5"]
    
    # for env_name in ale_ram_envs:
    #     main(env_name, cuda_device=1)


    # 使用ThreadPoolExecutor来管理线程
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # 提交任务到线程池
        futures = {executor.submit(threaded_main, env, device%3): env for device, env in enumerate(ale_ram_envs)}
        
        # 等待每个线程完成并获取结果
        for future in concurrent.futures.as_completed(futures):
            env = futures[future]
            try:
                future.result()  # 获取线程函数的返回结果
            except Exception as exc:
                print(f'{env} 生成异常: {exc}')
            else:
                print(f'{env} 完成无错误。')

    print("所有环境训练已完成。")
