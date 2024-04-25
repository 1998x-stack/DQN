import torch, json, argparse

with open('data/latest_env.json', 'r') as f:
    env_info = json.load(f)

class CONFIG:
    def __init__(self) -> None:
        self.env_name = 'YarsRevenge-ramNoFrameskip-v4'
        self.hidden_size_list = [16, 16] # Hidden layer sizes
        # self.env_name = 'ALE/Adventure-v5'
        # self.env_name = 'ALE/Adventure-ram-v5'
        self.channel_kernel_stride_list = [(32, 8, 3), (64, 8, 3), (1, 3, 2)]
        
        self.capacity = 1000 # Buffer capacity
        self.batch_size = 32 # Batch size
        self.discount_factor = 0.99 # Discount factor
        self.lr = 0.001 # Learning rate
        self.tau = 1e-3 # Tau value for soft update
        self.l2_reg = 1e-3 # L2 regularization factor
        
        self.episodes = 2000 # Number of episodes
        self.epsilon = 1.0 # Initial epsilon value
        self.epsilon_decay = 1 - 1e-4 # Epsilon decay rate
        self.epsilon_min = 0.01 # Minimum epsilon value
        
        self.dueling = True # Dueling DQN
        self.target = True # Target DQN
        self.is_noisy = True # Noisy DQN
        self.DEVICE = torch.device(f'cuda:2' if torch.cuda.is_available() else 'cpu') # Device to use for training
        self.print_all()
        
    def print_all(self):
        print("CONFIG Parameters:")
        print("-------------------"*4)
        for key, value in self.__dict__.items():
            print(f"||  {key:<30} | {str(value):<35}  ||")
        print("-------------------"*4)
    
    def suffix(self):
        suf = self.env_name.split('/')[-1].split('-')[0]
        suf += '_Dueling' if self.dueling else ''
        suf += f'_Target{self.tau}' if self.target else ''
        suf += '_Noisy' if self.is_noisy else ''
        suf += '_r' + str(self.l2_reg) if self.l2_reg > 0 else ''
        suf += '_bs{}'.format(self.batch_size)
        
        return suf

def parse_arguments(config):
    parser = argparse.ArgumentParser(description='DQN Configuration')
    
    # Environment
    parser.add_argument('--env_name', type=str, default=config.env_name, help='Name of the environment')
    
    # Network architecture
    parser.add_argument('--hidden_size_list', nargs='+', type=int, default=config.hidden_size_list, help='List of hidden layer sizes')
    parser.add_argument('--channel_kernel_stride_list', nargs='+', type=int, default=config.channel_kernel_stride_list, help='List of tuples containing (channel, kernel_size, stride) for convolutional layers')
    
    # Hyperparameters
    parser.add_argument('--capacity', type=int, default=config.capacity, help='Buffer capacity')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='Batch size')
    parser.add_argument('--discount_factor', type=float, default=config.discount_factor, help='Discount factor')
    parser.add_argument('--lr', type=float, default=config.lr, help='Learning rate')
    parser.add_argument('--tau', type=float, default=config.tau, help='Tau value for soft update')
    parser.add_argument('--l2_reg', type=float, default=config.l2_reg, help='L2 regularization factor')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=config.episodes, help='Number of episodes')
    parser.add_argument('--epsilon', type=float, default=config.epsilon, help='Initial epsilon value')
    parser.add_argument('--epsilon_decay', type=float, default=config.epsilon_decay, help='Epsilon decay rate')
    parser.add_argument('--epsilon_min', type=float, default=config.epsilon_min, help='Minimum epsilon value')
    
    # DQN options
    parser.add_argument('--dueling', action='store_true', default=config.dueling, help='Enable dueling DQN')
    parser.add_argument('--target', action='store_true', default=config.target, help='Enable target DQN')
    
    return parser.parse_args()

if __name__ == "__main__":
    config = CONFIG()