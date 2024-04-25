import torch
import torch.nn as nn


# Defining the neural network for the DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size_list, dueling=False):
        super(DQN, self).__init__()
        self.dueling = dueling
        self.network = nn.Sequential()
        for i, hidden_size in enumerate(hidden_size_list):
            if i == 0:
                self.network.add_module('fc{}'.format(i), NoisyLinear(input_size, hidden_size))
            else:
                self.network.add_module('relu{}'.format(i), nn.ReLU())
                self.network.add_module('fc{}'.format(i), NoisyLinear(hidden_size_list[i - 1], hidden_size))
        self.network.add_module('relu{}'.format(len(hidden_size_list)), nn.ReLU())
        if not dueling:
            self.network.add_module('fc{}'.format(len(hidden_size_list)), NoisyLinear(hidden_size_list[-1], output_size))
        else:
            self.value_stream = nn.Sequential(
                NoisyLinear(hidden_size_list[-1], 12),
                nn.ReLU(),
                NoisyLinear(12, 1)
            )
            self.advantage_stream = nn.Sequential(
                NoisyLinear(hidden_size_list[-1], 12),
                nn.ReLU(),
                NoisyLinear(12, output_size)
            )
        
    def forward(self, x):
        out = self.network(x)
        if not self.dueling:
            return out
        value = self.value_stream(out)
        advantage = self.advantage_stream(out)
        return value + advantage - advantage.mean()
    
    def reset_noise(self):
        for name, module in self.named_children():
            # Check if the module is an instance of NoisyLinear
            if isinstance(module, NoisyLinear):
                module.reset_noise()
                        

class DQN_CNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size_list=[(32, 8, 3), (64, 6, 2), (1, 4, 1)], dueling=False):
        super(DQN_CNN, self).__init__()
        self.dueling = dueling
        c = input_size[2]  # Assuming channels-last format in input specification
        w = input_size[0]
        h = input_size[1]
        self.network = nn.Sequential()
        
        for i, (out_channels, kernel_size, stride) in enumerate(hidden_size_list):
            if i == 0:
                self.network.add_module('conv{}'.format(i), nn.Conv2d(c, out_channels, kernel_size, stride))
            else:
                self.network.add_module('relu{}'.format(i - 1), nn.ReLU())
                self.network.add_module('conv{}'.format(i), nn.Conv2d(hidden_size_list[i - 1][0], out_channels, kernel_size, stride))
            
            w = self.conv2d_size_out(w, kernel_size, stride)
            h = self.conv2d_size_out(h, kernel_size, stride)

        self.network.add_module('relu{}'.format(len(hidden_size_list)), nn.ReLU())
        linear_input_size = w * h * hidden_size_list[-1][0]
        assert linear_input_size > 0, f"The size of the output {(w, h)} of the convolutional layers is invalid"
        if not dueling:
            self.head = nn.Sequential(
                NoisyLinear(linear_input_size, linear_input_size//4),
                nn.ReLU(),
                NoisyLinear(linear_input_size//4, output_size),
            )
        if dueling:
            self.value_stream = nn.Sequential(
                NoisyLinear(linear_input_size, linear_input_size//4),
                nn.ReLU(),
                NoisyLinear(linear_input_size//4, 1)
            )
            self.advantage_stream = nn.Sequential(
                NoisyLinear(linear_input_size, linear_input_size//4),
                nn.ReLU(),
                NoisyLinear(linear_input_size//4, output_size)
            )

    @staticmethod
    def conv2d_size_out(size, kernel_size=3, stride=1):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Change from [N, W, H, C] to [N, C, H, W]
        x = self.network(x)
        x = x.reshape(x.size(0), -1)  # Use .reshape() to handle non-contiguous layout
        if not self.dueling:
            return self.head(x)
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean()
    
    def reset_noise(self):
        for name, module in self.named_children():
            # Check if the module is an instance of NoisyLinear
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / torch.sqrt(torch.tensor(self.in_features, dtype=torch.float32))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / torch.sqrt(torch.tensor(self.in_features, dtype=torch.float32)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / torch.sqrt(torch.tensor(self.out_features, dtype=torch.float32)))

    def reset_noise(self):
        torch.randn(self.weight_epsilon.size(), out=self.weight_epsilon)
        torch.randn(self.bias_epsilon.size(), out=self.bias_epsilon)

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return torch.addmm(bias, input, weight.t())
        else:
            return torch.addmm(self.bias_mu, input, self.weight_mu.t())
        

if __name__ == "__main__":
    # # Creating an instance of the model
    # input_size = 4
    # output_size = 2
    # hidden_size_list = [16, 16]
    # model = DQN(input_size, output_size, hidden_size_list)
    # print(model)
    # # Creating a dummy input
    # x = torch.randn(1, input_size)
    # output = model(x)
    # print(output)
    
    # Create an instance of the model CNN
    input_size = (250, 160, 3)
    output_size = 2
    model = DQN_CNN(input_size, output_size)
    print(model)
    # Creating a dummy input
    x = torch.randn(2, *input_size)
    output = model(x)
    print(output)