from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    
    def __init__(self,
                 layers = [1024, 128, 10],
                 end_with_activation = False):
        super(MLP, self).__init__()
        
        self.end_with_activation = end_with_activation
        
        self.n_layers = len(layers) - 1
        
        self.input_size = layers[0]
        self.output_size = layers[-1]
        
        self.linear_layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(self.n_layers)])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(layers[i+1]) for i in range(self.n_layers - 1)])
        
        if self.end_with_activation:
            self.last_bn = nn.BatchNorm1d(layers[-1])
        
    def forward(self, x):
        for i in range(self.n_layers):
            x = self.linear_layers[i](x)
            
            if (i != self.n_layers - 1):
                x = self.bn_layers[i](x)
                x = F.relu(x)
            else:
                if self.end_with_activation:
                    x = self.last_bn(x)
                    x = F.relu(x)
            
        return x
    
    def load(self, module):
        self.load_state_dict(module.state_dict())