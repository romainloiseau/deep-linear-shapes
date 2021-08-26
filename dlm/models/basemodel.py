import torch
from torch import nn

import numpy as np

class BaseModel(nn.Module):
    
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv_type = "DENSE"
        
    def describe(self, writer = None, verbose = True):
        if verbose:
            model_name = type(self).__name__
            print("Model{}{}".format(" " * 20, model_name))
            
        for module in self.named_children():
            n_parameters = 0
            for block in module[1].parameters():
                n_parameters += np.prod(block.size())
            
            rg = [param.requires_grad for param in module[1].parameters()]
            rg = np.mean(rg) if len(rg) > 0 else 0.
            
            if rg == 0:
                rg = False
            elif rg == 1:
                rg = True
            else:
                rg = f"{int(100 * rg)}%"
    
            log = "{}{}{} parameters{}requiers gradients: {}".format(type(module[1]).__name__,
                                                                     " " * (25 - int(len(type(module[1]).__name__))),
                                                                     n_parameters,
                                                                     " " * (16 - len(str(n_parameters))),
                                                                     rg)
            if writer is not None:
                writer.add_text(module[0], log, 0)
            
            if verbose:
                print(log)
                
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path, strict = True):
        try:
            self.load_state_dict(torch.load(path)["model_state_dict"], strict = strict)
        except:
            state_dict = torch.load(path)["model_state_dict"]
            
            print("Trying to load only positions and alignments")

            todel = []
            for k in state_dict.keys():
                if "fields" in k or "PNet" in k:
                    todel.append(k)

            for k in todel:
                del state_dict[k]
                
            self.load_state_dict(state_dict, strict = False)
            
            