import sys

import numpy as np

import torch
from torch import nn
from torch import optim

from dlm.utils import copy_with_noise
from .decoders import MLP

from ..global_variables import DECODER_INIT_MEAN, DECODER_INIT_STD, COPY_NOISE_SCALE, NKP
import dlm.models.aligners as aligners

class LinearShapeModels(nn.ModuleList):
    
    def __init__(self, args):
        super(LinearShapeModels, self).__init__(args)
    
    def prototypes_as_clouds(self, q=.5, i = None):
        if i is not None:
            return torch.cat([
            lsm(
                torch.tensor([[]]),
                a = i * torch.cat(
                    [torch.tensor([torch.quantile(lsm.running_parameters.a[:, 0], q=.9)]).to(lsm.running_parameters.a.device), torch.quantile(lsm.running_parameters.a, q=.5, dim=0)]
                ).unsqueeze(0) + (1. - i) * torch.cat(
                    [torch.tensor([torch.quantile(lsm.running_parameters.a[:, 0], q=.1)]).to(lsm.running_parameters.a.device), torch.quantile(lsm.running_parameters.a, q=.5, dim=0)]
                ).unsqueeze(0),
                A = lsm.running_parameters.A.mean(0).unsqueeze(0)
            )[0][0].detach().unsqueeze(0) for lsm in self
        ], 0)
        return torch.cat([
            lsm(
                torch.tensor([[]]),
                a = torch.cat(
                    [torch.tensor([torch.quantile(lsm.running_parameters.a[:, 0], q=q)]).to(lsm.running_parameters.a.device), torch.quantile(lsm.running_parameters.a, q=.5, dim=0)]
                ).unsqueeze(0),
                A = lsm.running_parameters.A.mean(0).unsqueeze(0)
            )[0][0].detach().unsqueeze(0) for lsm in self
        ], 0)
        
    @property    
    def poss(self):
        return torch.cat([lsm.pos.unsqueeze(0) for lsm in self])
    
    @property    
    def ys(self):
        return torch.tensor([lsm.y for lsm in self]).to(self[0].y.device)
    
    @property
    def point_ys(self):
        return torch.cat([lsm.point_y.unsqueeze(0) for lsm in self], 0)
    
class Fields(nn.Module):
    def __init__(self,
                 D_parametric = 0,
                 architecture = None,
                 D_pointwise = 0,
                 size = None):
        super(Fields, self).__init__()
        
        self.D_parametric = D_parametric
        self.D_pointwise = D_pointwise
        
        if D_parametric > 0:
            architecture = [3] + architecture + [3]
            self.parametric = nn.ModuleList([
                nn.Sequential(
                    *[item for i in range(len(architecture)-1) for item in [nn.Conv1d(architecture[i], architecture[i+1], 1), nn.ReLU()]][:-1]
                )
            for _ in range(D_parametric)])
        
            for mlp in self.parametric:
                nn.init.normal_(list(mlp.children())[-1].weight,
                                mean=DECODER_INIT_MEAN,
                                std=DECODER_INIT_STD)
                nn.init.normal_(list(mlp.children())[-1].bias,
                                mean=DECODER_INIT_MEAN,
                                std=DECODER_INIT_STD)
            
        if D_pointwise > 0:
            self.pointwise = nn.Parameter(torch.zeros([D_pointwise] + size),
                                          requires_grad = True)
            
    def forward(self, x, D = None):
        
        if D is None:
            D = self.D_parametric + self.D_pointwise
            
        if D <= self.D_parametric:
            return torch.cat([self.parametric[d](x) for d in range(D)])
        else:
            return torch.cat([self.parametric[d](x) for d in range(self.D_parametric)] + [self.pointwise[d].unsqueeze(0) for d in range(D-self.D_parametric)])
        
class LinearShapeModel(nn.Module):
    
    def __init__(self, data, D_parametric, D_pointwise, alignment, decoders_layers, archi_parametric):
        super(LinearShapeModel, self).__init__()
        
        self.initialize_prototype(data)
            
        #Fields
        self.D = D_parametric + D_pointwise
        self.fields = Fields(
            D_parametric, archi_parametric,
            D_pointwise, list(self.pos.size())
        )
        
        self.PNet = MLP(decoders_layers + [self.D])
        
        #Alignment
        self.align = True
        self.Aligner = getattr(
            aligners,
            alignment + "Aligner")()
        self.ANet = MLP(decoders_layers + [self.Aligner.n_parameters])
        
        self.running_parameters = nn.ParameterDict({
            "x_latent": nn.Parameter(torch.zeros(512, self.ANet.input_size), requires_grad = False),
            "a": nn.Parameter(torch.zeros(512, self.D), requires_grad = False),
            "A": nn.Parameter(torch.zeros(512, self.Aligner.n_parameters), requires_grad = False)
        })
        
        self.initialize_decoders()
            
    def initialize_prototype(self, data):
        #Prototype
        if type(data) == int:
            self.pos = nn.Parameter(torch.zeros((3, data)), requires_grad = True)
        elif hasattr(data, "pos"):
            self.pos = nn.Parameter(torch.Tensor(data.pos).clone().permute(1, 0),
                                       requires_grad = True)
            if hasattr(data, "y"):
                self.y = nn.Parameter(data.y[0], requires_grad = False)
                
            if hasattr(data, "point_y"):
                assert data.point_y.size(0) == data.pos.size(0)
                self.point_y = nn.Parameter(data.point_y.clone(), requires_grad = False)
                
            if hasattr(data, "keypoints"):                
                choice = np.random.choice(self.pos.size(1), data.keypoints.size(0), replace = False)
                choice[NKP[data.y[0].item()]:] = -1
                self.idxkeypoints = nn.Parameter(torch.tensor(choice), requires_grad = False)
                    
        else:
            self.pos = nn.Parameter(data[:3].clone(), requires_grad = True)
            
        if not hasattr(self, "y"):
                self.y = nn.Parameter(torch.tensor(0), requires_grad = False)
    
    def forward(self, x_latent = None, A = None, a = None):
        rec = self.pos.unsqueeze(0)
        params = {
            "a": a if a is not None else self.PNet(x_latent),
            "A_reg_loss": torch.zeros(x_latent.size(0)).to(x_latent.device),
            "a_reg_loss": torch.zeros(x_latent.size(0)).to(x_latent.device)
        }
        
        if self.D > 0:
            fields = self.fields(rec, self.D).unsqueeze(0)
            a = params["a"][:, :self.D]
            rec = rec + (fields * a.unsqueeze(-1).unsqueeze(-1)).sum(1)
            params["a_reg_loss"] = (torch.abs(a) * torch.norm(fields.detach(), dim = 2).mean(-1)).sum(-1)
        else:
            rec = rec.expand(x_latent.size(0), -1, -1)
        
        params["A"] = A if A is not None else self.ANet(x_latent)    
        if self.align:
            rec, params["A_reg_loss"] = self.Aligner(rec, params["A"])
                                 
        return rec, params
        
    def initialize_decoders(self):
        nn.init.normal_(self.ANet.linear_layers[-1].weight,
                        mean=DECODER_INIT_MEAN,
                        std=DECODER_INIT_STD)
        nn.init.normal_(self.ANet.linear_layers[-1].bias,
                        mean=DECODER_INIT_MEAN,
                        std=DECODER_INIT_STD)
        
    @property
    def get_fields(self):
        return self.fields(self.pos.unsqueeze(0))
    
    def copy(self, lsm, optimizer = None):
        if type(lsm) != type(self):
            self.pos.data.copy_(lsm.pos.clone().permute(1, 0).to(self.pos.device))
            if hasattr(lsm, "y"):
                self.y.data.copy_(lsm.y[0].float().to(self.y.device))
            if hasattr(lsm, "point_y"):
                assert lsm.point_y.size(0) == lsm.pos.size(0)
                self.point_y.data.copy_(torch.Tensor(lsm.point_y.clone()).to(self.point_y.device))
                
        else:
            self.pos.data.copy_(copy_with_noise(lsm.pos,
                                                noise_scale = COPY_NOISE_SCALE))
            
            if hasattr(self, "fields"):
                self.fields.load_state_dict(lsm.fields.state_dict())
                    
            for attr in ["point_y", "idxkeypoints"]:
                if hasattr(self, attr):
                    #getattr(self, attr).data = getattr(lsm, attr).data.detach().clone()
                    getattr(self, attr).data.copy_(getattr(lsm, attr).data)
            if hasattr(self, "y"):
                self.y = lsm.y
            
            #Copy decoders
            if hasattr(self, "PNet"):
                self.PNet.load_state_dict(lsm.PNet.state_dict())
            self.ANet.load_state_dict(lsm.ANet.state_dict())

            #Copy running parameters
            for k in self.running_parameters.keys():
                self.running_parameters[k].data.copy_(lsm.running_parameters[k])

            if optimizer is not None:
                #Copying gradients
                if isinstance(optimizer, (optim.Adam,)):
                    if hasattr(self, "PNet"):
                        for param_i, param_j in zip(self.PNet.parameters(), lsm.PNet.parameters()):
                            if param_i in optimizer.state:
                                optimizer.state[param_i]['exp_avg'] = optimizer.state[param_j]['exp_avg']
                                optimizer.state[param_i]['exp_avg_sq'] = optimizer.state[param_j]['exp_avg_sq']
                                
                    for param_i, param_j in zip(self.ANet.parameters(), lsm.ANet.parameters()):
                        if param_i in optimizer.state:
                            optimizer.state[param_i]['exp_avg'] = optimizer.state[param_j]['exp_avg']
                            optimizer.state[param_i]['exp_avg_sq'] = optimizer.state[param_j]['exp_avg_sq']
                else:
                    raise NotImplementedError('Unknown optimizer: you should define how to reinstanciate statistics if any')
                
    def update_running_parameters(self, k, params):
        self.running_parameters[k].copy_(torch.cat([self.running_parameters[k][params.size(0):].detach(), params.detach()], 0))